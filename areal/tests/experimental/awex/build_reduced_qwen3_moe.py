#!/usr/bin/env python3
"""Build a reduced Qwen3 MoE HF checkpoint by slicing layers and experts.

This keeps hidden size unchanged and reduces:
- num_hidden_layers (keep the first N layers)
- num_experts (keep the first K experts in each MoE layer)

This is intended for Awex integration tests that need real weight loading
on limited GPU memory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
GATE_SUFFIX = ".mlp.gate.weight"

DTYPE_SIZES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _parse_layer(key: str) -> int | None:
    match = LAYER_RE.match(key)
    if not match:
        return None
    return int(match.group(1))


def _parse_expert(key: str) -> int | None:
    match = EXPERT_RE.search(key)
    if not match:
        return None
    return int(match.group(1))


def _should_keep(key: str, num_layers: int, num_experts: int) -> bool:
    layer = _parse_layer(key)
    if layer is None:
        return True
    if layer >= num_layers:
        return False
    expert = _parse_expert(key)
    if expert is not None and expert >= num_experts:
        return False
    return True


def _tensor_nbytes(shape: Iterable[int], dtype: str) -> int:
    size = DTYPE_SIZES.get(dtype)
    if size is None:
        raise ValueError(f"Unsupported dtype {dtype}")
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * size


def _plan_shards(
    keys: list[str],
    weight_map: dict[str, str],
    num_experts: int,
    max_shard_bytes: int,
) -> list[list[str]]:
    sizes: dict[str, int] = {}
    keys_by_file: dict[str, list[str]] = {}
    for key in keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)

    for filename, file_keys in keys_by_file.items():
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in file_keys:
                sl = handle.get_slice(key)
                shape = list(sl.get_shape())
                dtype = sl.get_dtype()
                if key.endswith(GATE_SUFFIX):
                    shape[0] = min(shape[0], num_experts)
                sizes[key] = _tensor_nbytes(shape, dtype)

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in keys:
        size = sizes[key]
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def _copy_support_files(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        if item.name.startswith("model-") or item.name == "model.safetensors.index.json":
            continue
        if item.is_dir():
            if item.name.startswith("."):
                continue
            shutil.copytree(item, dst_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_dir / item.name)


def _update_config(path: Path, num_layers: int, num_experts: int, num_experts_per_tok: int) -> None:
    cfg = json.loads(path.read_text())
    cfg["num_hidden_layers"] = num_layers
    cfg["num_experts"] = num_experts
    cfg["num_experts_per_tok"] = min(num_experts_per_tok, num_experts)
    if "max_window_layers" in cfg:
        cfg["max_window_layers"] = num_layers
    if "mlp_only_layers" in cfg and isinstance(cfg["mlp_only_layers"], list):
        cfg["mlp_only_layers"] = [i for i in cfg["mlp_only_layers"] if i < num_layers]
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source HF model directory.")
    parser.add_argument("--output", help="Output directory for reduced checkpoint.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers to keep.")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts to keep.")
    parser.add_argument(
        "--num-experts-per-tok",
        type=int,
        default=2,
        help="Number of experts per token (will be clipped to num_experts).",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=2.0,
        help="Max shard size in GB for output safetensors.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory.")
    args = parser.parse_args()

    src_dir = Path(args.input).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Input model path not found: {src_dir}")

    output_dir = Path(
        args.output
        if args.output
        else f"{src_dir}-reduced-l{args.num_layers}-e{args.num_experts}"
    ).resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    selected_keys = [
        k
        for k in sorted(weight_map.keys())
        if _should_keep(k, args.num_layers, args.num_experts)
    ]
    if not selected_keys:
        raise RuntimeError("No tensors selected. Check num_layers/num_experts.")

    max_shard_bytes = int(args.max_shard_size_gb * (1 << 30))
    shard_plan = _plan_shards(
        selected_keys,
        {k: str(src_dir / v) for k, v in weight_map.items()},
        args.num_experts,
        max_shard_bytes,
    )
    num_shards = len(shard_plan)
    print(
        f"[reduce] Selected {len(selected_keys)} tensors into {num_shards} shard(s)."
    )

    weight_map_out: dict[str, str] = {}
    total_size = 0
    file_handles: dict[str, safe_open] = {}

    try:
        for shard_idx, keys in enumerate(shard_plan, start=1):
            shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            shard_path = output_dir / shard_name
            tensors: dict[str, torch.Tensor] = {}
            for key in keys:
                filename = str(src_dir / weight_map[key])
                if filename not in file_handles:
                    file_handles[filename] = safe_open(
                        filename, framework="pt", device="cpu"
                    )
                tensor = file_handles[filename].get_tensor(key)
                if key.endswith(GATE_SUFFIX):
                    tensor = tensor[: args.num_experts].contiguous()
                tensors[key] = tensor
                weight_map_out[key] = shard_name
                total_size += tensor.numel() * tensor.element_size()
            save_file(tensors, shard_path)
            print(f"[reduce] Wrote {shard_path} with {len(tensors)} tensors.")
    finally:
        for handle in file_handles.values():
            handle.__exit__(None, None, None)

    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_out, indent=2) + "\n"
    )

    _copy_support_files(src_dir, output_dir)
    for cfg_name in ("config.json", "config_1m.json"):
        cfg_path = output_dir / cfg_name
        if cfg_path.exists():
            _update_config(
                cfg_path, args.num_layers, args.num_experts, args.num_experts_per_tok
            )
    print(f"[reduce] Reduced checkpoint written to: {output_dir}")


if __name__ == "__main__":
    main()
