"""Integration test for Awex weight exchange between Megatron and vLLM."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    InferenceEngineConfig,
    MegatronEngineConfig,
    TrainEngineConfig,
    vLLMConfig,
)
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.megatron_engine import MegatronEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.tests.utils import get_model_path
from areal.utils import network
from areal.utils.pkg_version import is_available
from areal.utils.proc import kill_process_tree

IS_VLLM_INSTALLED = is_available("vllm")
IS_AWEX_INSTALLED = is_available("awex")

DENSE_HF_ID = "Qwen/Qwen3-0.6B"
DENSE_LOCAL_PATH = "/home/model/Qwen3-0.6B/"
MOE_LOCAL_PATH = "/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l2-e8"


def _select_devices(tp_size: int = 1):
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env:
        visible_devices = [int(x) for x in visible_env.split(",") if x.strip()]
    else:
        visible_devices = list(range(torch.cuda.device_count()))

    need = 1 + tp_size
    if len(visible_devices) < need:
        pytest.skip(
            f"Need at least {need} CUDA devices (1 for Megatron + {tp_size} for vLLM). "
            f"Found {len(visible_devices)}."
        )

    megatron_device = visible_devices[0]
    vllm_devices = visible_devices[1 : 1 + tp_size]
    return megatron_device, vllm_devices


@contextmanager
def _temp_env(overrides: dict[str, str]):
    old = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_awex_integration(result_queue, model_path: str):
    def _safe_destroy(fn, name: str, timeout: int = 10) -> None:
        done = threading.Event()

        def _run():
            try:
                fn()
            finally:
                done.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        if not done.wait(timeout):
            print(f"[awex-test] {name} timed out after {timeout}s.", flush=True)

    try:
        _, vllm_devices = _select_devices(tp_size=1)

        # Start Awex meta server
        from awex.meta.meta_server import start_meta_server, stop_meta_server

        meta_ip, meta_port = start_meta_server()
        meta_server_addr = f"{meta_ip}:{meta_port}"

        # Launch vLLM server with Awex plugin enabled (real weights + validation).
        enable_validation = True

        vllm_config = vLLMConfig(
            skip_tokenizer_init=False,
            model=model_path,
            gpu_memory_utilization=0.6,
            max_num_seqs=1,
            max_model_len=128,
            enforce_eager=True,
            load_format="auto",
        )
        vllm_args = vLLMConfig.build_args(
            vllm_config=vllm_config,
            tp_size=1,
            pp_size=1,
        )

        vllm_env = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, vllm_devices)),
            "VLLM_PLUGINS": "awex_adapter",
        }

        inf_engine = None
        train_engine = None

        try:
            temp_config = InferenceEngineConfig(
                experiment_name="test_awex_megatron_vllm",
                trial_name="trial_0",
                setup_timeout=360,
                request_timeout=60,
            )
            inf_engine = RemotevLLMEngine(temp_config)
            with _temp_env(vllm_env):
                inf_engine.launch_server(vllm_args)
            inf_engine.initialize()
            inf_engine.set_version(1)

            # Initialize Megatron engine
            dist_port = network.find_free_ports(1)[0]
            os.environ.update(
                {
                    "WORLD_SIZE": "1",
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(dist_port),
                }
            )

            # Pin Megatron to the first visible GPU
            torch.cuda.set_device(0)

            train_config = TrainEngineConfig(
                experiment_name="test_awex_megatron_vllm",
                trial_name="trial_0",
                path=model_path,
                init_from_scratch=False,
                optimizer=None,
                megatron=MegatronEngineConfig(),
            )
            train_engine = MegatronEngine(train_config)
            alloc_mode = AllocationMode.from_str("d1p1t1")
            train_engine.create_process_group(alloc_mode.train)
            ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=2)
            train_engine.initialize(addr=None, ft_spec=ft_spec)
            train_engine.set_version(1)

            update_meta = WeightUpdateMeta.from_awex(
                meta_server_addr=meta_server_addr,
                comm_backend="nccl",
                weights_validation_steps=1 if enable_validation else 0,
                validate_weights_every_n_steps=1,
                enable_debug_mode=enable_validation,
                debug_mode_config=(
                    {"raise_on_validation_fail": True} if enable_validation else {}
                ),
            )

            train_engine.connect_engine(inf_engine, update_meta)
            train_engine.update_weights(update_meta)

            result_queue.put(("ok", None))
        finally:
            if train_engine is not None:
                _safe_destroy(train_engine.destroy, "train_engine.destroy")
            if inf_engine is not None:
                _safe_destroy(inf_engine.destroy, "inf_engine.destroy")
            stop_meta_server()
            # Ensure any leftover subprocesses are cleaned up before returning.
            kill_process_tree(os.getpid(), include_parent=False, graceful=False)
            # Avoid blocking teardown in the worker process; exit promptly.
    except Exception as exc:
        result_queue.put(("error", repr(exc)))
        raise


def _resolve_moe_model_path() -> str:
    env_path = os.environ.get("AREAL_AWEX_MOE_MODEL_PATH")
    if env_path:
        if not os.path.exists(env_path):
            pytest.skip(
                f"AREAL_AWEX_MOE_MODEL_PATH not found: {env_path}. "
                "Set it to a reduced MoE checkpoint path."
            )
        return env_path
    if os.path.exists(MOE_LOCAL_PATH):
        return MOE_LOCAL_PATH
    pytest.skip(
        "Reduced MoE model not found locally. Build it with "
        "`areal/tests/experimental/awex/build_reduced_qwen3_moe.py` and set "
        "AREAL_AWEX_MOE_MODEL_PATH."
    )


def _resolve_dense_model_path() -> str:
    env_path = os.environ.get("AREAL_AWEX_DENSE_MODEL_PATH")
    if env_path:
        if not os.path.exists(env_path):
            pytest.skip(
                f"AREAL_AWEX_DENSE_MODEL_PATH not found: {env_path}. "
                "Set it to a valid local path or enable download."
            )
        return env_path
    if os.path.exists(DENSE_LOCAL_PATH):
        return DENSE_LOCAL_PATH
    if os.environ.get("AREAL_AWEX_ALLOW_HF_DOWNLOAD") == "1":
        return get_model_path(DENSE_LOCAL_PATH, DENSE_HF_ID)
    pytest.skip(
        "Dense model not found locally. Set AREAL_AWEX_DENSE_MODEL_PATH or "
        "AREAL_AWEX_ALLOW_HF_DOWNLOAD=1 to download."
    )


@pytest.mark.slow
@pytest.mark.multi_gpu
def test_awex_megatron_to_vllm_nccl(tmp_path_factory):
    if not IS_VLLM_INSTALLED:
        pytest.skip("vLLM is not installed")
    if not IS_AWEX_INSTALLED:
        pytest.skip("awex is not installed")

    timeout_seconds = 60
    model_kind = os.environ.get("AREAL_AWEX_MODEL", "dense").lower()
    if model_kind == "moe":
        model_path = _resolve_moe_model_path()
    elif model_kind == "dense":
        model_path = _resolve_dense_model_path()
    else:
        raise ValueError(
            f"Unknown AREAL_AWEX_MODEL value: {model_kind}. Use 'dense' or 'moe'."
        )
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_awex_integration,
        args=(result_queue, model_path),
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        kill_process_tree(proc.pid, graceful=False)
        proc.join(5)
        pytest.fail(f"Awex integration test exceeded {timeout_seconds}s timeout.")

    status = None
    detail = None
    try:
        status, detail = result_queue.get_nowait()
    except queue.Empty:
        status, detail = ("error", f"Worker exited with code {proc.exitcode}.")

    if status != "ok":
        pytest.fail(f"Awex integration test failed: {detail}")
