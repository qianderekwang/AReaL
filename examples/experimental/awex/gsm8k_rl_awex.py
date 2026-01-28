import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer


DEFAULT_DENSE_HF_ID = "Qwen/Qwen3-0.6B"
DEFAULT_DENSE_LOCAL_PATH = "/home/model/Qwen3-0.6B"
DEFAULT_MOE_HF_ID = "Qwen/Qwen3-30B-A3B"
DEFAULT_MOE_LOCAL_PATH = "/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l1-e2"


def _resolve_model_path(model_kind: str, default_path: str, default_hf_id: str) -> str:
    env_key = (
        "AREAL_GSM8K_DENSE_MODEL_PATH"
        if model_kind == "dense"
        else "AREAL_GSM8K_MOE_MODEL_PATH"
    )
    env_path = os.environ.get(env_key)
    if env_path:
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f"{env_key} not found: {env_path}. Provide a valid local path."
            )
        return env_path
    if os.path.exists(default_path):
        return default_path
    allow_download = os.environ.get("AREAL_GSM8K_ALLOW_HF_DOWNLOAD") == "1"
    if allow_download:
        return default_hf_id
    raise FileNotFoundError(
        f"Model path not found for {model_kind}. Set {env_key} or "
        "AREAL_GSM8K_ALLOW_HF_DOWNLOAD=1."
    )


def _apply_model_overrides(config) -> None:
    model_kind = os.environ.get("AREAL_GSM8K_MODEL", "").strip().lower()
    if not model_kind:
        return
    if model_kind not in {"dense", "moe"}:
        raise ValueError(
            f"AREAL_GSM8K_MODEL must be 'dense' or 'moe', got {model_kind}."
        )

    if model_kind == "dense":
        model_path = _resolve_model_path(
            "dense", DEFAULT_DENSE_LOCAL_PATH, DEFAULT_DENSE_HF_ID
        )
    else:
        model_path = _resolve_model_path(
            "moe", DEFAULT_MOE_LOCAL_PATH, DEFAULT_MOE_HF_ID
        )

    config.actor.path = model_path
    if getattr(config, "ref", None) is not None:
        config.ref.path = model_path
    config.tokenizer_path = model_path
    config.vllm.model = model_path
    config.sglang.model_path = model_path

    if model_kind == "moe":
        _apply_moe_overrides(config)


def _apply_moe_overrides(config) -> None:
    # Reduce memory footprint for small MoE validation runs on 2x16GB GPUs.
    max_new_tokens = 32
    max_prompt_len = 128
    max_model_len = 256
    max_tokens_per_mb = 256

    config.gconfig = config.gconfig.new(
        max_new_tokens=min(config.gconfig.max_new_tokens, max_new_tokens),
        max_tokens=min(config.gconfig.max_tokens, max_model_len),
    )
    config.actor.max_new_tokens = config.gconfig.max_new_tokens
    config.actor.gradient_checkpointing = True
    config.actor.mb_spec.max_tokens_per_mb = min(
        config.actor.mb_spec.max_tokens_per_mb or max_tokens_per_mb,
        max_tokens_per_mb,
    )
    if getattr(config, "ref", None) is not None:
        config.ref.mb_spec.max_tokens_per_mb = min(
            config.ref.mb_spec.max_tokens_per_mb or max_tokens_per_mb,
            max_tokens_per_mb,
        )

    if getattr(config, "train_dataset", None) is not None:
        config.train_dataset.batch_size = min(config.train_dataset.batch_size, 1)
        if getattr(config.train_dataset, "max_length", None) is not None:
            config.train_dataset.max_length = min(
                config.train_dataset.max_length, max_prompt_len
            )
    if getattr(config, "valid_dataset", None) is not None:
        config.valid_dataset.batch_size = min(config.valid_dataset.batch_size, 1)

    if getattr(config, "rollout", None) is not None:
        config.rollout.max_concurrent_rollouts = min(
            config.rollout.max_concurrent_rollouts, 1
        )
        config.rollout.consumer_batch_size = config.train_dataset.batch_size

    config.vllm.max_model_len = min(
        config.vllm.max_model_len or max_model_len, max_model_len
    )
    config.vllm.gpu_memory_utilization = min(config.vllm.gpu_memory_utilization, 0.6)


def _ensure_awex_runtime(config) -> None:
    if config.actor.weight_update_mode != "awex":
        return
    os.environ.setdefault("VLLM_PLUGINS", "awex_adapter")
    meta_addr = os.environ.get("AREAL_GSM8K_AWEX_META_SERVER")
    if not meta_addr:
        meta_addr = config.awex.meta_server_addr
    if not meta_addr or meta_addr.lower() == "auto":
        from awex.meta.meta_server import start_meta_server

        meta_ip, meta_port = start_meta_server()
        meta_addr = f"{meta_ip}:{meta_port}"
    config.awex.meta_server_addr = meta_addr


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    _apply_model_overrides(config)
    _ensure_awex_runtime(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
