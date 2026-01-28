# Experimental Awex GSM8K Example

This folder provides a minimal GRPO example using Awex weight exchange in
single-controller mode.

## Files

| File | Purpose |
| --- | --- |
| `gsm8k_rl_awex.py` | Awex-enabled GSM8K training entry. It resolves model path by env vars and can auto-start Awex meta server. |
| `gsm8k_grpo_awex_sample.yaml` | Minimal config for local validation on 2 GPUs. |

## Quickstart

Run from repo root (`AReaL/`).

### Dense model

```bash
AREAL_GSM8K_MODEL=dense \
python examples/experimental/awex/gsm8k_rl_awex.py \
  --config examples/experimental/awex/gsm8k_grpo_awex_sample.yaml
```

### MoE model

Use a reduced checkpoint first (recommended for 16GB-class GPUs), then run:

```bash
AREAL_GSM8K_MODEL=moe \
AREAL_GSM8K_MOE_MODEL_PATH=/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l1-e2 \
python examples/experimental/awex/gsm8k_rl_awex.py \
  --config examples/experimental/awex/gsm8k_grpo_awex_sample.yaml
```

You can create reduced MoE checkpoints with:
`areal/tests/experimental/awex/build_reduced_qwen3_moe.py`.

## Runtime behavior in `gsm8k_rl_awex.py`

- If `AREAL_GSM8K_MODEL` is set to `dense` or `moe`, the script rewrites
  `actor/ref/vllm/tokenizer` model paths consistently.
- If Awex is enabled and `awex.meta_server_addr` is empty or `auto`,
  the script starts a local Awex meta server automatically.
- In `moe` mode, the script applies low-memory overrides (smaller batch/token
  limits and lower vLLM memory utilization) for local validation runs.

## Environment variables

| Env var | Meaning |
| --- | --- |
| `AREAL_GSM8K_MODEL` | `dense` or `moe`. |
| `AREAL_GSM8K_DENSE_MODEL_PATH` | Override local dense model path. |
| `AREAL_GSM8K_MOE_MODEL_PATH` | Override local reduced MoE path. |
| `AREAL_GSM8K_ALLOW_HF_DOWNLOAD=1` | Allow fallback to HF model ID when local path is missing. |
| `AREAL_GSM8K_AWEX_META_SERVER` | Optional external meta server addr (`ip:port`). If unset, script auto-starts one. |

## Notes

- Default paths in script:
  - Dense: `/home/model/Qwen3-0.6B`
  - MoE: `/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l1-e2`
- If local paths do not exist, either set the env path explicitly or enable
  `AREAL_GSM8K_ALLOW_HF_DOWNLOAD=1`.
