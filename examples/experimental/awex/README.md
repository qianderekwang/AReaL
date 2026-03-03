# Experimental Awex GSM8K Example

This folder provides a minimal AWEX config for GSM8K GRPO in single-controller
mode.

## Files

| File | Purpose |
| --- | --- |
| `gsm8k_grpo_awex_sample.yaml` | Minimal config for local validation on 2 GPUs. |

## Quickstart

Run from repo root (`AReaL/`).

```bash
python examples/math/gsm8k_rl.py \
  --config examples/experimental/awex/gsm8k_grpo_awex_sample.yaml
```

## Runtime behavior

- If Awex is enabled and `awex.meta_server_addr` is empty or `auto`,
  `PPOTrainer` starts a local Awex meta server automatically before rollout
  initialization in single-controller mode.
- In SPMD mode, set `awex.meta_server_addr` explicitly instead of relying on
  auto-start.

## Environment variables

| Env var | Meaning |
| --- | --- |
| `AREAL_AWEX_META_SERVER_ADDR` | Preferred external meta server addr (`ip:port`). If unset, the trainer follows `awex.meta_server_addr`. |
| `AWEX_META_SERVER_ADDR` | Generic external meta server addr override. |

## Notes

- Use `actor.path`, `ref.path`, `tokenizer_path`, and `vllm.model` in the yaml
  to point to the model you want to validate.
- `gsm8k_grpo_awex_sample.yaml` targets a small local GPU validation setup.

## Adapting the sample for NPU

This sample can be adapted to NPU, but the defaults in
`gsm8k_grpo_awex_sample.yaml` are GPU-oriented.

The minimum config changes are:

- Keep `actor.weight_update_mode: awex`.
- Change the Awex backend:
  - `awex.comm_backend: hccl`
  - `awex.weights_exchange_ipc_backend: cpu`

Example Awex diff:

```yaml
awex:
  meta_server_addr: auto
  comm_backend: hccl
  weights_exchange_ipc_backend: cpu
  weights_comm_nccl_group_size: 1
  weights_validation_steps: 0
  validate_weights_every_n_steps: 1
```

If you run in SPMD mode instead of the single-controller sample here, do not
rely on `meta_server_addr: auto`; set `awex.meta_server_addr` explicitly.

For other NPU-specific runtime settings, use `examples/math/gsm8k_grpo_npu.yaml` as
the baseline reference, then apply the AWEX-specific settings from
`gsm8k_grpo_awex_sample.yaml` on top of it.
