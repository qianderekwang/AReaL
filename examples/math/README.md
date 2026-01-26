## Hyper-parameters for GSM8K Finetuning on Qwen2.5-1.5b-Instruct

The hyperparameters given in gsm8k_grpo.yaml is the set that we found to achieve the
highest max `grpo-eval/task_reward/avg` during training for `Qwen2.5-1.5b-Instruct`. You
are free to try out more of the hyperparameters listed below!

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

### Other Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)

## Awex GSM8K sample (single-controller)

This repo defaults to single-controller mode (no launcher needed). A minimal Awex
sample config is provided at `examples/math/gsm8k_grpo_awex_sample.yaml`, and the
Awex-tuned script lives in `examples/math/gsm8k_rl_awex.py` (so the original
`gsm8k_rl.py` stays generic).

Run dense:
```
AREAL_GSM8K_MODEL=dense python examples/math/gsm8k_rl_awex.py --config examples/math/gsm8k_grpo_awex_sample.yaml
```

Run MoE (reduced checkpoint recommended for 16GB GPUs):
```
AREAL_GSM8K_MODEL=moe python examples/math/gsm8k_rl_awex.py --config examples/math/gsm8k_grpo_awex_sample.yaml
```

Notes:
- The Awex meta server is started inside `gsm8k_rl_awex.py` when
  `awex.meta_server_addr` is empty or set to `auto`.
- Override model paths with `AREAL_GSM8K_DENSE_MODEL_PATH` or
  `AREAL_GSM8K_MOE_MODEL_PATH` if needed.
- For MoE, point `AREAL_GSM8K_MOE_MODEL_PATH` to a reduced checkpoint created with
  `areal/tests/experimental/awex/build_reduced_qwen3_moe.py`. For 2x16GB GPUs,
  a smaller variant like `--num-layers 1 --num-experts 2 --num-experts-per-tok 2`
  is recommended.
- For MoE, `gsm8k_rl_awex.py` applies low-memory overrides (batch size 1, max_new_tokens 32,
  max_model_len 256, gradient checkpointing on). Tune in the script if needed.
