# Eval Only

This directory provides standardized `eval_only` entrypoints that submit validation
or test samples to an eval rollout controller and dump trajectories into
`eval-rollout/` without running PPO training.

## Entry Points

### General Workflow

Use this for standard rollout workflows such as:

- `areal.workflow.rlvr.RLVRWorkflow`
- `areal.workflow.vision_rlvr.VisionRLVRWorkflow`

```bash
python -m offline_eval.eval_only.general_eval \
  --config offline_eval/eval_only/examples/general_eval_only.yaml \
  --max-items 8
```

The bundled general example is tuned for a local single-GPU demo rather than the
smallest possible smoke test. Short `max_new_tokens` often truncates GSM8K answers
and drives reward to zero; very small `max-items` also makes pass@k estimates noisy.

### Agent Workflow

Use this for agent-style workflows such as:

- `examples.openai_agents.train_agents.OpenAIAgentWorkflow`
- `areal.workflow.openai.*`
- `areal.workflow.openai_agent.*`

```bash
OPENAI_API_KEY=placeholder \
python -m offline_eval.eval_only.agent_eval \
  --config offline_eval/eval_only/examples/agent_openai_eval_only.yaml \
  --max-items 4
```

Recommended first choice:

- [examples/agent_openai_eval_only.yaml](./examples/agent_openai_eval_only.yaml)

This uses `areal.workflow.openai.math_agent.MathAgent`, which is the lightest
agent-style path we validated locally for the offline eval-only flow.

Because this workflow uses the OpenAI Python SDK, the simplest way to satisfy the
client-side API key requirement is to set a dummy value such as
`OPENAI_API_KEY=placeholder`.

Heavier examples such as [examples/agent_eval_only.yaml](./examples/agent_eval_only.yaml)
still exist, but they depend on a richer proxy/OpenAI-agents stack and are less
robust as a first smoke test.

## Config Contract

The eval-only scripts load a dedicated `EvalOnlyConfig`. The minimal config needs:

- `experiment_name`
- `trial_name`
- `tokenizer_path`
- `workflow`
- `workflow_kwargs`
- `valid_dataset`
- `rollout`
- `gconfig`
- `scheduler`
- `allocation_mode`
- `sglang` or `vllm`

Optional fields:

- `eval_workflow`
- `eval_workflow_kwargs`
- `processor_path`
- `eval_split`

## Important Settings

### Dump Output

Make sure:

```yaml
rollout:
  dump_to_file: true
```

The scripts always submit with `is_eval=True`, so trajectories are written into the
`eval-rollout/` subdirectory.

### Workflow Kwargs

`workflow_kwargs` can reference top-level config nodes such as `${gconfig}` and
`${tokenizer_path}`. For example:

```yaml
workflow_kwargs:
  reward_fn: areal.reward.gsm8k.gsm8k_reward_fn
  gconfig: ${gconfig}
  tokenizer: ${tokenizer_path}
  enable_thinking: false
```

### Agent Workflow Notes

For agent workflows, the most important requirement is semantic clarity of dumped
records. The `pass@k` tool assumes each `(tail_version, task_id)` corresponds to one
problem, and each emitted `sample_idx` corresponds to one candidate solution.

Be careful with:

- `rollout.openai.export_style=individual`
- multi-turn agent runs that emit multiple intermediate interactions

For the MVP, prefer workflows where a single task ends with one final reward-bearing
candidate solution.

## Example YAML

See:

- [examples/general_eval_only.yaml](./examples/general_eval_only.yaml)
- [examples/agent_eval_only.yaml](./examples/agent_eval_only.yaml)
- [examples/agent_openai_eval_only.yaml](./examples/agent_openai_eval_only.yaml)
