# Apple Silicon MPS

This subproject runs a small, serial RL experiment on an Apple Silicon GPU. It uses
PyTorch MPS for the policy, PEFT LoRA for memory-efficient updates, and installed
verifiers v1 tasksets for task loading and reward scoring.

From the repository root:

```bash
uv run --project mps mps-rl @ mps/configs/reverse-text.toml
```

The example loads `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`, samples four responses per reverse-text task,
normalizes their rewards within the group, and optimizes a PPO-style clipped surrogate over
`ppo_epochs` passes per batch (GRPO; see `mps/docs/rl-on-mps.md`). The final LoRA adapter is written to
`outputs/mps-reverse-text`.

Run the MPS check with:

```bash
uv run --project mps pytest mps/tests -q
```

## Scope

This is a development path, not the production Prime-RL launcher. It supports dense
causal language models and prompted, single-turn, tool-free tasksets with offline
reward functions. It does not support vLLM, multi-turn harnesses, tools, FSDP, NCCL,
multi-GPU training, live weight broadcast, checkpoint resume, or the production AIPO
trainer. Those remain on the CUDA path.

MPS requires the whole model to fit in unified memory. Reduce `group_size`,
`max_completion_tokens`, or choose a smaller instruct model if macOS reports an
out-of-memory error.
