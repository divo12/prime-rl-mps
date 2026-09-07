# Maximum viable RL algorithm on MPS

Decision: **GRPO — group-relative policy gradient with a PPO-style clipped
surrogate and multiple optimization epochs per rollout batch.** That is the
ceiling for a single Apple Silicon device through PyTorch MPS, and it is what
`mps/src/prime_rl_mps/train.py` implements (upgraded from plain on-policy REINFORCE).

## Why GRPO is the ceiling

Hardware constraints from `capacity.md` drive the algorithm choice:

1. **One device, no separation of rollout and training.** vLLM, NCCL, FSDP, and
   weight-broadcast transport are all CUDA-path features. Generation runs through
   HF `transformers` on the same GPU that trains, at ~10–30 tok/s for sub-1B
   models. Rollout generation is 5–10× more expensive than the gradient step.
2. **Slow generation must be amortized.** If every rollout batch is used for
   exactly one gradient step (pure on-policy PG/REINFORCE), most wall-clock time
   is spent re-sampling. Reusing each batch for `ppo_epochs` optimization passes
   recovers that cost — but reuse breaks the on-policy assumption, which is
   exactly what PPO's importance ratio + clipping exists to fix.
3. **Memory rules out value-model PPO and large reference models.** A critic
   doubles weights + activations; a frozen reference model doubles them again.
   Group-relative advantages (GRPO) replace the critic entirely; KL to a
   reference is optional and only affordable for ≤1B models on this machine.

The result is textbook DeepSeek-style GRPO: sample `group_size` completions per
prompt, score with verifiers reward functions, normalize rewards within the
group, then optimize a clipped importance-weighted surrogate:

```
ratio   = exp(logp_new − logp_old)
surrogate = min(ratio · A, clip(ratio, 1−ε, 1+ε) · A)
```

with `logp_old` frozen at sampling time. On the first epoch the ratio is 1 and
the update is identical to the original on-policy PG; later epochs are exactly
the PPO correction.

## Cross-checks

- **mlx-tune** (`~/mlx-tune`, native MLX) arrived at the same shape for Mac:
  `GRPOTrainer` does two-phase generate-then-optimize, group-normalized
  advantages, `std < ε` skip when all rewards tie, LoRA-only updates, and
  amortization tricks (shared prompt KV cache forked per generation,
  `mx.compile`d steps). It offers DPO/ORPO/KTO/SimPO as offline preference
  methods — those need static datasets rather than live rollouts, so they dodge
  the generation bottleneck but do not exercise an environment. GRPO is the
  strongest *online* RL loop that fits.
- **prime-rl production trainer** uses the same GRPO/AIPO family at cluster
  scale with vLLM rollouts and FSDP sharding; the MPS runner mirrors its loss
  semantics minus distributed machinery.

## What is deliberately out of scope on MPS

| Feature | Why not |
| --- | --- |
| PPO with learned value head | doubles optimizer/activation memory for little gain over group baselines |
| Multi-turn / tool-using tasksets | requires server-based rollout harness (vLLM path) |
| Full-parameter FT >1B | AdamW state math exceeds the 17.8 GB working set (see capacity.md) |
| Async actor–learner | single device; nothing to overlap except CPU scoring |

## Upgrade path beyond PyTorch MPS

The real performance ceiling sits in the framework, not the algorithm: MLX
reaches several times the matmul throughput and supports 4-bit QLoRA-scale
models. A future `mps` track could keep this runner as the prime-rl-compatible
baseline and add an MLX generation/training backend behind the same config —
mlx-tune demonstrates both halves (fast generation with shared prompt caches,
LoRA GRPO) already work natively on Apple silicon.
