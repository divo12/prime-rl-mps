# Apple Silicon GPU capacity for RL post-training

Measurements in this document come from `mps/benchmarks/capacity.py` (PyTorch MPS,
torch 2.13.0, macOS 26.2) on the reference machine, an **Apple M4 Pro with 24 GB
unified memory and a 16-core GPU**. Datacenter figures are manufacturer datasheets.

## Measured on this machine

| Metric | Measured | Note |
| --- | --- | --- |
| Achieved copy bandwidth | ~136 GB/s | 2 GiB fp16 buffer copy; datasheet peak is 273 GB/s |
| FP32 matmul (4096³) | 4.3 TFLOPS | PyTorch MPS backend |
| FP16 matmul (4096³) | 4.5 TFLOPS | PyTorch MPS backend |
| FP16 matmul (8192³ / FFN-shaped 16384×4096×11008) | 4.6–4.7 TFLOPS | saturates around this level |
| Recommended max working set | 17.8 GB | of 24 GB unified memory (`torch.mps.recommended_max_memory`) |
| Driver default | GPU may wire ~75% of RAM | raise via `iogpu.wired_limit_mb` if needed |

Two structural facts dominate everything else:

1. **Bandwidth, not capacity, is the Apple bottleneck.** Unified memory gives a
   24 GB–512 GB single-device pool — larger than any single A100/H100 — but the
   bus feeding it is 7–25× slower than HBM.
2. **PyTorch MPS leaves compute on the table.** ~4.5 TFLOPS fp16 through MPS is
   far below what M-series hardware does with tuned stacks (MLX reaches roughly
   3–5× higher matmul throughput on the same chips by using MPSGraph fusion and
   Apple's matrix accelerators). The runner in `mps/` trades that peak throughput
   for prime-rl code compatibility (verifiers tasksets, PEFT, transformers).

## Chip lineup vs datacenter GPUs

| GPU | Memory | Bandwidth | FP16 vector/tensor class | Training-relevant takeaway |
| --- | --- | --- | --- | --- |
| **M4 Pro (this Mac)** | up to 64 GB unified (24 here) | 273 GB/s | ~5 TFLOPS achieved via MPS | LoRA RL on ≤4B models; full-FT ≤1B |
| M4 Max | up to 128 GB | 546 GB/s | ~18 TFLOPS fp16 (est.) | LoRA RL on ≤14B models |
| M3 Ultra | up to 512 GB | 819 GB/s | largest unified pool | QLoRA-class RL on very large models |
| A100 SXM 80GB | 80 GB HBM2e | 2,039 GB/s | 312 TFLOPS dense tensor | production RL workhorse |
| H100 SXM 80GB | 80 GB HBM3 | 3,350 GB/s | 1,979 TFLOPS dense tensor | ~10–50× Apple compute per device |

Rough ratios, M4 Pro vs one A100: **0.33× bandwidth, ~0.02× achievable matmul
throughput, ≥1× memory capacity at 24 GB (up to 6× across the lineup)**. vs an
H100: 0.08× bandwidth, ~0.004× throughput.

## What fits under the 17.8 GB working set

Training-state math per parameter: full-FT AdamW ≈ 16 B (fp32 master + m + v +
fp16 grads/weights); LoRA adds <1% on top of frozen fp16 weights (2 B/param).

| Model class | Full-FT AdamW | LoRA (fp16 base) | Verdict on 24 GB |
| --- | --- | --- | --- |
| 0.5B (Qwen3-0.6B) | ~9 GB ✓ | ~1.5 GB ✓ | both modes comfortable |
| 1.5B | ~26 GB ✗ | ~4 GB ✓ | LoRA only |
| 4B | ✗ | ~10 GB ✓ | LoRA + grad checkpointing + short context |
| 7–8B | ✗ | ~17–19 GB ⚠ | borderline; needs reduced KV cache/`wired_limit_mb` bump |
| 14B+ | ✗ | ✗ via PyTorch fp16 | only via MLX 4-bit quantized training |

KV cache and activations scale with batch × context and share the same pool;
group-relative rollouts multiply generation traffic by `group_size`, which is why
the algorithm section argues for reusing each rollout batch across multiple
optimization epochs.
