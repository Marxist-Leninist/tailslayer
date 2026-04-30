# CUDA VRAM Tailslayer Experiment

This experiment asks a narrower question than "does CUDA training get faster?":

> If transformer-like random gathers are replicated in VRAM, does taking the fastest replica reduce global-memory tail latency?

That matters most for embedding tables, sparse lookups, KV-cache paging, and batch-1 inference paths. Dense transformer training is usually dominated by GEMM/attention kernels, so replicated VRAM reads must beat their memory and synchronization cost before this can help end-to-end training.

## What It Measures

`vram_tail_bench.py` compiles a small CUDA kernel with NVRTC and records `clock64()` around random global-memory loads:

- `single`: one warp reads one random element from one table.
- `hedged`: two independent warps read the same logical element from two replicated tables; the host reports `min(replica0, replica1)` as the ideal winner latency.

The benchmark also reports CUDA event wall time. This is important: a lower `min` latency is only useful for a real transformer if the kernel/software stack can consume the winner early enough to offset duplicated work.

## Requirements

On the GTX 1050 Ti in this machine, CUDA 13 NVRTC cannot compile for `compute_61`, so the script uses CUDA 11.8 Python runtime/NVRTC wheels:

```powershell
python -m pip install cuda-python==11.8.7 nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11
```

## Run

```powershell
python .\experiments\cuda_vram_tailslayer\vram_tail_bench.py --table-mb 128 --samples 200000 --repeats 5
```

Interpretation:

- `effective_min` p99 lower than `single` means the replicated VRAM addresses have partially independent latency tails.
- `hedged_kernel_ms` much larger than `single_kernel_ms` means the ideal latency win may not translate to transformer speed without a persistent kernel or early-winner design.
