# reuse.cpp benchmarks

Measured with `reuse-bench` (`tools/bench.cpp`):

```
reuse-bench [--threads N] <gguf> <device> <wav> [repeat=3] [out.wav]
```

It prints the load time, the wall time of each `reuse_enhance` call, the
real-time factor (RTF = audio length / enhance time; higher is faster, 1.0 = real time), and the
largest backend compute buffer. For windowed inputs (> 20 s) that is the largest
buffer over all windows.

Model: F32 GGUF converted from `nvidia/NVIDIA-NemotronLabs-RE-USE` @ `31f82f2`.

## Apple M1 Max (64 GB), 2026-11-04

| Input | Device | Mode | Enhance time | RTF | Compute buffer |
|---|---|---|---:|---:|---:|
| 10.74 s speech, 24 kHz | MTL0 | single pass | 17.0 s | 0.63 | 2799 MiB |
| 10.74 s speech, 24 kHz | CPU, 8 threads | single pass | 54.3 s | 0.20 | — |
| 60 s speech, 24 kHz | MTL0 | single pass (pre-windowing) | 96.5 s | 0.62 | 15619 MiB |
| 60 s speech, 24 kHz | MTL0 | windowed (4 × ~16.5 s) | 123.5 s | 0.49 | 4298 MiB peak |

Notes:

- Single-pass compute memory grows roughly linearly with input length (~2.6 GB
  per 10 s), so inputs longer than 20 s are split into even windows with 2 s
  overlap and a raised-cosine crossfade. Inputs of 20 s or less run as one exact
  pass. Windowing costs ~28 % more time on 60 s but cuts peak memory from
  15.6 GB to 4.3 GB.
- Metal needs the F32 precision requests on the Mamba mul_mats (see
  `tests/TOLERANCES.md`). Without them, Metal stages F32×F32 matrix products in
  half precision and 44.1 kHz input overflows to inf/NaN.
- Numerical parity on CPU and MTL0 is in `tests/TOLERANCES.md`.

## AMD (ROCm / Vulkan), CUDA

Not measured yet.
