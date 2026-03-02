# TurboPi Performance Notes

## Test machine

- Date: 2026-03-02
- GPUs: 2x NVIDIA GeForce RTX 4090
- Driver: 580.126.18

## Benchmark setup

- Script: `scripts/libero_eval_full_optimized.py`
- Mode: `--quick` (3 tasks, 3 trials each)
- Denoising steps: `6`
- Metric: end-to-end policy inference latency and component breakdown

## Results

### CUDA 12.9 (`turbo_pi:cuda129-2506-trtllm`)

- Mean: `60.50 ms`
- Std: `0.35 ms`
- P50: `60.48 ms`
- P95: `60.77 ms`
- Throughput: `16.5 Hz`
- Breakdown:
  - Vision TRT: `7.46 ms`
  - KV Cache TRT: `27.91 ms`
  - Denoise CUDA: `23.82 ms`

### CUDA 13 (`turbo_pi:latest`)

- Mean: `55.53 ms`
- Std: `0.21 ms`
- P50: `55.53 ms`
- P95: `55.79 ms`
- Throughput: `18.0 Hz`
- Breakdown:
  - Vision TRT: `8.10 ms`
  - KV Cache TRT: `22.94 ms`
  - Denoise CUDA: `23.50 ms`

## Delta

- CUDA 13 vs CUDA 12.9 at 6 denoising steps:
  - `4.97 ms` lower mean latency (`~8.9%` faster)
