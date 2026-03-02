# TurboPi Performance Notes

## Test machine

- Date: 2026-03-02
- GPUs: 2x NVIDIA GeForce RTX 4090
- Driver: 580.126.18

## Benchmark setup

- Mode: `--quick` (3 tasks, 3 trials each), `--denoising_steps 6`
- LIBERO config: `/workspace/.libero/config.yaml`
- Task suite: `libero_spatial`

## Results

### CUDA 12.9 optimized (`scripts/libero_eval_full_optimized.py`, `turbo_pi:cuda129-2506-trtllm`)

- Mean: `60.44 ms`
- Std: `0.27 ms`
- P50: `60.40 ms`
- P95: `60.70 ms`
- Throughput: `16.5 Hz`
- Breakdown:
  - Vision TRT: `7.42 ms`
  - KV Cache TRT: `27.87 ms`
  - Denoise CUDA: `23.82 ms`

### CUDA 13 optimized (`scripts/libero_eval_full_optimized.py`, `turbo_pi:latest`)

- Mean: `55.55 ms`
- Std: `0.25 ms`
- P50: `55.53 ms`
- P95: `55.78 ms`
- Throughput: `18.0 Hz`
- Breakdown:
  - Vision TRT: `8.14 ms`
  - KV Cache TRT: `22.91 ms`
  - Denoise CUDA: `23.51 ms`

### CUDA 13 serve-policy (`scripts/libero_eval_serve_policy.py`, `turbo_pi:latest`)

- Mean infer latency: `140.45 ms`
- P50 infer latency: `139.75 ms`
- P95 infer latency: `140.73 ms`
- Throughput: `7.12 Hz`
- Policy infer calls: `396`
- Episodes: `9` (`0/9` success in quick benchmark)

## Notes

- For CUDA13 serve-policy in this environment, the container needed:
  - `pip install -q "jax==0.5.3" "jaxlib==0.5.3" "chex==0.1.88" "numpy<2.4"`
  - `PYTHONPATH=/workspace/src:/workspace/packages/openpi-client/src:/workspace/third_party/libero:${PYTHONPATH:-}`

## Delta

- CUDA 13 vs CUDA 12.9 at 6 denoising steps:
  - `4.89 ms` lower mean latency (`~8.1%` faster)
- CUDA 13 optimized vs CUDA 13 serve-policy:
  - `84.90 ms` lower mean infer latency (`~2.53x` faster)

## Benchmark Update (2026-03-02, commit `73d7ed7`)

### CUDA 12.9 optimized (`scripts/libero_eval_full_optimized.py`, `turbo_pi:cuda129-2506-trtllm`)

- Mean: `51.45 ms`
- Std: `0.40 ms`
- P50: `51.49 ms`
- P95: `52.04 ms`
- Throughput: `19.4 Hz`
- Accuracy: `0/9`
- Breakdown:
  - Vision TRT: `6.18 ms`
  - KV Cache TRT: `20.65 ms`
  - Denoise CUDA: `19.54 ms`

### CUDA 13 optimized (`scripts/libero_eval_full_optimized.py`, `turbo_pi:latest`)

- Mean: `47.82 ms`
- Std: `0.42 ms`
- P50: `47.85 ms`
- P95: `48.41 ms`
- Throughput: `20.9 Hz`
- Accuracy: `0/9`
- Breakdown:
  - Vision TRT: `5.66 ms`
  - KV Cache TRT: `19.37 ms`
  - Denoise CUDA: `18.02 ms`

### CUDA 13 serve-policy (`scripts/libero_eval_serve_policy.py`, `turbo_pi:latest`)

- Mean infer latency: `95.53 ms`
- P50 infer latency: `95.01 ms`
- P95 infer latency: `95.76 ms`
- Throughput: `10.47 Hz`
- Policy infer calls: `396`
- Episodes: `9` (`0/9` success in quick benchmark)

### Update Delta

- CUDA 13 vs CUDA 12.9 at 6 denoising steps:
  - `3.63 ms` lower mean latency (`~7.1%` faster)
- CUDA 13 optimized vs CUDA 13 serve-policy:
  - `47.71 ms` lower mean infer latency (`~2.00x` faster)
