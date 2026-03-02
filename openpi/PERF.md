# TurboPi Performance Notes

## Latest run (CUDA 12.9 quickstart)

- Date: `2026-03-02`
- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `580.126.18`
- Image: `turbo_pi:cuda129-2506-trtllm`
- Command:
  - `python scripts/libero_eval_full_optimized.py --quick --denoising_steps 6`

## Benchmark config

- Task suite: `libero_spatial`
- Quick mode: `3` tasks, `3` trials/task (`9` total episodes)
- Denoising steps: `6`
- Inference samples in latency stats: `396`

## Results

- Accuracy: `0/9 (0.0%)`
- Latency:
  - Mean: `51.36 ms`
  - Std: `0.41 ms`
  - P50: `51.37 ms`
  - P95: `51.98 ms`
  - Throughput: `19.5 Hz`
- Component breakdown:
  - Vision TRT: `6.16 ms` (`12.0%`)
  - KV Cache TRT: `20.65 ms` (`40.2%`)
  - Denoise CUDA: `19.53 ms` (`38.0%`)
  - Total: `51.36 ms` (`100.0%`)

## Notes

- First run included TRT engine build and FP8 extension warmup (`modelopt_cuda_ext_fp8` loaded in `118.3s`).
- Run log: `/tmp/turbopi_cuda129_bench.log`
