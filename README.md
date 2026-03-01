# Turbo-Pi0.5

**Optimized Pi0.5 VLA Model for NVIDIA Jetson Thor - 8.6x Faster Inference**

## Performance

| Metric | Original (JAX) | Turbo-Pi (FP8) |
|--------|----------------|----------------|
| Inference Speed | 1.4 Hz | **12.0 Hz** |
| Latency | 714 ms | **83.5 ms** |
| LIBERO Accuracy | - | **100%** |

## Optimization Approach

Torch-TRT FP8 Static Graph - bypasses ONNX, eliminates reformat operators:

| Component | Technology | Speedup |
|-----------|------------|---------|
| Vision Encoder | torch_tensorrt FP16 | 2.03x |
| KV Cache MLP | ModelOpt FP8 + torch_tensorrt | 2.94x |
| Denoise Loop | CUDA Graph | 2.59x |

## Quick Start (Docker)

```bash
# 1. Build LIBERO evaluation image
cd openpi && docker build -f Dockerfile.libero_eval -t turbo_pi:latest .

# 2. Start container
docker run -d --name turbo_pi \
    --runtime nvidia --gpus all \
    -v $(pwd):/workspace \
    -v ~/.cache/openpi:/root/.cache/openpi \
    -e MUJOCO_GL=egl \
    turbo_pi:latest sleep infinity

# 3. Download model
docker exec turbo_pi huggingface-cli download liangsu9988/Turbo-Pi0.5-1.1.2 \
    --local-dir /root/.cache/openpi/checkpoints/pi05_libero

# 4. Run benchmark
docker exec turbo_pi bash -c "cd /workspace && \
    python scripts/libero_eval_full_optimized.py --quick --denoising_steps 1"
```

## Benchmark Options

```bash
# Quick test (3 tasks, 3 trials)
python scripts/libero_eval_full_optimized.py --quick --denoising_steps 1

# Full LIBERO Spatial (10 tasks, 10 trials)
python scripts/libero_eval_full_optimized.py \
    --task_suite_name libero_spatial \
    --denoising_steps 1 \
    --output_file results.json

# Different denoising steps
python scripts/libero_eval_full_optimized.py --quick --denoising_steps 3  # 9.7 Hz
python scripts/libero_eval_full_optimized.py --quick --denoising_steps 10 # 5.8 Hz
```

## Benchmark Reproduction (RTX 4090/5090)

These commands run inside `openpi/` and use the optimized pathway and `serve_policy.py` pathway on the same machine.

```bash
# 1) Build CUDA13 benchmark image
cd openpi
docker build -f Dockerfile.libero_eval -t turbo_pi:latest .

# 2) Start container
docker run -d --name turbo_pi \
  --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v ~/.cache/openpi:/root/.cache/openpi \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e PYOPENGL_PLATFORM=egl \
  turbo_pi:latest sleep infinity

# 3) Create non-interactive LIBERO config (avoids first-run prompt)
mkdir -p .libero
cat > .libero/config.yaml <<'EOF'
benchmark_root: /workspace/third_party/libero/libero/libero
bddl_files: /workspace/third_party/libero/libero/libero/bddl_files
init_states: /workspace/third_party/libero/libero/libero/init_files
datasets: /workspace/third_party/libero/libero/datasets
assets: /workspace/third_party/libero/libero/libero/assets
EOF
```

### Optimized pathway benchmark

```bash
# Quick sanity run
docker exec turbo_pi bash -lc '
  export LIBERO_CONFIG_PATH=/workspace/.libero
  cd /workspace
  python scripts/libero_eval_full_optimized.py --quick --denoising_steps 1
'

# 20 denoising steps
docker exec turbo_pi bash -lc '
  export LIBERO_CONFIG_PATH=/workspace/.libero
  cd /workspace
  python scripts/libero_eval_full_optimized.py --quick --denoising_steps 20
'
```

### serve_policy.py pathway benchmark

```bash
# Through websocket server + client loop with LIBERO env, using 20 denoising steps
docker exec turbo_pi bash -lc '
  export LIBERO_CONFIG_PATH=/workspace/.libero
  cd /workspace
  python scripts/libero_eval_serve_policy.py --quick --denoising_steps 20
'
```

Notes:
- CUDA13 container can run the optimized benchmark path, but TRT-LLM plugins are CUDA12.x-only in this repo's current setup.
- If you see a first-run LIBERO input prompt, your `LIBERO_CONFIG_PATH` was not set correctly.

## Python API

```python
from libero_eval_full_optimized import FullOptimizedPolicy

policy = FullOptimizedPolicy(
    checkpoint_dir="~/.cache/openpi/checkpoints/pi05_libero",
    num_denoising_steps=1,
)

result = policy.infer({
    "observation/image": image,           # (224, 224, 3) uint8
    "observation/wrist_image": wrist_img, # (224, 224, 3) uint8
    "observation/state": state,           # (8,) float32
    "prompt": "pick up the black bowl",
})
actions = result["actions"]  # (50, 7) action chunk
```

## Model

**HuggingFace:** [liangsu9988/Turbo-Pi0.5-1.1.2](https://huggingface.co/liangsu9988/Turbo-Pi0.5-1.1.2)

- Vision: SigLIP-SO400M (400M params)
- Language: Gemma 2B
- Action: Gemma 300M Expert

## Hardware Requirements

- **Target:** NVIDIA Jetson Thor (JetPack 7.1+)
- **Container:** nvcr.io/nvidia/pytorch:25.12-py3
- **Dependencies:** torch-tensorrt, modelopt, pycuda

## License

Apache License 2.0

Based on [OpenPi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence.
