#!/usr/bin/env python3
"""Minimal repro for TensorRT invalid-resource-handle in TurboPi pipeline.

Focuses on the Vision TRT path used by libero_eval_full_optimized.py:
1) Load pi0.5 PyTorch model checkpoint
2) Compile vision tower with Torch-TensorRT
3) Execute repeated forwards (default + side stream)
4) Report first failure with environment/version diagnostics
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import torch
import torch.nn as nn

# Repo-local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

for path in ["/workspace/src", "/workspace/packages/openpi-client/src", "/workspace/third_party/libero"]:
    if path not in sys.path:
        sys.path.insert(0, path)


class VisionWrapper(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, pixel_values):
        outputs = self.vision_tower(pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state


def print_versions() -> None:
    print("=== Versions ===")
    print("python", sys.version.replace("\n", " "))
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device", torch.cuda.get_device_name(0))
        print("cuda capability", torch.cuda.get_device_capability(0))
    for mod_name in ["torch_tensorrt", "modelopt", "tensorrt", "tensorrt_llm"]:
        try:
            mod = __import__(mod_name)
            print(mod_name, getattr(mod, "__version__", "<no __version__>"))
        except Exception as exc:
            print(mod_name, "IMPORT_FAIL", repr(exc))
    print("PYTHONBREAKPOINT", os.environ.get("PYTHONBREAKPOINT", "<unset>"))
    print("CUDA_MODULE_LOADING", os.environ.get("CUDA_MODULE_LOADING", "<unset>"))
    print("TRTLLM_PLUGINS_PATH", os.environ.get("TRTLLM_PLUGINS_PATH", "<unset>"))
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True)
        print("=== nvidia-smi ===")
        print(out)
    except Exception as exc:
        print("nvidia-smi failed:", repr(exc))


def load_model(checkpoint_dir: pathlib.Path, device: str):
    from safetensors.torch import load_file
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    else:
        model_config = {}

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=model_config.get("tokenizer_max_length", 200),
        pi05=True,
        dtype="bfloat16",
    )

    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device)
    model.eval()
    return model


def run_repro(args: argparse.Namespace) -> int:
    print_versions()

    if not torch.cuda.is_available():
        print("CUDA not available; cannot repro runtime handle issue")
        return 1

    device = args.device
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = False

    ckpt = pathlib.Path(args.checkpoint_dir).expanduser()
    print(f"Loading checkpoint from {ckpt}")
    model = load_model(ckpt, device)

    if args.init_kv_engine:
        print("Initializing TorchTRTFP8KVCacheEngine...")
        from openpi.inference.torch_trt_fp8_kv_cache import TorchTRTFP8KVCacheEngine

        kv_t0 = time.perf_counter()
        kv_engine = TorchTRTFP8KVCacheEngine(str(ckpt), device, compile_trt=True)
        print(
            "KV engine ready in "
            f"{(time.perf_counter() - kv_t0):.2f}s, "
            f"TRT layers={getattr(kv_engine, '_trt_compiled_count', 'unknown')}"
        )

    if args.init_libero_env:
        print("Initializing one LIBERO OffScreenRenderEnv...")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
        task = task_suite.get_task(0)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=args.libero_resolution,
            camera_widths=args.libero_resolution,
        )
        env.seed(args.seed)
        init_states = task_suite.get_task_init_states(0)
        env.reset()
        _ = env.set_init_state(init_states[0])
        for _ in range(args.libero_warm_steps):
            _, _, done, _ = env.step([0.0] * 6 + [-1.0])
            if done:
                break
        env.close()
        print("LIBERO env init/warmup done")

    if args.capture_denoise_graph:
        print("Capturing denoising CUDA graph...")
        import importlib.util

        eval_path = pathlib.Path(__file__).resolve().parent / "libero_eval_full_optimized.py"
        spec = importlib.util.spec_from_file_location("libero_eval_full_optimized", str(eval_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        CUDAGraphDenoiseLoop = module.CUDAGraphDenoiseLoop
        DenoiseStepWrapper = module.DenoiseStepWrapper
        SEQ_LEN = module.SEQ_LEN

        wrapper = DenoiseStepWrapper(model, prefix_len=SEQ_LEN).to(device).eval()
        denoise_graph = CUDAGraphDenoiseLoop(wrapper, num_steps=args.denoise_steps)

        num_layers = 18
        num_kv_heads = 1
        head_dim = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.head_dim
        dummy_keys = torch.randn(1, num_layers, num_kv_heads, SEQ_LEN, head_dim, device=device, dtype=torch.bfloat16)
        dummy_values = torch.randn(1, num_layers, num_kv_heads, SEQ_LEN, head_dim, device=device, dtype=torch.bfloat16)
        dummy_pad_masks = torch.ones(1, SEQ_LEN, device=device, dtype=torch.bool)

        denoise_graph.capture_graph(dummy_keys, dummy_values, dummy_pad_masks, device)
        x_t = torch.randn(1, model.config.action_horizon, model.config.action_dim, device=device, dtype=torch.bfloat16)
        _ = denoise_graph.infer(x_t)
        torch.cuda.synchronize()
        print("Denoising CUDA graph capture + one replay done")

    import torch_tensorrt

    vision_tower = model.paligemma_with_expert.paligemma.vision_tower
    projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector

    wrapper = VisionWrapper(vision_tower).to(device).half().eval()

    print("Compiling vision TRT...")
    t0 = time.perf_counter()
    vision_trt = torch_tensorrt.compile(
        wrapper,
        inputs=[torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float16)],
        enabled_precisions={torch.float16},
        workspace_size=4 << 30,
        min_block_size=1,
    )
    torch.cuda.synchronize()
    print(f"Vision TRT compile took {(time.perf_counter() - t0):.2f}s")

    side_stream = torch.cuda.Stream()

    # Warmup on default stream
    for _ in range(args.warmup):
        img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
        out = vision_trt(img)
        _ = projector(out.to(torch.bfloat16))
    torch.cuda.synchronize()
    print(f"Warmup done ({args.warmup} iters)")

    # Repro loop
    for i in range(args.iters):
        try:
            img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
            out = vision_trt(img)
            _ = projector(out.to(torch.bfloat16))

            # Stress alternate stream every N steps to surface handle bugs earlier
            if args.side_stream_every > 0 and (i + 1) % args.side_stream_every == 0:
                with torch.cuda.stream(side_stream):
                    img2 = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
                    out2 = vision_trt(img2)
                    _ = projector(out2.to(torch.bfloat16))

            torch.cuda.synchronize()
            if (i + 1) % max(1, args.log_every) == 0:
                print(f"iter={i+1} OK")
        except Exception as exc:
            print(f"\nFAIL at iter={i+1}: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            if "invalid resource handle" in str(exc).lower():
                print("Detected invalid resource handle in TRT execution path")
                return 2
            return 3

    print(f"Completed {args.iters} iterations without failure")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repro invalid-resource-handle in TRT vision path")
    parser.add_argument(
        "--checkpoint_dir",
        default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"),
        help="Path to pi05_libero checkpoint directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--side_stream_every",
        type=int,
        default=5,
        help="Run one inference on a side CUDA stream every N iterations (0 disables)",
    )
    parser.add_argument(
        "--init_kv_engine",
        action="store_true",
        help="Initialize TorchTRTFP8KVCacheEngine before vision TRT loop",
    )
    parser.add_argument(
        "--init_libero_env",
        action="store_true",
        help="Initialize a LIBERO OffScreenRenderEnv before TRT execution",
    )
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--libero_resolution", type=int, default=256)
    parser.add_argument("--libero_warm_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capture_denoise_graph", action="store_true")
    parser.add_argument("--denoise_steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_repro(parse_args()))
