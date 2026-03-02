#!/usr/bin/env python3
"""Validate output parity between optimized and serve_policy pathways.

This script feeds identical observations to:
1) FullOptimizedPolicy (direct in-process call)
2) A websocket policy server (serve_policy-style path)

It compares output action tensors and reports numerical closeness metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (
    os.path.join(_THIS_DIR, "..", "src"),
    os.path.join(_THIS_DIR, "..", "packages", "openpi-client", "src"),
    "/workspace/src",
    "/workspace/packages/openpi-client/src",
):
    _abs = os.path.abspath(_path)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

from libero_eval_full_optimized import FullOptimizedPolicy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.policies.libero_policy import LiberoOutputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate parity: optimized vs serve_policy pathway")
    parser.add_argument("--checkpoint_dir", default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--policy_config", default="pi05_libero")
    parser.add_argument("--num_steps", type=int, default=6)
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--server_timeout_s", type=int, default=240)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--compare_space",
        choices=["model", "libero"],
        default="libero",
        help="Compare in raw model action space (32D) or LIBERO action space (7D).",
    )
    parser.add_argument(
        "--dump_dir",
        default=None,
        help="If set, dump per-case outputs and summary files here to avoid rerunning comparisons.",
    )
    parser.add_argument("--server_log", default="/tmp/validate_serve_policy.log")
    return parser.parse_args()


def wait_for_port(host: str, port: int, timeout_s: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        with contextlib.suppress(OSError):
            with socket.create_connection((host, port), timeout=1):
                return True
        time.sleep(1)
    return False


def make_server_code(policy_config_name: str, checkpoint_dir: str, denoising_steps: int, host: str, port: int) -> str:
    # Add control hooks to inspect transformed inputs and to force identical noise tensors.
    return f"""
import jax
import numpy as np
import torch
import time
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

class SeededPolicy:
    def __init__(self, base_policy):
        self._base = base_policy
        self.metadata = getattr(base_policy, "metadata", {{}})

    def _infer_from_transformed(self, transformed_inputs, noise=None, apply_output_transform=True):
        inputs = jax.tree.map(lambda x: x, transformed_inputs)

        def _to_pytorch(x):
            t = torch.from_numpy(np.array(x))
            if t.dtype in (torch.float32, torch.float64):
                t = t.to(torch.bfloat16)
            return t.to(self._base._pytorch_device)[None, ...]

        inputs = jax.tree.map(_to_pytorch, inputs)
        sample_kwargs = dict(self._base._sample_kwargs)
        if noise is not None:
            noise_t = torch.from_numpy(np.asarray(noise, dtype=np.float32)).to(torch.bfloat16).to(self._base._pytorch_device)
            if noise_t.ndim == 2:
                noise_t = noise_t[None, ...]
            sample_kwargs["noise"] = noise_t

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {{
            "state": inputs["state"],
            "actions": self._base._sample_actions(self._base._pytorch_device, observation, **sample_kwargs),
        }}
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu().float()), outputs)
        if apply_output_transform:
            outputs = self._base._output_transform(outputs)
        outputs["policy_timing"] = {{"infer_ms": model_time * 1000}}
        return outputs

    def infer(self, obs):
        seed = int(obs.pop("_seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        noise = obs.pop("_noise", None)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)

        dump_transformed_only = bool(obs.pop("_dump_transformed_only", False))
        if dump_transformed_only:
            transformed = self._base._input_transform(dict(obs))
            return {{"transformed_inputs": transformed}}

        transformed_inputs = obs.pop("_transformed_inputs", None)
        skip_output_transform = bool(obs.pop("_skip_output_transform", False))
        if transformed_inputs is not None:
            return self._infer_from_transformed(
                transformed_inputs,
                noise=noise,
                apply_output_transform=not skip_output_transform,
            )
        return self._base.infer(obs, noise=noise)
    def reset(self):
        return self._base.reset()

train_config = _config.get_config({policy_config_name!r})
base = policy_config.create_trained_policy(
    train_config,
    {checkpoint_dir!r},
    sample_kwargs={{'num_steps': {denoising_steps}}},
    norm_stats={{}},
    pytorch_device='cuda',
)
seeded = SeededPolicy(base)
server = WebsocketPolicyServer(policy=seeded, host={host!r}, port={port}, metadata=seeded.metadata)
server.serve_forever()
"""


def make_obs(seed: int, idx: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + idx * 9973)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    state = rng.standard_normal(size=(8,)).astype(np.float32)
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": "pick up the black bowl",
    }


def to_numpy_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: to_numpy_tree(v) for k, v in tree.items()}
    return np.asarray(tree)


def load_log_tail(path: str, max_chars: int = 4000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return "<log file not found>"
    return text[-max_chars:]


def main() -> int:
    args = parse_args()
    libero_outputs = LiberoOutputs()
    dump_dir = None
    if args.dump_dir:
        dump_dir = os.path.abspath(os.path.expanduser(args.dump_dir))
        os.makedirs(dump_dir, exist_ok=True)

    if not os.path.isdir(os.path.expanduser(args.checkpoint_dir)):
        print(f"Checkpoint directory does not exist: {args.checkpoint_dir}", file=sys.stderr)
        return 2

    print("Loading optimized policy ...")
    optimized = FullOptimizedPolicy(
        checkpoint_dir=args.checkpoint_dir,
        num_denoising_steps=args.num_steps,
    )

    print("Starting serve_policy pathway server ...")
    server_env = os.environ.copy()
    repo_root = os.path.abspath(os.path.join(_THIS_DIR, ".."))
    server_env["PYTHONPATH"] = (
        f"{os.path.join(repo_root, 'src')}:"
        f"{os.path.join(repo_root, 'packages', 'openpi-client', 'src')}:"
        f"{server_env.get('PYTHONPATH', '')}"
    )
    with open(args.server_log, "w", encoding="utf-8") as log_f:
        server_proc = subprocess.Popen(
            [
                "python",
                "-u",
                "-c",
                make_server_code(args.policy_config, args.checkpoint_dir, args.num_steps, args.host, args.port),
            ],
            stdout=log_f,
            stderr=log_f,
            cwd=repo_root,
            env=server_env,
        )

    try:
        if not wait_for_port(args.host, args.port, args.server_timeout_s):
            print("serve_policy pathway server failed to start", file=sys.stderr)
            print("=== server log tail ===", file=sys.stderr)
            print(load_log_tail(args.server_log), file=sys.stderr)
            return 1

        client = WebsocketClientPolicy(args.host, args.port)
        print("Connected to serve_policy pathway server.")

        case_summaries: list[dict[str, float | int | bool]] = []
        pass_flags: list[bool] = []

        for i in range(args.num_cases):
            raw_obs = make_obs(args.seed, i)
            noise_seed = int(args.seed + i * 1009)
            noise_rng = np.random.default_rng(noise_seed)
            noise = noise_rng.standard_normal((optimized.action_horizon, optimized.action_dim), dtype=np.float32)

            transformed_resp = client.infer({**raw_obs, "_dump_transformed_only": True})
            transformed_inputs = to_numpy_tree(transformed_resp["transformed_inputs"])

            opt_actions = np.asarray(
                optimized.infer(
                    transformed_inputs,
                    noise=noise,
                    apply_output_transform=False,
                )["actions"],
                dtype=np.float32,
            )

            serve_obs = {
                "_seed": noise_seed,
                "_noise": noise,
                "_transformed_inputs": transformed_inputs,
                "_skip_output_transform": True,
            }
            serve_resp = client.infer(serve_obs)
            srv_actions_raw = np.asarray(serve_resp["actions"], dtype=np.float32)
            srv_actions = srv_actions_raw

            opt_actions_raw = opt_actions.copy()
            if args.compare_space == "libero":
                if opt_actions.shape[-1] != 7:
                    opt_actions = np.asarray(libero_outputs({"actions": opt_actions})["actions"], dtype=np.float32)
                if srv_actions.shape[-1] != 7:
                    srv_actions = np.asarray(libero_outputs({"actions": srv_actions})["actions"], dtype=np.float32)

            if opt_actions.shape != srv_actions.shape:
                print(f"Case {i}: SHAPE MISMATCH optimized={opt_actions.shape} serve={srv_actions.shape}")
                case_summaries.append(
                    {
                        "case": i,
                        "shape_match": False,
                        "allclose": False,
                        "max_abs_diff": float("inf"),
                        "mean_abs_diff": float("inf"),
                    }
                )
                pass_flags.append(False)
                if dump_dir:
                    np.savez_compressed(
                        os.path.join(dump_dir, f"case_{i:03d}.npz"),
                        optimized_actions_raw=opt_actions_raw,
                        serve_actions_raw=srv_actions_raw,
                        optimized_actions=opt_actions,
                        serve_actions=srv_actions,
                        noise=noise,
                    )
                continue

            abs_diff = np.abs(opt_actions - srv_actions)
            max_abs_diff = float(np.max(abs_diff))
            mean_abs_diff = float(np.mean(abs_diff))
            allclose = bool(np.allclose(opt_actions, srv_actions, atol=args.atol, rtol=args.rtol))
            pass_flags.append(allclose)
            case_summaries.append(
                {
                    "case": i,
                    "shape_match": True,
                    "allclose": allclose,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                }
            )
            print(
                f"Case {i}: allclose={allclose} "
                f"max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f}"
            )
            if dump_dir:
                np.savez_compressed(
                    os.path.join(dump_dir, f"case_{i:03d}.npz"),
                    optimized_actions_raw=opt_actions_raw,
                    serve_actions_raw=srv_actions_raw,
                    optimized_actions=opt_actions,
                    serve_actions=srv_actions,
                    abs_diff=abs_diff,
                    transformed_state=np.asarray(transformed_inputs["state"], dtype=np.float32),
                    transformed_prompt_tokens=np.asarray(transformed_inputs["tokenized_prompt"], dtype=np.int64),
                    noise=noise,
                )

        max_diffs = [float(c["max_abs_diff"]) for c in case_summaries if np.isfinite(c["max_abs_diff"])]
        mean_diffs = [float(c["mean_abs_diff"]) for c in case_summaries if np.isfinite(c["mean_abs_diff"])]
        overall_pass = bool(pass_flags) and all(pass_flags)

        print("\n=== PARITY SUMMARY ===")
        print(f"num_cases: {args.num_cases}")
        print(f"rtol: {args.rtol}  atol: {args.atol}")
        print(f"overall_pass: {overall_pass}")
        if max_diffs:
            print(f"max_abs_diff: max={max(max_diffs):.6f} median={statistics.median(max_diffs):.6f}")
        if mean_diffs:
            print(f"mean_abs_diff: max={max(mean_diffs):.6f} median={statistics.median(mean_diffs):.6f}")

        if dump_dir:
            summary = {
                "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "num_cases": args.num_cases,
                "num_steps": args.num_steps,
                "compare_space": args.compare_space,
                "rtol": args.rtol,
                "atol": args.atol,
                "overall_pass": overall_pass,
                "cases": case_summaries,
            }
            summary_path = os.path.join(dump_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Dumped comparison artifacts to: {dump_dir}")

        return 0 if overall_pass else 3
    finally:
        with contextlib.suppress(Exception):
            server_proc.send_signal(signal.SIGTERM)
            server_proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
