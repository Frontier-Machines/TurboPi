#!/usr/bin/env python3
"""Benchmark the serve_policy pathway on LIBERO via websocket client/server."""

import argparse
import collections
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

for _path in ["/workspace/src", "/workspace/packages/openpi-client/src", "/workspace/third_party/libero"]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from openpi_client import image_tools
from openpi_client import websocket_client_policy as ws_policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUBPROCESS_PY_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "src"),
    os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"),
    os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"),
    "/workspace/src",
    "/workspace/packages/openpi-client/src",
    "/workspace/third_party/libero",
]


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    path_entries = []
    seen = set()
    for p in SUBPROCESS_PY_PATHS + existing.split(":"):
        if not p or p in seen:
            continue
        seen.add(p)
        path_entries.append(p)
    env["PYTHONPATH"] = ":".join(path_entries)
    return env


def wait_for_port(host: str, port: int, timeout_s: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def get_env(task, resolution: int, seed: int):
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def make_server_code(policy_config_name: str, checkpoint_dir: str, denoising_steps: int, host: str, port: int) -> str:
    return f"""
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

train_config = _config.get_config({policy_config_name!r})
policy = policy_config.create_trained_policy(
    train_config,
    {checkpoint_dir!r},
    sample_kwargs={{'num_steps': {denoising_steps}}},
    norm_stats={{}},
    pytorch_device='cuda',
)
server = WebsocketPolicyServer(policy=policy, host={host!r}, port={port}, metadata=policy.metadata)
server.serve_forever()
"""


def run_benchmark(args: argparse.Namespace) -> Dict[str, float]:
    np.random.seed(args.seed)

    server_log = "/tmp/serve_policy_benchmark.log"
    log_f = open(server_log, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", make_server_code(args.policy_config, args.checkpoint_dir, args.denoising_steps, args.host, args.port)],
        stdout=log_f,
        stderr=log_f,
        env=build_subprocess_env(),
    )

    try:
        if not wait_for_port(args.host, args.port, args.server_timeout_s):
            log_f.flush()
            with open(server_log, "r", encoding="utf-8", errors="ignore") as f:
                print("=== server log (tail) ===")
                print(f.read()[-4000:])
                print("=== end server log ===")
            raise RuntimeError("Policy server did not start")

        client = ws_policy.WebsocketClientPolicy(args.host, args.port)
        task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()

        num_tasks = 3 if args.quick else args.num_tasks
        num_trials = 3 if args.quick else args.num_trials

        total_eps = 0
        total_success = 0
        infer_ms: List[float] = []

        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env = get_env(task, args.resolution, args.seed)
            task_desc = str(task.language)
            task_success = 0

            for ep in range(num_trials):
                env.reset()
                obs = env.set_init_state(initial_states[ep % len(initial_states)])
                action_plan = collections.deque()
                step = 0

                while step < args.max_steps + args.num_steps_wait:
                    if step < args.num_steps_wait:
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                        step += 1
                        continue

                    if not action_plan:
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": task_desc,
                        }

                        out = client.infer(element)
                        timing = out.get("policy_timing", {})
                        if "infer_ms" in timing:
                            infer_ms.append(float(timing["infer_ms"]))

                        for action in out["actions"][: args.replan_steps]:
                            action_plan.append(action)

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
                    if done:
                        task_success += 1
                        total_success += 1
                        break
                    step += 1

                total_eps += 1

            print(f"Task {task_id}: {task_success}/{num_trials}")

        mean_ms = statistics.mean(infer_ms) if infer_ms else float("nan")
        p50_ms = statistics.median(infer_ms) if infer_ms else float("nan")
        p95_ms = float(np.percentile(infer_ms, 95)) if infer_ms else float("nan")
        hz = (1000.0 / mean_ms) if infer_ms and mean_ms > 0 else float("nan")
        success_rate = (100.0 * total_success / total_eps) if total_eps else 0.0

        print("\n=== SERVE_POLICY BENCHMARK ===")
        print(f"Task suite: {args.task_suite_name}")
        print(f"Denoising steps: {args.denoising_steps}")
        print(f"Episodes: {total_eps}")
        print(f"Success: {total_success}/{total_eps} ({success_rate:.1f}%)")
        print(f"Policy infer calls: {len(infer_ms)}")
        print(f"Infer latency: mean={mean_ms:.2f} ms, p50={p50_ms:.2f} ms, p95={p95_ms:.2f} ms, hz={hz:.2f}")

        return {
            "episodes": float(total_eps),
            "success_rate": success_rate,
            "mean_ms": mean_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "hz": hz,
        }
    finally:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log_f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark serve_policy pathway on LIBERO")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy_config", default="pi05_libero")
    parser.add_argument("--checkpoint_dir", default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--denoising_steps", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Run 3 tasks x 3 episodes")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--server_timeout_s", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
