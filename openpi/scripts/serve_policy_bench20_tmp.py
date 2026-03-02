import os
import time
import math
import socket
import collections
import statistics
import subprocess
import signal

import numpy as np
from openpi_client import websocket_client_policy as ws_policy
from openpi_client import image_tools
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

HOST = "127.0.0.1"
PORT = 8000

server_code = """
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
train_config = _config.get_config('pi05_libero')
policy = policy_config.create_trained_policy(
    train_config,
    '/root/.cache/openpi/checkpoints/pi05_libero',
    sample_kwargs={'num_steps': 20},
    norm_stats={},
    pytorch_device='cuda',
)
server = WebsocketPolicyServer(policy=policy, host='0.0.0.0', port=8000, metadata=policy.metadata)
server.serve_forever()
"""


def wait_port(host, port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            time.sleep(1)
    return False


def quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_env(task, resolution=256, seed=7):
    task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env_args = {'bddl_file_name': task_bddl_file, 'camera_heights': resolution, 'camera_widths': resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


np.random.seed(7)
server_log = "/tmp/serve_policy_server20.log"
_logf = open(server_log, "w")
proc = subprocess.Popen(['python', '-u', '-c', server_code], stdout=_logf, stderr=_logf)
try:
    if not wait_port(HOST, PORT):
        _logf.flush()
        try:
            with open(server_log, "r") as f:
                print("=== server log ===")
                print(f.read()[-4000:])
                print("=== end server log ===")
        except Exception:
            pass
        raise RuntimeError('Server did not start')

    client = ws_policy.WebsocketClientPolicy(HOST, PORT)
    task_suite = benchmark.get_benchmark_dict()['libero_spatial']()
    max_steps = 220
    num_steps_wait = 10
    replan_steps = 5

    total_eps = 0
    total_success = 0
    infer_ms = []

    for task_id in range(3):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env = get_env(task)
        task_desc = str(task.language)
        task_success = 0

        for ep in range(3):
            env.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            action_plan = collections.deque()
            t = 0

            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                    t += 1
                    continue

                if not action_plan:
                    img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                    element = {
                        'observation/image': img,
                        'observation/wrist_image': wrist,
                        'observation/state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])),
                        'prompt': task_desc,
                    }
                    out = client.infer(element)
                    if 'policy_timing' in out and 'infer_ms' in out['policy_timing']:
                        infer_ms.append(float(out['policy_timing']['infer_ms']))
                    action_plan.extend(out['actions'][:replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_success += 1
                    total_success += 1
                    break
                t += 1

            total_eps += 1

        print(f'Task {task_id}: {task_success}/3')

    mean_ms = statistics.mean(infer_ms) if infer_ms else float('nan')
    p50 = statistics.median(infer_ms) if infer_ms else float('nan')
    p95 = float(np.percentile(infer_ms, 95)) if infer_ms else float('nan')
    hz = (1000.0 / mean_ms) if infer_ms and mean_ms > 0 else float('nan')

    print('=== SERVE_POLICY BENCHMARK (num_steps=20) ===')
    print(f'Episodes: {total_eps}, Success: {total_success}/{total_eps} ({(100.0*total_success/total_eps):.1f}%)')
    print(f'Policy infer calls: {len(infer_ms)}')
    print(f'Infer latency: mean={mean_ms:.2f} ms, p50={p50:.2f} ms, p95={p95:.2f} ms, hz={hz:.2f}')
finally:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    _logf.close()
