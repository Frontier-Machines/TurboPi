#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-turbo_pi}"
IMAGE_NAME="${IMAGE_NAME:-turbo_pi:latest}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-liangsu9988/Turbo-Pi0.5-1.1.2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/.cache/openpi/checkpoints/pi05_libero}"
LIBERO_ASSET_SENTINEL="third_party/libero/libero/libero/assets/scenes/libero_tabletop_base_style.xml"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

ensure_container() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[setup] Building image: $IMAGE_NAME"
    docker build -f Dockerfile.libero_eval -t "$IMAGE_NAME" .
  fi

  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "[setup] Creating container: $CONTAINER_NAME"
    docker run -d --name "$CONTAINER_NAME" \
      --gpus all --ipc=host \
      -v "$ROOT_DIR":/workspace \
      -v "$HOME/.cache/openpi:/root/.cache/openpi" \
      -e MUJOCO_GL=egl \
      -e MUJOCO_EGL_DEVICE_ID=0 \
      -e PYOPENGL_PLATFORM=egl \
      "$IMAGE_NAME" sleep infinity >/dev/null
  fi

  if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "[setup] Starting container: $CONTAINER_NAME"
    docker start "$CONTAINER_NAME" >/dev/null
  fi
}

echo "[setup] Validating required commands"
require_cmd git
require_cmd docker
require_cmd huggingface-cli

cd "$ROOT_DIR"

echo "[setup] Syncing LIBERO submodule"
git submodule sync --recursive
git submodule update --init --recursive third_party/libero

if [[ ! -f "$LIBERO_ASSET_SENTINEL" ]]; then
  echo "[setup] LIBERO assets missing; backfilling from upstream LIBERO"
  TMP_DIR="$(mktemp -d /tmp/libero_upstream.XXXXXX)"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$TMP_DIR/LIBERO"
  mkdir -p third_party/libero/libero/libero/assets
  cp -a "$TMP_DIR/LIBERO/libero/libero/assets/." third_party/libero/libero/libero/assets/
  rm -rf "$TMP_DIR"
fi
test -f "$LIBERO_ASSET_SENTINEL"

echo "[setup] Writing non-interactive LIBERO config"
mkdir -p .libero
cat > .libero/config.yaml <<'EOF'
benchmark_root: /workspace/third_party/libero/libero/libero
bddl_files: /workspace/third_party/libero/libero/libero/bddl_files
init_states: /workspace/third_party/libero/libero/libero/init_files
datasets: /workspace/third_party/libero/libero/datasets
assets: /workspace/third_party/libero/libero/libero/assets
EOF

if [[ ! -d "$CHECKPOINT_DIR" ]] || [[ -z "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null || true)" ]]; then
  echo "[setup] Downloading checkpoint to $CHECKPOINT_DIR"
  huggingface-cli download "$CHECKPOINT_REPO" --local-dir "$CHECKPOINT_DIR"
else
  echo "[setup] Checkpoint exists: $CHECKPOINT_DIR"
fi

ensure_container

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/benchmark_logs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

run_benchmark() {
  local label="$1"
  local script_name="$2"
  local log_file="$LOG_DIR/${label}.log"

  echo "[run] $label (6 denoising steps, quick mode)"
  docker exec "$CONTAINER_NAME" bash -lc "
    export LIBERO_CONFIG_PATH=/workspace/.libero
    cd /workspace
    python scripts/${script_name} --quick --denoising_steps 6
  " | tee "$log_file"
  echo "[done] $label log: $log_file"
}

run_benchmark "optimized_pathway" "libero_eval_full_optimized.py"
run_benchmark "serve_policy_pathway" "libero_eval_serve_policy.py"

echo "[done] Benchmarks completed."
echo "[done] Logs directory: $LOG_DIR"
