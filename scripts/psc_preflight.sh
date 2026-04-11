#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PSC_ROOT="${PSC_ROOT:-${PROJECT:-}}"

if [[ -z "${PSC_ROOT}" ]]; then
    echo "ERROR: Set PSC_ROOT to your PSC project path, or run from a shell where PROJECT is defined."
    exit 1
fi

AI_MODULE="${AI_MODULE:-AI}"
ENV_PREFIX="${ENV_PREFIX:-$PSC_ROOT/conda_envs/project_Lumina_psc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PSC_ROOT/results/project_Lumina/smoke}"
HF_HOME="${HF_HOME:-$PSC_ROOT/cache/huggingface}"
DEVICE="${DEVICE:-cuda}"
TASK_IDX="${TASK_IDX:-0}"
CONDITION="${CONDITION:-none}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MAX_STEPS="${MAX_STEPS:-5}"
SEED="${SEED:-42}"

module load "${AI_MODULE}"
source activate "${ENV_PREFIX}"

mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "==> PSC preflight"
echo "Repo root    : ${REPO_ROOT}"
echo "Env prefix   : ${ENV_PREFIX}"
echo "HF_HOME      : ${HF_HOME}"
echo "MUJOCO_GL    : ${MUJOCO_GL}"
echo "Device       : ${DEVICE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

cd "${REPO_ROOT}"

python - <<'PY'
import os
import torch

print("torch version       :", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count   :", torch.cuda.device_count())
    print("cuda device 0 name  :", torch.cuda.get_device_name(0))
print("MUJOCO_GL           :", os.environ.get("MUJOCO_GL"))
PY

python baseline_eval.py \
    --device "${DEVICE}" \
    --task_idx "${TASK_IDX}" \
    --condition "${CONDITION}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --max_steps "${MAX_STEPS}" \
    --log_every 1 \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_ROOT}"
