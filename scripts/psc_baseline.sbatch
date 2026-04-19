#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J lumina-baseline
#SBATCH -o %x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PSC_ROOT="${PSC_ROOT:-${PROJECT:-}}"

if [[ -z "${PSC_ROOT}" ]]; then
    echo "ERROR: Set PSC_ROOT to your PSC project path, or submit from a shell where PROJECT is defined."
    exit 1
fi

AI_MODULE="${AI_MODULE:-AI/pytorch_23.02-1.13.1-py3}"
ENV_PREFIX="${ENV_PREFIX:-$PSC_ROOT/conda_envs/project_Lumina_psc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PSC_ROOT/results/project_Lumina}"
HF_HOME="${HF_HOME:-$PSC_ROOT/cache/huggingface}"
DEVICE="${DEVICE:-cuda}"
TASK_IDX="${TASK_IDX:-0}"
CONDITION="${CONDITION:-none}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
SEED="${SEED:-42}"

activate_conda_env() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$1"
        return
    fi

    if [[ -n "${CONDA_EXE:-}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
        conda activate "$1"
        return
    fi

    echo "ERROR: Could not find a usable conda command after loading ${AI_MODULE}."
    exit 1
}

module load "${AI_MODULE}"
activate_conda_env "${ENV_PREFIX}"

mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

JOB_OUTPUT_DIR="${OUTPUT_ROOT}/task${TASK_IDX}/${CONDITION}"
mkdir -p "${JOB_OUTPUT_DIR}"

echo "==> Bridges-2 batch run"
echo "SLURM_JOB_ID : ${SLURM_JOB_ID:-unknown}"
echo "Repo root    : ${REPO_ROOT}"
echo "Env prefix   : ${ENV_PREFIX}"
echo "Output dir   : ${JOB_OUTPUT_DIR}"
echo "Condition    : ${CONDITION}"
echo "Task index   : ${TASK_IDX}"
echo "Device       : ${DEVICE}"
echo "MUJOCO_GL    : ${MUJOCO_GL}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

cd "${REPO_ROOT}"

python baseline_eval.py \
    --device "${DEVICE}" \
    --task_idx "${TASK_IDX}" \
    --condition "${CONDITION}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --max_steps "${MAX_STEPS}" \
    --seed "${SEED}" \
    --output_dir "${JOB_OUTPUT_DIR}"
