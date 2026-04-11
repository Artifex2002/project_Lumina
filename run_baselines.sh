#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_baselines.sh — Run all 3 baseline conditions for LIBERO-Long task 0
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x run_baselines.sh
#   conda activate project_Lumina
#   ./run_baselines.sh
#
# Each condition writes results to baseline_results/
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # exit on first error

if [[ "${CONDA_DEFAULT_ENV:-}" != "project_Lumina" ]]; then
    echo "ERROR: Activate the project_Lumina environment before running this script."
    echo "  conda activate project_Lumina"
    exit 1
fi

echo "========================================"
echo " Project Lumina — Baseline Evaluation"
echo "========================================"
echo ""

# ── Condition 1: No steering (baseline) ──────────────────────────────────────
echo ">>> [1/3] Running: condition=none"
python baseline_eval.py \
    --condition none \
    --task_idx 0 \
    --num_rollouts 10 \
    --device auto
echo ""

# ── Condition 2: Prompt-based fast steering ──────────────────────────────────
echo ">>> [2/3] Running: condition=prompt_fast"
python baseline_eval.py \
    --condition prompt_fast \
    --task_idx 0 \
    --num_rollouts 10 \
    --device auto
echo ""

# ── Condition 3: Random neuron injection ─────────────────────────────────────
echo ">>> [3/3] Running: condition=random_injection"
python baseline_eval.py \
    --condition random_injection \
    --task_idx 0 \
    --num_rollouts 10 \
    --device auto
echo ""

echo "========================================"
echo " All baselines complete!"
echo " Results in: baseline_results/"
echo "========================================"
