# Bridges-2 Runbook

This repo now includes a concrete Bridges-2 single-V100 workflow for getting the SmolVLA + LIBERO baseline onto PSC with the least amount of moving parts.

## What This Setup Assumes

- One GPU on `GPU-shared`
- `v100-32:1`
- GitHub is the source of truth for code updates
- LIBERO is installed from a separate editable clone under `PSC_ROOT/external/LIBERO`
- Results and caches live under `PSC_ROOT`, not inside the repo
- `PSC_ROOT` means your Bridges-2 project directory. If your shell already defines `$PROJECT`, the scripts will use that automatically.

## Why This Path

- The current baseline evaluator needed an explicit CUDA path. [`baseline_eval.py`](baseline_eval.py) now supports `--device auto|cpu|cuda|mps`.
- The current stack is sensitive to `robosuite` numba disk caching. The setup script creates `robosuite/macros_private.py` with `CACHE_NUMBA = False` so PSC runs do not depend on fragile numba cache behavior.
- Bridges-2 batch jobs should use activated shell environments, not `conda run`.

## Estimated Runtime

For the current full task-0 experiment:

- `3 conditions x 10 rollouts x 600 max steps = 18,000 max environment-policy steps`
- Conservative steady-state estimate on one V100: `45 to 105 minutes`
- First full cluster run should reserve `3 hours` to absorb model download, cache warm-up, and cluster bring-up issues

Bridges-2 defines GPU usage in GPU-hours, where `1 GPU-hour = 1 SU`. Source: [PSC Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide/).

## One-Time Setup On Bridges-2

1. Inspect and choose an AI module:
   - `module spider AI`
   - set `AI_MODULE` to the exact module version you want to use
2. Define your PSC working root if `$PROJECT` is not already set:
   - `export PSC_ROOT=/ocean/projects/<grant>/<username>`
3. Clone the repo:
   - `git clone https://github.com/Artifex2002/project_Lumina.git ${PSC_ROOT:-$PROJECT}/project_Lumina`
4. Bootstrap the environment:
   - `cd ${PSC_ROOT:-$PROJECT}/project_Lumina`
   - `AI_MODULE=<your-chosen-module> bash scripts/psc_setup_env.sh`

The setup script will:

- clone PSC's `AI_ENV` into `PSC_ROOT/conda_envs/project_Lumina_psc`
- install the pinned runtime overlay from [`scripts/psc_runtime_requirements.txt`](scripts/psc_runtime_requirements.txt)
- clone and install LIBERO editable without letting it rewrite the rest of the environment
- disable brittle `robosuite` numba disk caching

## Smoke Test

Run this from an interactive GPU session:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 01:00:00
export PSC_ROOT=${PSC_ROOT:-$PROJECT}
cd $PSC_ROOT/project_Lumina
AI_MODULE=<your-chosen-module> bash scripts/psc_preflight.sh
```

The preflight does four things:

- verifies CUDA visibility
- sets `MUJOCO_GL=egl`
- primes Hugging Face caches under `PSC_ROOT/cache/huggingface`
- runs a tiny baseline smoke test: `1 rollout`, `5 max steps`

## Full Batch Run

Submit one condition at a time first:

```bash
export PSC_ROOT=${PSC_ROOT:-$PROJECT}
cd $PSC_ROOT/project_Lumina
sbatch --export=ALL,AI_MODULE=<your-chosen-module>,CONDITION=none,TASK_IDX=0 scripts/psc_baseline.sbatch
sbatch --export=ALL,AI_MODULE=<your-chosen-module>,CONDITION=prompt_fast,TASK_IDX=0 scripts/psc_baseline.sbatch
sbatch --export=ALL,AI_MODULE=<your-chosen-module>,CONDITION=random_injection,TASK_IDX=0 scripts/psc_baseline.sbatch
```

Results will be written under:

- `PSC_ROOT/results/project_Lumina/task0/none/`
- `PSC_ROOT/results/project_Lumina/task0/prompt_fast/`
- `PSC_ROOT/results/project_Lumina/task0/random_injection/`

## Updating PSC When GitHub Changes

Use Git on PSC, not `scp`, as the normal update path.

If the PSC clone has no local edits:

```bash
export PSC_ROOT=${PSC_ROOT:-$PROJECT}
cd $PSC_ROOT/project_Lumina
git pull --ff-only origin main
```

`--ff-only` means Git will only update the branch if it can move straight forward safely. If PSC has diverged local commits, Git will stop instead of creating a surprise merge commit.

Use `scp` or `rsync` for:

- large result directories
- large Hugging Face caches
- one-off backups

## V100 And Future Video Capture

The V100 does not have RT cores, but that is not a blocker for MuJoCo / robosuite video capture.

- MuJoCo offscreen rendering uses OpenGL / EGL, not RTX ray tracing
- MP4 writing is a separate CPU / ffmpeg-style encoding step

So the lack of RT cores does not prevent future rollout-video recording for this project.
