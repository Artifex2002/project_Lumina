# Bridges-2 Runbook

This repo now includes a concrete Bridges-2 single-V100 workflow for getting the SmolVLA + LIBERO baseline onto PSC with the least amount of moving parts.

## SSH Access From This Machine

Your local SSH config now has these Bridges-2 aliases:

- `bridges2` or `bridges2-login`: log in to PSC's Bridges-2 login host at `bridges2.psc.edu`
- `bridges2-gpu-shared`: start an interactive `GPU-shared` allocation for `1x v100-32` and drop into that session
- `bridges2-v007`: jump through the login host to `v007.bridges2.psc.edu`

Important operational note:

- PSC's documented workflow is `laptop -> login node -> interactive or batch job -> compute node`
- a fixed compute node alias like `bridges2-v007` is only useful if Slurm has actually placed your job on that node
- for normal development and CLI work, use `bridges2` and `bridges2-gpu-shared`

Convenience wrappers were added under [`scripts/`](scripts):

- [`scripts/bridges2_login.sh`](scripts/bridges2_login.sh): open a shell on the Bridges-2 login node or run a one-off login-node command
- [`scripts/bridges2_remote.sh`](scripts/bridges2_remote.sh): run an arbitrary remote command on the login node
- [`scripts/bridges2_gpu_shell.sh`](scripts/bridges2_gpu_shell.sh): request an interactive GPU session through `interact`
- [`scripts/bridges2_repo_shell.sh`](scripts/bridges2_repo_shell.sh): open a shell directly in the remote `project_Lumina` checkout

Examples:

```bash
ssh bridges2
ssh bridges2-gpu-shared
bash scripts/bridges2_remote.sh 'hostname && pwd && squeue -u $USER'
bash scripts/bridges2_gpu_shell.sh
bash scripts/bridges2_repo_shell.sh
```

Current local public key fingerprint:

```text
SHA256:rtbK5q7BswFexzjHPrpno9YRqi4MuCFVfHodabT/VDI
```

The key is present locally as `~/.ssh/id_ed25519`, but live auth testing showed that PSC is not currently accepting it for `lwang31`. Until PSC has that key associated with your account, SSH will fall back to password auth.

Live verification from this machine:

- login host reached successfully as `lwang31`
- project repo found at `/ocean/projects/cis250038p/lwang31/project_Lumina`
- short GPU allocation succeeded on `GPU-shared`
- allocated node reported `v014.ib.bridges2.psc.edu`
- `nvidia-smi` saw `Tesla V100-SXM2-32GB`

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
   - on the current Bridges-2 install, the PyTorch path is `AI/pytorch_23.02-1.13.1-py3`
   - set `AI_MODULE` to the exact module version you want to use
2. Define your PSC working root if `$PROJECT` is not already set:
   - `export PSC_ROOT=/ocean/projects/<grant>/<username>`
3. Clone the repo:
   - `git clone https://github.com/Artifex2002/project_Lumina.git ${PSC_ROOT:-$PROJECT}/project_Lumina`
4. Bootstrap the environment:
   - `cd ${PSC_ROOT:-$PROJECT}/project_Lumina`
   - `AI_MODULE=AI/pytorch_23.02-1.13.1-py3 bash scripts/psc_setup_env.sh`

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
sbatch --export=ALL,AI_MODULE=AI/pytorch_23.02-1.13.1-py3,CONDITION=none,TASK_IDX=0 scripts/psc_baseline.sbatch
sbatch --export=ALL,AI_MODULE=AI/pytorch_23.02-1.13.1-py3,CONDITION=prompt_fast,TASK_IDX=0 scripts/psc_baseline.sbatch
sbatch --export=ALL,AI_MODULE=AI/pytorch_23.02-1.13.1-py3,CONDITION=random_injection,TASK_IDX=0 scripts/psc_baseline.sbatch
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

## Security And Operational Notes

- PSC says you connect to `bridges2.psc.edu` first and then reach compute resources through interactive or batch jobs. Do not treat compute nodes as permanent SSH targets.
- PSC uses one PSC password across its production systems. That means password reuse is a bigger risk than on a single machine.
- The SSH config explicitly sets `ForwardAgent no`. Keep it that way unless you have a very specific reason, because agent forwarding lets remote systems ask your local agent to authenticate elsewhere.
- Login nodes are shared infrastructure. Do editing, Git, environment management, light compilation, and job submission there. Do not run heavy training or evaluation on the login node.
- Prefer Git or `rsync` for code and data movement. Avoid ad hoc copies of secrets, caches, and checkpoints into random shared directories.
- Anything under project or shared filesystems should be treated as institutional infrastructure, not as a private workstation. Keep API tokens in shell startup files or secret stores you control, not in repo files.
- Before trusting a new host key prompt, verify that it is really PSC. A changed host key on an existing alias is worth investigating.
- For day-to-day development, use the login node for editing and a Slurm allocation for GPU execution. If you need a persistent browser IDE, PSC OnDemand is usually the supported path.

## V100 And Future Video Capture

The V100 does not have RT cores, but that is not a blocker for MuJoCo / robosuite video capture.

- MuJoCo offscreen rendering uses OpenGL / EGL, not RTX ray tracing
- MP4 writing is a separate CPU / ffmpeg-style encoding step

So the lack of RT cores does not prevent future rollout-video recording for this project.
