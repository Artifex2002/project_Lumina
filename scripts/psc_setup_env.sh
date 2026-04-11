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
LIBERO_ROOT="${LIBERO_ROOT:-$PSC_ROOT/external/LIBERO}"
REQ_FILE="${SCRIPT_DIR}/psc_runtime_requirements.txt"

echo "==> Loading PSC AI module: ${AI_MODULE}"
module load "${AI_MODULE}"

if [[ -z "${AI_ENV:-}" ]]; then
    echo "ERROR: AI_ENV is not set after loading ${AI_MODULE}."
    echo "Inspect Bridges-2 modules with: module spider AI"
    exit 1
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
    echo "==> Cloning PSC AI environment into ${ENV_PREFIX}"
    source activate "${AI_ENV}"
    conda create --prefix "${ENV_PREFIX}" --clone "${AI_ENV}" -y
fi

echo "==> Activating ${ENV_PREFIX}"
source activate "${ENV_PREFIX}"

echo "==> Installing pinned runtime overlay"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

mkdir -p "$(dirname "${LIBERO_ROOT}")"
if [[ ! -d "${LIBERO_ROOT}/.git" ]]; then
    echo "==> Cloning LIBERO into ${LIBERO_ROOT}"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_ROOT}"
else
    echo "==> Updating existing LIBERO clone"
    git -C "${LIBERO_ROOT}" pull --ff-only origin main
fi

echo "==> Installing LIBERO editable package"
python -m pip install --no-deps -e "${LIBERO_ROOT}"

echo "==> Creating robosuite private macros to disable brittle numba disk caching"
python - <<'PY'
import site
from pathlib import Path

candidates = []
for site_dir in site.getsitepackages():
    candidates.append(Path(site_dir) / "robosuite")

robosuite_dir = next((p for p in candidates if p.exists()), None)
if robosuite_dir is None:
    raise SystemExit("Could not locate robosuite inside site-packages")

macros_private = robosuite_dir / "macros_private.py"
macros_private.write_text(
    '"""PSC-specific robosuite overrides."""\n'
    "CACHE_NUMBA = False\n",
    encoding="ascii",
)
print(f"Wrote {macros_private}")
PY

echo "==> Environment bootstrap complete"
echo "Repo root   : ${REPO_ROOT}"
echo "Env prefix  : ${ENV_PREFIX}"
echo "LIBERO root : ${LIBERO_ROOT}"
echo "Next step   : run scripts/psc_preflight.sh inside an interactive GPU session"
