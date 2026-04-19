#!/bin/bash
set -euo pipefail

HOST="${BRIDGES2_HOST:-bridges2}"
PARTITION="${BRIDGES2_PARTITION:-GPU-shared}"
GPU_TYPE="${BRIDGES2_GPU_TYPE:-v100-32}"
GPU_COUNT="${BRIDGES2_GPU_COUNT:-1}"
WALLTIME="${BRIDGES2_TIME:-01:00:00}"

REMOTE_CMD="interact -p ${PARTITION} --gres=gpu:${GPU_TYPE}:${GPU_COUNT} -t ${WALLTIME}"

if hostname -f 2>/dev/null | grep -q 'bridges2\.psc\.edu\|\.bridges2\.psc\.edu'; then
    exec bash -lc "${REMOTE_CMD}"
fi

exec ssh -tt "${HOST}" "bash -lc '${REMOTE_CMD}'"
