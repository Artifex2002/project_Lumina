#!/bin/bash
set -euo pipefail

HOST="${BRIDGES2_HOST:-bridges2}"
REMOTE_REPO="${BRIDGES2_REPO:-/ocean/projects/cis250038p/lwang31/project_Lumina}"

if hostname -f 2>/dev/null | grep -q 'bridges2\.psc\.edu\|\.bridges2\.psc\.edu'; then
    if [[ $# -eq 0 ]]; then
        cd "${REMOTE_REPO}"
        exec bash -l
    fi

    cd "${REMOTE_REPO}"
    exec "$@"
fi

if [[ $# -eq 0 ]]; then
    exec ssh -t "${HOST}" "cd '${REMOTE_REPO}' && exec bash -l"
fi

REMOTE_CMD="$(printf "%q " "$@")"
exec ssh -t "${HOST}" "cd '${REMOTE_REPO}' && bash -lc ${REMOTE_CMD@Q}"
