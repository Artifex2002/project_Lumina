#!/bin/bash
set -euo pipefail

HOST="${BRIDGES2_HOST:-bridges2}"

if [[ $# -eq 0 ]]; then
    exec ssh "${HOST}"
fi

exec ssh "${HOST}" "$@"
