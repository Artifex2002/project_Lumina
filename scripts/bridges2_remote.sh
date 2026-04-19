#!/bin/bash
set -euo pipefail

HOST="${BRIDGES2_HOST:-bridges2}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 '<remote command>'" >&2
    exit 1
fi

exec ssh "${HOST}" "$@"
