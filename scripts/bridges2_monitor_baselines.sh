#!/bin/bash
set -euo pipefail

HOST="${BRIDGES2_HOST:-bridges2}"
REMOTE_ROOT="${BRIDGES2_ROOT:-/ocean/projects/cis250038p/lwang31/project_Lumina}"
POLL_SECONDS="${POLL_SECONDS:-30}"
JOB_NAME="${BRIDGES2_JOB_NAME:-lumina-baseline}"
CONDITIONS=(none prompt_fast random_injection)

declare -A JOB_IDS
declare -A STATES
declare -A DETAILS

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

notify_user() {
    local message="$1"
    printf '[%s] %s\n' "$(timestamp)" "$message"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "Bridges-2 Baselines" "$message" >/dev/null 2>&1 || true
    fi
    printf '\a'
}

run_remote() {
    local cmd="$1"
    local quoted
    printf -v quoted '%q' "$cmd"
    ssh "${HOST}" "bash -lc ${quoted}"
}

fetch_job_id() {
    local condition="$1"
    local cmd="
log='${REMOTE_ROOT}/results/project_Lumina/launch_logs/${condition}.submit.log'
if [[ -f \"\$log\" ]]; then
  grep -Eo 'Submitted batch job [0-9]+' \"\$log\" | awk '{print \$4}' | tail -n 1
fi
"
    run_remote "$cmd" 2>/dev/null || true
}

fetch_job_status() {
    local job_id="$1"
    local cmd="
line=\$(squeue -h -j '${job_id}' -o '%T|%M|%R' 2>/dev/null | head -n 1 || true)
if [[ -n \"\$line\" ]]; then
  echo \"\$line\"
  exit 0
fi
sacct -n -P -j '${job_id}' --format=JobIDRaw,State,Elapsed,Start,End 2>/dev/null | awk -F'|' '\$1==\"${job_id}\" {print \$2 \"|\" \$3 \"|\" \$4 \"|\" \$5; exit}'
"
    run_remote "$cmd" 2>/dev/null || true
}

fetch_progress_line() {
    local job_id="$1"
    local cmd="
out='${REMOTE_ROOT}/${JOB_NAME}-${job_id}.out'
if [[ -f \"\$out\" ]]; then
  grep -E 'Rollout [0-9]+/[0-9]+|step [0-9]+/[0-9]+|SUMMARY|Results saved to|Environment closed\\. Done\\.' \"\$out\" | tail -n 1
fi
"
    run_remote "$cmd" 2>/dev/null || true
}

all_jobs_known() {
    local condition
    for condition in "${CONDITIONS[@]}"; do
        [[ -n "${JOB_IDS[$condition]:-}" ]] || return 1
    done
}

all_jobs_finished() {
    local condition state
    for condition in "${CONDITIONS[@]}"; do
        state="${STATES[$condition]:-UNKNOWN}"
        case "$state" in
            COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|NODE_FAIL|PREEMPTED)
                ;;
            *)
                return 1
                ;;
        esac
    done
}

notify_user "Watching Bridges-2 baseline submissions for ${CONDITIONS[*]}"

while ! all_jobs_known; do
    for condition in "${CONDITIONS[@]}"; do
        if [[ -n "${JOB_IDS[$condition]:-}" ]]; then
            continue
        fi

        job_id="$(fetch_job_id "$condition" | tr -d '[:space:]')"
        if [[ -n "$job_id" ]]; then
            JOB_IDS["$condition"]="$job_id"
            notify_user "Job submitted for ${condition}: ${job_id}"
        fi
    done

    if ! all_jobs_known; then
        sleep "${POLL_SECONDS}"
    fi
done

notify_user "All three baseline jobs have job IDs. Monitoring state changes."

while true; do
    for condition in "${CONDITIONS[@]}"; do
        job_id="${JOB_IDS[$condition]}"
        status="$(fetch_job_status "$job_id")"

        if [[ -z "$status" ]]; then
            state="UNKNOWN"
            detail="no Slurm status yet"
        else
            IFS='|' read -r state field2 field3 field4 <<< "$status"
            detail="${field2:-}"
            if [[ -n "${field3:-}" ]]; then
                detail="${detail} ${field3}"
            fi
            if [[ -n "${field4:-}" ]]; then
                detail="${detail} ${field4}"
            fi
            detail="$(echo "$detail" | xargs)"
        fi

        if [[ "${STATES[$condition]:-}" != "$state" || "${DETAILS[$condition]:-}" != "$detail" ]]; then
            STATES["$condition"]="$state"
            DETAILS["$condition"]="$detail"
            notify_user "${condition}: ${state}${detail:+ (${detail})}"
        fi

        progress="$(fetch_progress_line "$job_id")"
        if [[ -n "$progress" ]]; then
            printf '[%s] %s progress: %s\n' "$(timestamp)" "$condition" "$progress"
        fi
    done

    if all_jobs_finished; then
        notify_user "All baseline jobs finished."
        break
    fi

    sleep "${POLL_SECONDS}"
done
