#!/usr/bin/env bash
set -euo pipefail

umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS="${ROOT}/runs/fraction_50_extended/status.json"
COMPLETED="${ROOT}/runs/fraction_50_extended/stage2/completed.json"
RUNNER="${ROOT}/evaluation/run_official_fraction.sh"
WAIT_LOG="${ROOT}/evaluation/logs/fraction_50_extended_waiter.log"

mkdir -p "${ROOT}/evaluation/logs"
exec >"${WAIT_LOG}" 2>&1

echo "Waiting for the 50% extended Stage-2 checkpoint..."
while true; do
  state="$(jq -r '.jobs.stage2_50_extended.state // .state // "unknown"' "${STATUS}")"
  case "${state}" in
    completed)
      break
      ;;
    failed)
      echo "Stage 2 failed; evaluation will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done

test -f "${COMPLETED}"
test -f "${ROOT}/runs/fraction_50_extended/stage2/step_300/adapter_config.json"

echo "Stage 2 completed; launching the matched official evaluation."
exec "${RUNNER}" 50_extended 0
