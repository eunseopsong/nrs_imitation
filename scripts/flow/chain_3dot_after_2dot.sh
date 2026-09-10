#!/usr/bin/env bash
# Wait for the running 2-dot FLOW training to finish, then start the 3-dot run.
set -uo pipefail
cd "$(dirname "$0")/../.."
WAIT_PID="${1:?usage: chain_3dot_after_2dot.sh <2dot_train_pid>}"
echo "[chain $(date '+%F %T')] waiting for 2-dot training pid $WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[chain $(date '+%F %T')] pid $WAIT_PID exited -- starting 3-dot FLOW training"
TS=$(date +%Y%m%d_%H%M)
echo "[chain] TS=$TS  log=logs/flow_train_3dot_calibpatch_${TS}.log  ckpt=checkpoints/flow/polishing/single_cam/${TS}_3dot_calibpatch"
bash scripts/flow/run_3dot_calibpatch.sh "$TS" > "logs/flow_train_3dot_calibpatch_${TS}.log" 2>&1
echo "[chain $(date '+%F %T')] 3-dot training exited ($?)"
