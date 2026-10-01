#!/usr/bin/env bash
# Config-free A entry, using the original motion-only checkpoint.
# The launch supplies F0/protection defaults; drivers and sensor setup stay separate.
set -e

if (( $# > 1 )); then
    printf 'Usage: %s [session_id]\n' "$0" >&2
    exit 64
fi
nrs_a_session="${1:-E2_A_PILOT_01}"
if [[ ! "$nrs_a_session" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
    printf 'Invalid session_id: use 1-80 letters, digits, underscores, dots or hyphens.\n' >&2
    exit 64
fi

source /home/eunseop/nrs_imitation/scripts/e2_A_env.sh

cd /home/eunseop/nrs_imitation
exec ros2 launch nrs_imitation e2_a.launch.py \
    mode:=run "session:=$nrs_a_session"
