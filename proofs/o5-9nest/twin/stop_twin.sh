#!/bin/bash
# Stop the #148 CPU twin within the ALISIOS 5-minute return rule: touch ROOT/STOP (the supervisor tears down the whole MPI
# session, writes outputs/registry end); if it is not finished after 240 s, TERM then KILL every process of the recorded MPI
# session (SID from ROOT/PROCESS.json) -- never a broad pkill.
set -u
ROOT=${1:?usage: stop_twin.sh ROOT}
touch "$ROOT/STOP"
for i in $(seq 1 48); do [ -s "$ROOT/cpu_run.json" ] && { echo "stopped cleanly: $(cat "$ROOT/cpu_run.json" | head -c 300)"; exit 0; }; sleep 5; done
SID=$(python3 -c "import json;print(json.load(open('$ROOT/PROCESS.json'))['sid'])")
for sig in TERM KILL; do
  for p in $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s{print $1}'); do kill -$sig "$p" 2>/dev/null; done; sleep 10
done
echo "forced stop of session $SID; remaining: $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s' | wc -l)"
