#!/usr/bin/env bash
# Run queued lane jobs sequentially inside ONE RAM slot. usage: slot_worker.sh JOBS_FILE
# JOBS_FILE: one shell command per line (data file, re-read every iteration; append-only).
# Done markers: JOBS_FILE.done (line numbers); job logs: JOBS_FILE.<n>.log. Exits when no job is pending.
JOBS=$1
touch "$JOBS.done"
while :; do
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  n=0; next=
  while IFS= read -r line; do
    n=$((n + 1))
    [[ -z "$line" || "$line" = \#* ]] && continue
    grep -qx "$n" "$JOBS.done" && continue
    next=$line; break
  done < "$JOBS"
  [ -z "$next" ] && { echo "$(date -u +%T) no pending job"; exit 0; }
  echo "$(date -u +%T) start $n: $next"
  bash -c "$next" > "$JOBS.$n.log" 2>&1
  echo "$(date -u +%T) end $n rc=$?"
  echo "$n" >> "$JOBS.done"
done
