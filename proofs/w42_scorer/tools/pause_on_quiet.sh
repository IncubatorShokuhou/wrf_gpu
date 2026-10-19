#!/usr/bin/env bash
# Run CMD in its own process group; SIGSTOP the whole group while /tmp/wrf_gpu2_quiet exists, SIGCONT when it is gone
# (AGENTS QUIET rule for long CPU/disk work during foreign timed arms). Peak RSS and CPU time are unaffected by pauses;
# wall time is not a timing number. Prints paused seconds to stderr at the end. Exit code = CMD's.
# usage: pause_on_quiet.sh CMD ARGS...
set -u
Q=/tmp/wrf_gpu2_quiet
while [ -e "$Q" ]; do sleep 1; done
set -m
"$@" &
pid=$!
set +m
paused=0 stopped=0
trap 'kill -TERM -- -"$pid" 2>/dev/null; kill -CONT -- -"$pid" 2>/dev/null' TERM INT
while kill -0 "$pid" 2>/dev/null; do
  if [ -e "$Q" ]; then
    if [ $stopped = 0 ]; then kill -STOP -- -"$pid" 2>/dev/null; stopped=1; echo "[pause_on_quiet] STOP $(date -u +%T)" >&2; fi
    paused=$((paused + 1))
  elif [ $stopped = 1 ]; then
    kill -CONT -- -"$pid" 2>/dev/null; stopped=0; echo "[pause_on_quiet] CONT $(date -u +%T)" >&2
  fi
  sleep 1
done
wait "$pid"; rc=$?
echo "[pause_on_quiet] done rc=$rc paused_s=$paused" >&2
exit $rc
