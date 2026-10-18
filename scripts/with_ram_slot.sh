#!/usr/bin/env bash
# Run a heavy CPU job (> ~1 GB) in one of N global RAM slots, under a hard cgroup cap.
# Manager rule 2026-10-14: during the ALISIOS nightly wrf_gpu2 stays <= 15 GB total, so only
# N (file /tmp/wrf_gpu2_ram_slots/slots, default 2; 1 while MemAvailable < 40 GiB) heavy jobs may run at once, each capped at MemoryMax (default 6G, no swap).
# FIFO: waiters take a ticket; only the oldest live ticket may grab a free slot (flock alone starved waiters).
# Usage: scripts/with_ram_slot.sh [--label L] [--max 6G] [--timeout SECONDS] -- cmd args...
set -euo pipefail
LABEL=unlabelled MAX=6G TIMEOUT=14400
while [[ $# -gt 0 ]]; do
  case $1 in
    --label) LABEL=$2; shift 2;;
    --max) MAX=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;;
    --) shift; break;;
    *) echo "with_ram_slot: unknown option $1" >&2; exit 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "with_ram_slot: no command" >&2; exit 2; }
DIR=/tmp/wrf_gpu2_ram_slots; Q=$DIR/queue; mkdir -p "$Q"
# Slot count: manager-controlled file (default 2), capped by memory: at most floor((MemAvailable - 16 GiB) / 6 GiB) slots, min 1
# (16 GiB stays free for ALISIOS/OS; manager 2026-10-17: the old fixed 40 GiB -> 1 slot rule serialised the whole wave). slots=0 stays 0.
slots_now() {
  local n avail_kb
  n=$(cat "$DIR/slots" 2>/dev/null || echo "${WRF_GPU2_RAM_SLOTS:-2}"); [[ $n =~ ^[0-9]+$ ]] || n=2
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  local cap=$(( (avail_kb/1048576 - 16) / 6 )); (( cap < 1 )) && cap=1
  (( n > cap )) && n=$cap
  echo "$n"
}
TICKET="$Q/$(date +%s%N)-$$"
printf '%s\n' "$LABEL" > "$TICKET"
trap 'rm -f "$TICKET"' EXIT
deadline=$(( SECONDS + TIMEOUT ))
oldest_live() {  # print the oldest ticket whose owner pid is alive; drop dead tickets
  local t pid
  for t in $(ls "$Q" 2>/dev/null | sort); do
    pid=${t##*-}
    if kill -0 "$pid" 2>/dev/null; then echo "$Q/$t"; return; fi
    rm -f "$Q/$t"
  done
}
while :; do
  if [[ "$(oldest_live)" == "$TICKET" ]]; then
    for i in $(seq 1 "$(slots_now)"); do
      exec 8>"$DIR/slot$i.lock"
      if flock -n 8; then
        rm -f "$TICKET"
        printf 'label=%s pid=%s since=%s max=%s cmd=%s\n' "$LABEL" "$$" "$(date -u +%FT%TZ)" "$MAX" "$*" > "$DIR/slot$i.holder"
        echo "[with_ram_slot] $LABEL got slot $i (MemoryMax=$MAX, no swap)" >&2
        rc=0
        systemd-run --user --scope --quiet -p MemoryMax="$MAX" -p MemorySwapMax=0 -- "$@" || rc=$?
        : > "$DIR/slot$i.holder"
        exit $rc
      fi
      exec 8>&-
    done
  fi
  (( SECONDS < deadline )) || { echo "[with_ram_slot] $LABEL TIMED OUT waiting for a RAM slot" >&2; exit 124; }
  sleep 5
done
