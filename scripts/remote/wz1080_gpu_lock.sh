#!/usr/bin/env bash
# wz1080_gpu_lock.sh — REMOTE-side GPU lock for the wz1080 test box (installed as ~/.local/bin/wz-gpu-lock).
#
# Mirrors scripts/with_gpu_lock.sh closely enough that the product preflight's lock proof passes
# (gpuwrf.runtime.gpu_preflight: lock file /tmp/wrf_gpu2_gpu.lock, holder file <lock>.holder carrying the
# token, GPUWRF_GPU_LOCK_HELD=1, GPUWRF_GPU_LOCK_FD pointing at the lock file). No manager lease here:
# the box is a debug system; the manager can still serialize via labels/holder.
#
# Usage: wz-gpu-lock [--label NAME] [--lock-wait SECONDS] -- <command> [args...]
# Exit: command rc; 2 usage; 124 lock wait timed out.
set -euo pipefail

LOCK_FILE="${WZ_GPU_LOCK_FILE:-/tmp/wrf_gpu2_gpu.lock}"   # override only for local harness tests
HOLDER_FILE="${LOCK_FILE}.holder"
LABEL="${USER:-worker}:$$"
WAIT=14400
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --lock-wait|--timeout) WAIT="$2"; shift 2;;
    --) shift; break;;
    *) echo "wz-gpu-lock: unknown arg '$1' (missing '--'?)" >&2; exit 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "wz-gpu-lock: no command after '--'" >&2; exit 2; }

[[ -e "$LOCK_FILE" ]] || { touch "$LOCK_FILE"; chmod 666 "$LOCK_FILE" 2>/dev/null || true; }
exec 9>>"$LOCK_FILE"
# FIFO queue (flock alone is not fair): one ticket per waiter, <ns>_<pid>_<label>; only the oldest live ticket
# (or the label named in /tmp/wrf_gpu2_gpu.next, < 1 h old — the manager's priority override) may take the flock.
QDIR="${LOCK_FILE%/*}/wz_gpu_queue"
mkdir -p "$QDIR" 2>/dev/null; chmod 1777 "$QDIR" 2>/dev/null || true
TICKET="$QDIR/$(date +%s%N)_$$_${LABEL}"
: > "$TICKET"
trap 'rm -f "$TICKET"' EXIT
NEXT_FILE="${LOCK_FILE%/*}/wrf_gpu2_gpu.next"
deadline=$(( SECONDS + WAIT )); said=0
queue_head() {  # prints the ticket allowed to go next; prunes dead waiters
  local head="" nexthead="" t pid _next
  _next="$(awk 'NR==1{print $1}' "$NEXT_FILE" 2>/dev/null || true)"
  if [[ -n "$_next" ]] && (( $(date +%s) - $(stat -c %Y "$NEXT_FILE" 2>/dev/null || echo 0) >= 3600 )); then _next=""; fi
  for t in $(ls -1 "$QDIR" 2>/dev/null | sort); do
    pid="$(cut -d_ -f2 <<<"$t")"
    if ! kill -0 "$pid" 2>/dev/null; then rm -f "$QDIR/$t"; continue; fi
    [[ -z "$head" ]] && head="$t"
    [[ -n "$_next" && -z "$nexthead" && "${t#*_*_}" == "$_next" ]] && nexthead="$t"
  done
  echo "${nexthead:-$head}"
}
while :; do
  if [[ "$(queue_head)" == "${TICKET##*/}" ]]; then
    rem=$(( deadline - SECONDS )); (( rem > 0 )) || rem=0
    if ! flock -w "$rem" 9; then
      echo "[wz-gpu-lock] $LABEL TIMED OUT after ${WAIT}s; holder: $(cat "$HOLDER_FILE" 2>/dev/null)" >&2; exit 124
    fi
    # A priority label may have queued while we were blocked in flock: yield to it (like with_gpu_lock.sh).
    if [[ "$(queue_head)" == "${TICKET##*/}" ]]; then
      [[ "$(awk 'NR==1{print $1}' "$NEXT_FILE" 2>/dev/null)" == "$LABEL" ]] && rm -f "$NEXT_FILE" 2>/dev/null
      break
    fi
    flock -u 9
  elif (( !said )); then
    echo "[wz-gpu-lock] $LABEL queued ($(ls -1 "$QDIR" | wc -l) waiting); holder: $(cat "$HOLDER_FILE" 2>/dev/null)" >&2; said=1
  fi
  (( SECONDS < deadline )) || { echo "[wz-gpu-lock] $LABEL TIMED OUT in queue after ${WAIT}s" >&2; exit 124; }
  sleep 2
done
rm -f "$TICKET"

TOKEN="wzgpu-lock-$$-${RANDOM:-0}-$(date -u +%s%N)"
printf 'holder=%s pid=%s pgid=%s since=%s token=%s cmd=%s\n' "$LABEL" "$$" "$(ps -o pgid= -p $$ | tr -d ' ')" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TOKEN" "$*" > "$HOLDER_FILE"
echo "[wz-gpu-lock] $LABEL ACQUIRED -> $*" >&2
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[wz-gpu-lock] $LABEL GPU at acquire: $(nvidia-smi --query-gpu=name,memory.free,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)" >&2
  echo "[wz-gpu-lock] $LABEL other GPU apps (pid, MiB): $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr "\n" ";")" >&2
fi

export GPUWRF_GPU_LOCK_HELD=1 GPUWRF_GPU_LOCK_FD=9 GPUWRF_GPU_LOCK_FILE="$LOCK_FILE"
export GPUWRF_GPU_LOCK_HOLDER_FILE="$HOLDER_FILE" GPUWRF_GPU_LOCK_LABEL="$LABEL" GPUWRF_GPU_LOCK_TOKEN="$TOKEN"

# Keep the box awake while we hold the GPU (Fedora/GNOME may auto-suspend an idle desktop). A sidecar, because
# systemd-inhibit closes every inherited fd of its child (it would drop the lock fd 9 the preflight checks).
inhib=""
if command -v systemd-inhibit >/dev/null 2>&1; then
  systemd-inhibit --what=sleep:idle --who=wrf_gpu2 --why="wz-gpu-lock $LABEL" --mode=block \
    sh -c "while kill -0 $$ 2>/dev/null; do sleep 20; done" 9>&- 2>/dev/null &
  inhib=$!
fi

set +e
if [[ -n "${WZ_GPU_CPUS:-}" ]] && command -v taskset >/dev/null 2>&1; then
  echo "[wz-gpu-lock] $LABEL pinned to cores $WZ_GPU_CPUS" >&2
  taskset -c "$WZ_GPU_CPUS" "$@" &
else
  "$@" &
fi
child=$!
trap 'echo "[wz-gpu-lock] $LABEL forwarding signal to $child" >&2; kill -TERM "$child" 2>/dev/null' TERM INT HUP
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
trap - TERM INT HUP
[[ -n "$inhib" ]] && kill "$inhib" 2>/dev/null
: > "$HOLDER_FILE"
echo "[wz-gpu-lock] $LABEL released (rc=$rc)" >&2
exit "$rc"
