# w44-rc v0.4.4 release session: shared bash helpers (sourced by the entries and the session payloads; never run).
# No JAX/gpuwrf import anywhere in here. Paths are absolute; every rm is on a literal or ${VAR:?}-guarded path.
W44R_TOOLS=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5
W44R_LANE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc
W44R_SCRATCH=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc
W44R_PY=<USER_HOME>/miniconda3/bin/python3
W44R_REAL_UTC=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
W44R_GPU_LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
# CPU follow-ups only. GPU entries use dedicated systemd scopes.
W44R_RAM_SLOT=<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh

W44R_MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
W44R_DATA_FLOOR_GIB=365          # manager 2026-10-24T15:45Z (ALISIOS target >= 360): refuse when free - forecast < 365 GiB
W44R_DATA_MARGIN_GIB=0
W44R_ROOT_MIN_GIB=3              # / is 98 % full: receipts only (KB-MB) go there
export PATH=<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin   # cuobjdump (E41) is in /usr/local/cuda/bin
export LC_ALL=C

w44_log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${W44R_LOG:-/dev/null}"; }

w44_quiet() {  # 0 when the global QUIET flag is set
  [ -e /tmp/wrf_gpu2_quiet ]
}

w44_real_now() {  # real UTC epoch (chrony-corrected, manager tool); refuses a non-numeric answer
  local e; e=$("$W44R_REAL_UTC") || return 7
  [[ $e =~ ^[0-9]+$ ]] || return 7
  echo "$e"
}

w44_free_gib() {  # free GiB of the filesystem holding $1
  df --output=avail -B1G "$1" | tail -1 | tr -d ' '
}

w44_disk_ok() {  # w44_disk_ok PRODUCTION_FORECAST_GIB -> 0 iff <DATA_ROOT> free - forecast >= 365 GiB (floor)
  # Same code path in both modes; only the forecast differs: GPU mode = the production forecast (arm/slot table, entry
  # header); CPU smoke = W44R_SMOKE_FORECAST_GIB (what the smoke itself writes, x1.5 margin) and a logged dry evaluation of
  # the production arithmetic (manager 2026-10-24: smoke receipts must show what GPU mode would decide).
  local free used=$1 need
  free=$(w44_free_gib <DATA_ROOT>) || return 1
  if [[ ${W44R_LAUNCH_SMOKE:-0} == 1 ]]; then
    used=${W44R_SMOKE_FORECAST_GIB:?smoke needs W44R_SMOKE_FORECAST_GIB}
    need=$(( W44R_DATA_FLOOR_GIB + W44R_DATA_MARGIN_GIB + $1 ))
    w44_log "DISK-GUARD smoke: free ${free} GiB - smoke forecast ${used} GiB -> $( (( free - used >= W44R_DATA_FLOOR_GIB + W44R_DATA_MARGIN_GIB )) && echo PASS || echo REFUSE ); GPU mode would need >= ${need} GiB (production forecast $1 GiB), now ${free} -> would $( (( free >= need )) && echo PASS || echo REFUSE )"
  fi
  (( free - used >= W44R_DATA_FLOOR_GIB + W44R_DATA_MARGIN_GIB ))
}

w44_sha() { sha256sum "$1" | cut -d' ' -f1; }
