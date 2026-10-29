# v45-rc v0.4.5 release session: shared bash helpers (sourced by the entries and the session payloads; never run).
# No JAX/gpuwrf import anywhere in here. Paths are absolute; every rm is on a literal or ${VAR:?}-guarded path.
W45R_TOOLS=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1
W45R_LANE=<USER_HOME>/wrf_gpu2_lanes/v45-rc
W45R_SCRATCH=<USER_HOME>/wrf_gpu2_lanes/v45-rc/runtime
W45R_PY=<USER_HOME>/miniconda3/bin/python3
W45R_REAL_UTC=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
W45R_GPU_LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
# CPU follow-ups only. GPU entries use dedicated systemd scopes.
W45R_RAM_SLOT=<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh

W45R_MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
W45R_DATA_FLOOR_GIB=365          # manager 2026-10-24T15:45Z (ALISIOS target >= 360): refuse when free - forecast < 365 GiB
W45R_DATA_MARGIN_GIB=0
W45R_ROOT_MIN_GIB=22              # manager01:06:22: original20GiB floor +2GiB global margin
export PATH=<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin   # cuobjdump (E41) is in /usr/local/cuda/bin
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu

w45_log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${W45R_LOG:-/dev/null}"; }

w45_quiet() {  # 0 when the global QUIET flag is set
  [ -e /tmp/wrf_gpu2_quiet ]
}

w45_real_now() {  # real UTC epoch (chrony-corrected, manager tool); refuses a non-numeric answer
  local e; e=$("$W45R_REAL_UTC") || return 7
  [[ $e =~ ^[0-9]+$ ]] || return 7
  echo "$e"
}

w45_free_gib() {  # free GiB of the filesystem holding $1
  local bytes
  bytes=$(df --output=avail -B1 "$1" | tail -1) || return 1
  echo $(( bytes / 1073741824 ))
}

w45_disk_ok() {  # w45_disk_ok PRODUCTION_FORECAST_GIB -> 0 iff <DATA_ROOT> free - forecast >= 365 GiB (floor)
  # Same code path in both modes; only the forecast differs: GPU mode = the production forecast (arm/slot table, entry
  # header); CPU smoke = W45R_SMOKE_FORECAST_GIB (what the smoke itself writes, x1.5 margin) and a logged dry evaluation of
  # the production arithmetic (manager 2026-10-24: smoke receipts must show what GPU mode would decide).
  local root_free root_bytes
  W45R_DISK_REFUSAL=ROOT
  root_free=$(df --output=avail -B1 / | tail -1) || return 1
  root_bytes=$(du -sx -B1 "$W45R_LANE" | cut -f1) || return 1
  (( root_free >= W45R_ROOT_MIN_GIB * 1073741824 && root_bytes <= W45R_ROOT_CAP_GIB * 1073741824 )) || { w45_log "ROOT GUARD refused: free=$root_free used=$root_bytes"; return 1; }
  W45R_DISK_REFUSAL=DATA
  local free used=$1 need
  free=$(w45_free_gib <DATA_ROOT>) || return 1
  if [[ ${W45R_LAUNCH_SMOKE:-0} == 1 ]]; then (( root_free >= W45R_ROOT_MIN_GIB * 1073741824 + 67108864 )) || { W45R_DISK_REFUSAL=ROOT; w45_log "ROOT SMOKE RESERVE refused: free=$root_free required=$(( W45R_ROOT_MIN_GIB * 1073741824 + 67108864 ))"; return 1; }; fi  # CPU-only missing3 CLI plans:64MiBmetadata reserve; GPUadmission stays24.25GiB
  if [[ ${W45R_LAUNCH_SMOKE:-0} == 1 ]]; then
    used=${W45R_SMOKE_FORECAST_GIB:?smoke needs W45R_SMOKE_FORECAST_GIB}
    need=$(( W45R_DATA_FLOOR_GIB + W45R_DATA_MARGIN_GIB + $1 ))
    w45_log "DISK-GUARD smoke: free ${free} GiB - smoke forecast ${used} GiB -> $( (( free - used >= W45R_DATA_FLOOR_GIB + W45R_DATA_MARGIN_GIB )) && echo PASS || echo REFUSE ); GPU mode would need >= ${need} GiB (production forecast $1 GiB), now ${free} -> would $( (( free >= need )) && echo PASS || echo REFUSE )"
  fi
  (( free - used >= W45R_DATA_FLOOR_GIB + W45R_DATA_MARGIN_GIB ))
}

w45_sha() { sha256sum "$1" | cut -d' ' -f1; }

W45R_DATA_SCRATCH=<DATA_ROOT>/wrf_gpu2/scratch/v45-rc
W45R_ROOT_CAP_GIB=11              # measured6.881GiB base incl paid seal +3.25GiB post reserve; manager01:06:22
