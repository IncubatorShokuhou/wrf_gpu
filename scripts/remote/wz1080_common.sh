# shellcheck shell=bash
# wz1080_common.sh — shared config for the wz1080 test-box helpers (sourced, not executed).
# The box: Owner's second machine (Fedora, GTX 1080 Ti sm_61). Debug/repro/validation-testing only,
# never final gates or benchmarks. Transport = the ssh key only; never copy credentials.

WZ_HOST="${WZ_HOST:-wz1080}"
WZ_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
WZ_LOCAL_REPO="${WZ_LOCAL_REPO:-<USER_HOME>/src/wrf_gpu2}"
WZ_LOCAL_LANE="${WZ_LOCAL_LANE:-<USER_HOME>/wrf_gpu2_lanes/wz1080}"
# Local CPU work (bundle packing, rsync) runs on the lane cores of AGENTS.md section 4, niced.
WZ_LOCAL_CPUS="${WZ_LOCAL_CPUS:-8,9,12,13,24,25,28,29}"

# Remote layout (paths relative to the remote $HOME; ssh commands start there).
WZ_R_REPO="src/wrf_gpu2"          # non-bare clone, always detached; refs/synced/<sha12> = every shipped commit
WZ_R_WT="src/wrf_gpu2_wt"         # immutable detached snapshots: <sha12>/
WZ_R_ENV="src/wrf_gpu2_env"       # venv/, activate.sh, tools/ (copy of scripts/remote)
WZ_R_DATA="src/wrf_gpu2_data"     # overlay/ = git-ignored test fixtures, refs/ = CPU-WRF references
WZ_R_SYNC="src/wrf_gpu2_sync"     # transient bundles
WZ_R_RUNS="wrf_gpu2_runs"         # per-label logs/artifacts of wz1080_run.sh

wz_ssh() { ssh "${WZ_SSH_OPTS[@]}" "$WZ_HOST" "$@"; }
wz_rsync() { rsync -e "ssh ${WZ_SSH_OPTS[*]}" "$@"; }

# QUIET (AGENTS.md section 4): wait while the manager's S1 quiet flag exists before local CPU/disk-heavy work.
wz_wait_quiet() {
  local waited=0 max="${1:-7200}"
  while [ -e /tmp/wrf_gpu2_quiet ]; do
    if (( waited == 0 )); then echo "[wz1080] QUIET flag present; waiting (max ${max}s)" >&2; fi
    if (( waited >= max )); then echo "[wz1080] QUIET still present after ${max}s; abort" >&2; return 3; fi
    sleep 30; waited=$(( waited + 30 ))
  done
}

wz_local_heavy() { taskset -c "$WZ_LOCAL_CPUS" nice -n 19 ionice -c3 "$@"; }

# Confined recursive delete (AGENTS.md: never rm -rf a bare variable path). Removes PATH only if ROOT is canonical and at
# least 3 levels deep, PATH is canonical (no symlink, no ..), a DIRECT child of ROOT, and its basename matches the ERE NAME_RE.
# Anything else is refused with rc 3 and left in place; a missing PATH is a no-op.
wz_rm_confined() {  # wz_rm_confined ROOT PATH NAME_RE
  local root="${1:-}" path="${2:-}" name_re="${3:-}" rroot rpath
  if [[ -z "$root" || -z "$path" || -z "$name_re" ]]; then echo "[wz1080] refuse delete: empty argument" >&2; return 3; fi
  [[ -e "$path" || -L "$path" ]] || return 0
  rroot="$(realpath -e -- "$root" 2>/dev/null)" || { echo "[wz1080] refuse delete: root '$root' missing" >&2; return 3; }
  if [[ "$rroot" != "$root" || "$rroot" != /?*/?*/?* ]]; then
    echo "[wz1080] refuse delete: root '$root' not canonical or too shallow" >&2; return 3
  fi
  rpath="$(realpath -e -- "$path" 2>/dev/null)" || { echo "[wz1080] refuse delete: '$path' unresolvable" >&2; return 3; }
  if [[ "$rpath" != "$path" || "$(dirname -- "$rpath")" != "$rroot" ]] || ! [[ "$(basename -- "$rpath")" =~ $name_re ]]; then
    echo "[wz1080] refuse delete: '$path' is not a ${name_re} child of $rroot" >&2; return 3
  fi
  rm -rf -- "$rpath"
}
