#!/usr/bin/env bash
# wz1080_bootstrap.sh — LOCAL orchestrator for the one-time setup of the wz1080 test box (idempotent steps).
# Usage: scripts/remote/wz1080_bootstrap.sh <step>...   (logs: $WZ_LOCAL_LANE/setup/<step>.log)
#   ssh        connectivity + who/where
#   tools      push scripts/remote to ~/src/wrf_gpu2_env/tools, copy the local uv binary if the box has none
#   inventory | python | jax-cpu | jax-cuda | activate | cpuwrf-env   -> remote wz1080_setup.sh phases
#   data       pristine WRF (~1.7 GB), git-ignored test fixtures overlay (~180 MB), Swiss CPU-WRF 24 h wrfouts (~140 MB)
#   repo       first/incremental sync of local main to the box's main clone (wz1080_sync.sh main --main)
#   all        ssh tools inventory python jax-cpu activate data repo   (jax-cuda after inventory confirms the driver)
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=wz1080_common.sh
source "$HERE/wz1080_common.sh"
LOG="$WZ_LOCAL_LANE/setup"; mkdir -p "$LOG"

SWISS_REF="${WZ_SWISS_REF:-<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu}"
PRISTINE="${WZ_PRISTINE:-<USER_HOME>/src/wrf_pristine}"

step_ssh() { wz_ssh 'echo "$(whoami)@$(hostname) $(uname -sr) home=$HOME shell=$SHELL"; df -h "$HOME" | tail -1'; }

step_tools() {
  wz_ssh "mkdir -p src/wrf_gpu2_env/tools .local/bin"
  wz_rsync -a --delete "$HERE/" "$WZ_HOST:$WZ_R_ENV/tools/"
  if ! wz_ssh 'test -x .local/bin/uv'; then
    wz_rsync -a "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" "$WZ_HOST:.local/bin/"
  fi
  wz_ssh '.local/bin/uv --version'
}

remote_phase() { wz_ssh "bash $WZ_R_ENV/tools/wz1080_setup.sh $1"; }

step_data() {
  wz_wait_quiet
  # Pristine WRF: oracles + run/ tables + CLWRF gas lookups (AGENTS: data/wrf_pristine must exist in every tree).
  wz_local_heavy rsync -e "ssh ${WZ_SSH_OPTS[*]}" -a --info=stats1 "$PRISTINE/" "$WZ_HOST:src/wrf_pristine/"
  # Git-ignored fixtures that tests read (never caches/scratch): relative paths from the local main checkout.
  local list="$LOG/overlay_files.txt"
  git -C "$WZ_LOCAL_REPO" status --ignored --porcelain data tests examples 2>/dev/null \
    | sed -n 's/^!! //p' | grep -v -E '__pycache__|^data/scratch|^data/wrf_pristine|\.pyc$|/\.pytest_cache' > "$list"
  echo "overlay entries: $(wc -l < "$list")"
  wz_local_heavy rsync -e "ssh ${WZ_SSH_OPTS[*]}" -a -r --info=stats1 --files-from="$list" \
    "$WZ_LOCAL_REPO/" "$WZ_HOST:$WZ_R_DATA/overlay/"
  # Swiss 24 h original CPU-WRF reference (VALIDATION_REFERENCES.md): history + namelist + rsl only.
  wz_local_heavy rsync -e "ssh ${WZ_SSH_OPTS[*]}" -a --info=stats1 --include='wrfout_d01_*' --include='namelist.*' \
    --include='rsl.*' --include='wrfinput_d01' --include='wrfbdy_d01' --exclude='*' \
    "$SWISS_REF/" "$WZ_HOST:$WZ_R_DATA/refs/swiss_rd11_cpu/"
  [[ -f "${SWISS_REF%/cpu}/cpu_header.json" ]] && wz_rsync -a "${SWISS_REF%/cpu}/cpu_header.json" "$WZ_HOST:$WZ_R_DATA/refs/swiss_rd11_cpu/"
  # CPU-WRF arm: the reference's own binary + run tables (+ precomputed Thompson tables so no one-off 1-2 min table build).
  local wb="${WZ_CPUWRF_BUILD:-<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar}"
  wz_ssh "mkdir -p $WZ_R_DATA/cpuwrf/run"
  wz_local_heavy rsync -e "ssh ${WZ_SSH_OPTS[*]}" -aL --info=stats1 --exclude='*.exe' --exclude=wrf --exclude=real \
    --exclude=tc --exclude=ndown --exclude='namelist.input*' "$wb/run/" "$WZ_HOST:$WZ_R_DATA/cpuwrf/run/"
  wz_local_heavy rsync -e "ssh ${WZ_SSH_OPTS[*]}" -a "$WZ_LOCAL_LANE/cpuwrf/thompson_tables/" "$WZ_HOST:$WZ_R_DATA/cpuwrf/run/"
  wz_rsync -aL "$wb/bin/wrf" "$WZ_HOST:$WZ_R_DATA/cpuwrf/wrf.exe"
  wz_ssh "du -sh src/wrf_pristine $WZ_R_DATA/cpuwrf $WZ_R_DATA/overlay $WZ_R_DATA/refs/*; df -h \$HOME | tail -1"
}

step_repo() { "$HERE/wz1080_sync.sh" main --main; }

[[ $# -gt 0 ]] || { sed -n '2,12p' "$0"; exit 2; }
steps=("$@"); [[ "${steps[0]}" == all ]] && steps=(ssh tools inventory python jax-cpu activate data repo)
for s in "${steps[@]}"; do
  echo "######## $s ($(date -u +%H:%M:%SZ))"
  set +e
  ( set -e
  case "$s" in
    ssh) step_ssh;; tools) step_tools;; data) step_data;; repo) step_repo;;
    inventory|python|jax-cpu|jax-cuda|activate|cpuwrf-env) remote_phase "$s";;
    *) echo "unknown step $s" >&2; false;;
  esac ) 2>&1 | tee "$LOG/$s.log"
  rc="${PIPESTATUS[0]}"
  set -e
  (( rc == 0 )) || { echo "step $s FAILED rc=$rc (log $LOG/$s.log)" >&2; exit "$rc"; }
done
