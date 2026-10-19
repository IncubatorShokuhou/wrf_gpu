#!/usr/bin/env bash
# w42-bundle042 RW4 CPU part: `gpuwrf aot-key` pairs on the pinned tree P/src (private temp cache since w42-aotkey; never
# sources bundle.env, never writes P), sequential, QUIET-aware between pairs. Copy of the v0.4.1 aotkey_cpu.sh with the
# icloud=0 pairs added (when the icloud0 case dirs exist) and the v0.4.2 paths.
# usage: taskset -c <cores> nice -n 19 scripts/with_ram_slot.sh --label w42-bundle042-aotkey --max 12G -- bash aotkey_cpu.sh SRC
set -uo pipefail
SRC=$1
L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042; C=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/cases
source "$(cd "$(dirname "$0")" && pwd)/pin.env"
test "$(git -C "$SRC" rev-parse HEAD:src)" = "$TREE" || { echo "SRC src tree != pin"; exit 2; }
test -z "$(git -C "$SRC" status --porcelain --untracked-files=all -- src)" || { echo "SRC dirty"; exit 2; }
# outputs per pin (a re-pin must never reuse an older pin's JSON: run() skips pairs whose JSON exists)
AK=$L/aotkey_${PIN:0:12}; AKX=$L/aotkey_x_${PIN:0:12}
mkdir -p "$AK" <DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/tmp/aotkey
dir_of() { [[ $1 == /* ]] && echo "$1" || echo "$C/$1"; }   # case name under $C, or an absolute input dir
run() { # name case compare hours [outdir]
  local O=${5:-$AK}; mkdir -p "$O"
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 20; done
  [ -s "$O/$1.json" ] && { echo "skip $1 (done)"; return; }
  echo "$(date -u +%FT%TZ) START $1"
  env -u GPUWRF_JAX_CACHE_DIR -u JAX_COMPILATION_CACHE_DIR -u GPUWRF_XLA_AUTOTUNE_CACHE_DIR JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SRC/src" TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/tmp/aotkey \
    /usr/bin/time -v python3 -m gpuwrf.cli aot-key --input-dir "$(dir_of "$2")" --compare-input-dir "$(dir_of "$3")" --hours "$4" \
    --json "$O/$1.json" > "$O/$1.log" 2>&1
  echo "$(date -u +%FT%TZ) END $1 rc=$?"; grep -E "cheap_key=|SHARED|UNAVAILABLE|DIFFERS|Maximum resident" "$O/$1.log" | head -12
}
run k0227_v_0502_h72 wn3_0227 wn3_0502 72
run k0227_v_0614_h48 wn3_0227 wn3_0614 48
run k0227_v_0227in48_h3 wn3_0227 wn3_0227_48h 3
if [ "${BS_ICLOUD0:-0}" = 1 ]; then
  # A4 input (w42-icloud0 spec, namelist run_hours 24 / restart_interval / icloud 0) vs B8 input (0502, only icloud 1 -> 0):
  # expected SHARED = the icloud=0 program is date-blind and blind to run_hours/restart_interval
  run ki0227_v_ki0502_h3 "$ICLOUD0_INPUT" wn3_0502_icloud0 3
  # expected NOT all shared (icloud=0 is a different program): kept out of the RW4 SHARED set (aotkey_x/), reported beside it
  run k0227_v_ki0227_h3 wn3_0227 "$ICLOUD0_INPUT" 3 "$AKX"
fi
echo ALL_DONE
