#!/usr/bin/env bash
# w43-release C0b: `gpuwrf aot-key --json` on the v0.4.2 bundle source (P42/src, read-only, private temp cache) and on the
# v0.4.3 pin snapshot for the same case pairs -> per-program cheap-key COMPONENTS compared by aotkey_compare.py.
# usage: with_ram_slot.sh --max 12G -- taskset -c C nice -n 19 bash aotkey_pair.sh SNAP_V043 OUT [TAG:TREE ...]
#   extra TAG:TREE pairs replace the default v042:P42/src v043:SNAP list (e.g. v043pre:<main 218e27ea28 snapshot>)
set -uo pipefail
SNAP=$1 OUT=$2; P42=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2/src; C=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases
mkdir -p "$OUT"
run() { # tag src case compare hours
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 20; done
  [ -s "$OUT/$1.json" ] && { echo "skip $1"; return; }
  echo "$(date -u +%FT%TZ) START $1"
  env -u GPUWRF_JAX_CACHE_DIR -u JAX_COMPILATION_CACHE_DIR -u GPUWRF_XLA_AUTOTUNE_CACHE_DIR JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$2/src" TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/tmp/aotkey \
    /usr/bin/time -v python3 -m gpuwrf.cli aot-key --input-dir "$C/$3" --compare-input-dir "$C/$4" --hours "$5" \
    --json "$OUT/$1.json" > "$OUT/$1.log" 2>&1
  echo "$(date -u +%FT%TZ) END $1 rc=$?"
}
LIST=("v042:$P42" "v043:$SNAP"); (( $# > 2 )) && LIST=("${@:3}")
for v in "${LIST[@]}"; do
  run "${v%%:*}_0227_v_0502_h72" "${v#*:}" wn3_0227 wn3_0502 72
  run "${v%%:*}_i0227_v_i0502_h3" "${v#*:}" wn3_0227_icloud0 wn3_0502_icloud0 3
done
echo ALL_DONE
