#!/usr/bin/env bash
# w44-rc v0.4.4 RW4 CPU part: `gpuwrf aot-key` pairs on the pinned source (copy of the v0.4.3 aotkey_cpu.sh recipe:
# private temp, GPUWRF_JAX_CACHE=0, never sources bundle.env, never writes a bundle), sequential, QUIET-aware.
# usage: aotkey_cpu.sh MANIFEST    (wraps itself in ONE RAM slot <= 12G, cores 24,25,28, nice 19)
# Expected: k0227_v_0502_h72, k0227_v_0614_h48, k0227_v_0227in48_h3, ki0227_v_ki0502_h3 all SHARED; k0227_v_ki0227_h3 (icloud
# flip) NOT all shared, reported beside them (aotkey_x).
set -uo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044/lib.sh
MAN=${1:?manifest}
eval "$("$W44R_PY" -I -c 'import json,sys,shlex; m=json.load(open(sys.argv[1])); print("SNAP="+shlex.quote(m["snapshot"]), "PIN="+m["pin"], "TREE="+m["src_tree"], "C="+shlex.quote(m["cases"]["dir"]))' "$MAN")"
AK=$W44R_LANE/receipts/aotkey_${PIN:0:12}; AKX=$W44R_LANE/receipts/aotkey_x_${PIN:0:12}; TMP=$W44R_SCRATCH/tmp_aotkey_${PIN:0:12}
if [[ -z ${W44R_AOTKEY_INNER:-} ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; then echo "WAIT: QUIET or GPU window"; exit 3; fi
  exec env W44R_AOTKEY_INNER=1 taskset -c 8,9,12 nice -n 19 bash "$W44R_RAM_SLOT" --label w44-rc-aotkey --max 12G --timeout 3600 -- bash "$0" "$MAN"
fi
test "$(git -C "$SNAP" rev-parse HEAD:src)" = "$TREE" || { echo "SNAP src tree != pin"; exit 2; }
test -z "$(git -C "$SNAP" status --porcelain --untracked-files=all -- src)" || { echo "SNAP src dirty"; exit 2; }
mkdir -p "$AK" "$AKX" "$TMP"
run() { # name case compare hours [outdir]
  local O=${5:-$AK}
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 20; done
  [ -s "$O/$1.json" ] && { echo "skip $1 (done)"; return; }
  echo "$(date -u +%FT%TZ) START $1"
  (cd "$SNAP" && env -u GPUWRF_JAX_CACHE_DIR -u JAX_COMPILATION_CACHE_DIR -u GPUWRF_XLA_AUTOTUNE_CACHE_DIR JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SNAP/src" TMPDIR="$TMP" \
    /usr/bin/time -v "$W44R_PY" -m gpuwrf.cli aot-key --input-dir "$C/$2" --compare-input-dir "$C/$3" --hours "$4" \
    --json "$O/$1.json") > "$O/$1.log" 2>&1
  echo "$(date -u +%FT%TZ) END $1 rc=$?"; grep -E "cheap_key=|SHARED|UNAVAILABLE|DIFFERS|Maximum resident" "$O/$1.log" | head -12
}
run k0227_v_0502_h72 wn3_0227 wn3_0502 72
run k0227_v_0614_h48 wn3_0227 wn3_0614 48
run k0227_v_0227in48_h3 wn3_0227 wn3_0227_48h 3
run ki0227_v_ki0502_h3 wn3_0227_icloud0 wn3_0502_icloud0 3
run k0227_v_ki0227_h3 wn3_0227 wn3_0227_icloud0 3 "$AKX"
rm -rf "${TMP:?}"
echo ALL_DONE
