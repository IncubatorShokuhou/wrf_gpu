#!/usr/bin/env bash
# w41-bundle041 CPU follow-up after the GPU session (cores 24,25 nice 19, one RAM slot <= 12G, manager 11:30:13Z).
# Order: E41 on P/jax (every rich JIT + AOT program incl. fused_d02, E201; before any cleanup, E166/E177) -> bundle check
# (E41 must not write P) -> RW3 72 h frozen-F20 D6 of B1 (v0.4.1) and B7 (v0.4.0, last-release fidelity reference) vs the
# original CPU-WRF -> wn3_integrity of both -> verdict.py. Waits while a QUIET flag exists before every heavy step.
# usage: taskset -c 24,25 nice -n 19 scripts/with_ram_slot.sh --label w41-bundle041 --max 12G -- bash followup.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1; S=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041; R=$S/runs
L=<USER_HOME>/wrf_gpu2_lanes/w41-bundle041; OUT=$L/followup; mkdir -p "$OUT"
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$P/src/src
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/followup.log"; }
quiet_wait() { while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }
check() { python3 "$P/tools/compile_bundle.py" check --cache "$P/jax" --manifest "$P/sealed/BUNDLE_MANIFEST.json" --out "$1.json" | tee "$1.txt"; }
log "START"
quiet_wait
if [ ! -e "$P/e41/wave_gate.json" ]; then
  log "E41 start"
  (cd "$P/src" && timeout 3600 python3 scripts/wave_cubin_gate.py "$P/jax" "$P/e41" \
     --run-proof "$R/A3_wn3_0227_72h/proof/nested_pipeline_run.json") > "$OUT/e41.log" 2>&1
  log "E41 rc=$? $(tail -1 "$OUT/e41.log" | cut -c1-200)"
fi
log "check after E41: $(check "$OUT/check_after_e41")"
for arm in B1_wn3_0227_72h B7_v040_wn3_0227_72h; do
  case=$OUT/d6_$arm
  [ -e "$case/SCORING_IO_d03.json" ] && continue
  test "$(cat "$R/$arm/rc.txt" 2>/dev/null)" = 0 || { log "D6 $arm: arm rc != 0, NOT_RUN"; continue; }
  quiet_wait; log "D6 $arm start"
  timeout 7200 python3 "$HERE/d6/d6_72h.py" "$P/src" "$case" "$R/$arm/wrfout" > "$OUT/d6_$arm.log" 2>&1
  log "D6 $arm rc=$? $(tail -1 "$OUT/d6_$arm.log" | cut -c1-200)"
  quiet_wait; log "integrity $arm start"
  timeout 3600 python3 "$P/src/scripts/wn3_integrity.py" --cpu-dir "$case/cpu" --gpu-dir "$R/$arm/wrfout" \
    --out "$OUT/integrity_$arm.json" > "$OUT/integrity_$arm.log" 2>&1
  log "integrity $arm rc=$? $(tail -1 "$OUT/integrity_$arm.log" | cut -c1-300)"
done
python3 "$HERE/verdict.py" "$S" "$P" "$OUT/RW_VERDICT.json" --d6 "$OUT/d6_B1_wn3_0227_72h" --d6old "$OUT/d6_B7_v040_wn3_0227_72h" \
  --int "$OUT/integrity_B1_wn3_0227_72h.json" --intold "$OUT/integrity_B7_v040_wn3_0227_72h.json" --e41 "$P/e41/wave_gate.json" \
  --aotkey "$L/aotkey" | tee -a "$OUT/followup.log"
log "check final: $(check "$OUT/check_final")"
log "FOLLOWUP DONE"
