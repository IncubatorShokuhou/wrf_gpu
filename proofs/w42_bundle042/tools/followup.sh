#!/usr/bin/env bash
# w42-bundle042 CPU follow-up after the GPU session (lane cores, nice 19, ONE RAM slot <= 12G). Derived from the v0.4.1
# w41-bundle041 followup.sh. Order: E41 on P/jax (every rich JIT + AOT program incl. fused, E201; P is kept, E166/E177)
# -> bundle check (E41 must not write P) -> RW3 72 h D6 of N72 (rw3t: v0.4.2 72 h warm replay) vs the ORIGINAL CPU-WRF with the NEW scorer
# (P/src/scripts/wn3_score.py --hours 72: sequential domains, bounded files; output integrity inside) -> frozen-A3 floor
# reading -> RW3T accept + pre-registered verdict -> verdict.py (RW1-RW4 table). Waits while a QUIET flag exists before
# every heavy step.
# usage: taskset -c <cores> nice -n 19 scripts/with_ram_slot.sh --label w42-bundle042-fu --max 12G -- bash followup.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042; R=$S/gpu/runs; RT=$S/rw3t
L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042; OUT=$L/followup; mkdir -p "$OUT"
CPU72=$L/cpu72_receipt/STAGED_INDEX.json   # stage_cpu72.py $S/cpu72 $L/cpu72_receipt (guarded original CPU-WRF 72 h frames)
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
source "$HERE/pin.env"
if [ "${BS_ICLOUD0:-0}" = 1 ] && [ -s "$R/A4_wn3_0227_icloud0/proof/nested_pipeline_run.json" ] && [ ! -e "$OUT/e41_icloud0/wave_gate.json" ]; then
  # the scan above covers every cached program; this pass also REQUIRES the icloud=0 run's programs (one proof per call)
  log "E41 icloud0 start"
  (cd "$P/src" && timeout 3600 python3 scripts/wave_cubin_gate.py "$P/jax" "$OUT/e41_icloud0" \
     --run-proof "$R/A4_wn3_0227_icloud0/proof/nested_pipeline_run.json") > "$OUT/e41_icloud0.log" 2>&1
  log "E41 icloud0 rc=$? $(tail -1 "$OUT/e41_icloud0.log" | cut -c1-200)"
fi
log "check after E41: $(check "$OUT/check_after_e41")"
if [ ! -s "$OUT/d6_N72/summary.json" ] && [ "$(cat "$RT/runs/N72/rc.txt" 2>/dev/null)" = 0 ]; then
  quiet_wait
  CPUD=$OUT/cpu72_view; rm -rf "$CPUD"; mkdir -p "$CPUD"
  python3 - "$CPU72" "$CPUD" <<'PY'
import hashlib, json, sys
from pathlib import Path
index, view = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
assert not index["sha_mismatch_vs_f20"] and index["sealed24_checked"] == 75
n = 0
for dom, files in index["index"].items():
    for name, rec in files.items():
        (view / name).symlink_to(rec["staged"]); n += 1
print("cpu72 view", n, "links")
PY
  log "D6 N72 start (new scorer, 72 h, sequential)"
  (cd "$P/src" && /usr/bin/time -v -o "$OUT/d6_N72.time" timeout 10800 python3 scripts/wn3_score.py --cpu-dir "$CPUD" \
     --gpu-dir "$RT/runs/N72/wrfout" --out "$OUT/d6_N72" --hours 72) > "$OUT/d6_N72.log" 2>&1
  log "D6 N72 rc=$? $(tail -1 "$OUT/d6_N72.log" | cut -c1-300)"
fi
[ -s "$OUT/d6_N72/summary.json" ] && python3 "$HERE/d6/floor72_new.py" "$OUT/d6_N72" "$OUT/FLOOR72_N72.json" > "$OUT/floor72.log" 2>&1
if [ "${BS_ICLOUD0:-0}" = 1 ] && [ "$(cat "$R/A4_wn3_0227_icloud0/rc.txt" 2>/dev/null)" = 0 ] && [ ! -s "$OUT/icloud0_score/D6_SUMMARY.json" ]; then
  # icloud=0 final-candidate D6 vs w42-icloud0's original CPU-WRF icloud=0 twin (manager 06:05:46Z), recipe of
  # BUNDLE_ARM_SPEC.md: v0.4.0 final_validation score.py per domain, then icloud0_report.py + d6_summary.py
  SC=$OUT/icloud0_score; mkdir -p "$SC"
  for l in input cpu; do [ -e "$SC/$l" ] || ln -s "$ICLOUD0_CASE/$l" "$SC/$l"; done
  [ -e "$SC/gpu" ] || ln -s "$R/A4_wn3_0227_icloud0/wrfout" "$SC/gpu"
  cp "$ICLOUD0_CASE/CPU_OUTPUTS.json" "$ICLOUD0_CASE/cpu_run.json" "$SC/"
  for d in d01 d02 d03; do
    quiet_wait
    log "D6 icloud0 $d start"
    (cd "$P/src" && /usr/bin/time -v -o "$OUT/icloud0_score_$d.time" timeout 5400 python3 proofs/v040/final_validation/score.py \
       "$P/src" "$SC" --domains "$d") > "$OUT/icloud0_score_$d.log" 2>&1
    log "D6 icloud0 $d rc=$? $(tail -1 "$OUT/icloud0_score_$d.log" | cut -c1-200)"
  done
  python3 "$ICLOUD0_TOOLS/icloud0_report.py" "$SC" "$SC/ICLOUD0_report.json" > "$OUT/icloud0_report.log" 2>&1
  python3 "$ICLOUD0_TOOLS/d6_summary.py" "$SC" "$SC/D6_SUMMARY.json" > "$OUT/icloud0_summary.log" 2>&1
  log "icloud0 report/summary rc=$? $(tail -1 "$OUT/icloud0_summary.log" | cut -c1-200)"
fi
log "floor: $(head -1 "$OUT/floor72.log" 2>/dev/null)"
if [ -d "$RT/runs" ]; then   # RW3T: pre-registered acceptance + verdict (PREREG.json in this frozen tools dir)
  python3 "$HERE/disk_watch.py" summarize "$RT/disk.jsonl" "$RT" "$OUT/RW3T_DISK.json" > "$OUT/rw3t_disk.log" 2>&1
  python3 "$HERE/accept.py" "$RT" "$OUT/RW3T_ACCEPT.json" "$OUT/RW3T_DISK.json" > "$OUT/rw3t_accept.log" 2>&1
  log "RW3T accept rc=$? $(tail -1 "$OUT/rw3t_accept.log" | cut -c1-200)"
  python3 "$HERE/attrib_detail.py" "$RT" "$OUT/RW3T_DETAIL24.json" N1 O1 N2 O2 N3 O3 > "$OUT/rw3t_detail24.log" 2>&1
  python3 "$HERE/attrib_detail.py" "$RT" "$OUT/RW3T_DETAIL72.json" N72 O72 > "$OUT/rw3t_detail72.log" 2>&1
  python3 "$HERE/rw3t_bytes_check.py" "$RT/runs" "$R/A3_wn3_0227_72h/wrfout_sha.txt" \
    "$L/ref041/w41-bundle041/runs/B1_wn3_0227_72h/wrfout_sha.txt" "$OUT/RW3T_BYTES.json" > "$OUT/rw3t_bytes.log" 2>&1
  python3 "$HERE/rw3t_verdict.py" "$OUT/RW3T_ACCEPT.json" "$OUT/RW3T_DETAIL24.json" "$OUT/RW3T_DETAIL72.json" \
    "$OUT/RW3T_BYTES.json" "$OUT/RW3T_VERDICT.json" "$HERE/PREREG.json" "$OUT/RW3T_DISK.json" > "$OUT/rw3t_verdict.log" 2>&1
  log "RW3T verdict rc=$? $(tail -1 "$OUT/rw3t_verdict.log" | cut -c1-200)"
fi
python3 "$HERE/verdict.py" "$S/gpu" "$P" "$OUT/RW_VERDICT.json" --rw3t-root "$RT" --rw3t-verdict "$OUT/RW3T_VERDICT.json" \
  --d6 "$OUT/d6_N72" --floor "$OUT/FLOOR72_N72.json" --e41 "$P/e41/wave_gate.json" --aotkey "$L/aotkey_${PIN:0:12}" --disk "$OUT/RW3T_DISK.json" \
  --icloud0-score "$OUT/icloud0_score" --icloud0-e41 "$OUT/e41_icloud0/wave_gate.json" --icloud0-tools "${ICLOUD0_TOOLS:-/nonexistent}" \
  --ref041 <USER_HOME>/src/wrf_gpu2_wt/w42-bundle042/proofs/w41_bundle041/receipts/RW_VERDICT.json | tee -a "$OUT/followup.log"
log "check final: $(check "$OUT/check_final")"
log "FOLLOWUP DONE"
