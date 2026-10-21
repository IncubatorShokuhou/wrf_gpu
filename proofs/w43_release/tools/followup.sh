#!/usr/bin/env bash
# w43-release CPU follow-up after the GPU session (lane cores, nice 19, ONE RAM slot <= 12G). Derived from the v0.4.2
# w42-bundle042 followup.sh. Order: E41 on P/jax (every rich JIT + AOT program incl. fused, E201; P is kept, E166/E177)
# -> bundle check (E41 must not write P) -> RELEASE_PLAN.md section 1: C1/C2 native identity (native_ids.py on P43 and P42,
# native_compare.py), C2b autotune entry count, C3/C4 wrfout bytes vs the v0.4.2 bundle receipts -> RW3T accept + verdict
# (timing) -> verdict.py (RW1-RW4 table) -> identity_verdict.py (carry-forward decision). The RW3 72 h D6 vs the original
# CPU-WRF is NOT run here: it carries forward from v0.4.2 when C3 PASS; otherwise identity_verdict says D6_REQUIRED and the
# D6 runs as a separate step (CPU 72 h reference staging). Waits while a QUIET flag exists before every heavy step.
# usage: scripts/with_ram_slot.sh --label w43-release-fu --max 12G -- taskset -c <cores> nice -n 19 bash followup.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3; P42=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2; P43=$P  # fu2: P43 was unset -> set -u abort in the native_ids loop (15:15Z)
S=<DATA_ROOT>/wrf_gpu2/scratch/w43-release; R=$S/gpu/runs; RT=$S/rw3t
L=<USER_HOME>/wrf_gpu2_lanes/w43-release; OUT=$L/followup; mkdir -p "$OUT"
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$P/src/src
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/followup.log"; }
quiet_wait() { while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }
check() { python3 "$P/tools/compile_bundle.py" check --cache "$P/jax" --manifest "$P/sealed/BUNDLE_MANIFEST.json" --out "$1.json" | tee "$1.txt"; }
source "$HERE/pin.env"
log "START pin $PIN"
quiet_wait
if [ ! -e "$P/e41/wave_gate.json" ]; then
  log "E41 start"
  (cd "$P/src" && timeout 3600 python3 scripts/wave_cubin_gate.py "$P/jax" "$P/e41" \
     --run-proof "$R/A3_wn3_0227_72h/proof/nested_pipeline_run.json") > "$OUT/e41.log" 2>&1
  log "E41 rc=$? $(tail -1 "$OUT/e41.log" | cut -c1-200)"
fi
if [ "${BS_ICLOUD0:-0}" = 1 ] && [ -s "$R/A4_wn3_0227_icloud0/proof/nested_pipeline_run.json" ] && [ ! -e "$OUT/e41_icloud0/wave_gate.json" ]; then
  log "E41 icloud0 start"   # the scan above covers every cached program; this pass also REQUIRES the icloud=0 run's programs
  (cd "$P/src" && timeout 3600 python3 scripts/wave_cubin_gate.py "$P/jax" "$OUT/e41_icloud0" \
     --run-proof "$R/A4_wn3_0227_icloud0/proof/nested_pipeline_run.json") > "$OUT/e41_icloud0.log" 2>&1
  log "E41 icloud0 rc=$? $(tail -1 "$OUT/e41_icloud0.log" | cut -c1-200)"
fi
log "check after E41: $(check "$OUT/check_after_e41")"
# C1/C2: native identity per program (both bundles read-only; P42 rescanned with the same tool version)
AOT43=$(ls -d "$P"/jax/aot/0.4.3-* | head -1); AOT42=$(ls -d "$P42"/jax/aot/0.4.2-* | head -1)
for v in 42 43; do
  quiet_wait
  eval "aot=\$AOT$v; root=\$P$v"; [ "$v" = 43 ] && root=$P
  [ -s "$OUT/native_ids_v0$v.json" ] || { log "native_ids v0.4.$v start ($aot)"
    timeout 3600 python3 "$HERE/native_ids.py" "$P/src" "$aot" "$OUT/native_ids_v0$v.json" "$root" > "$OUT/native_ids_v0$v.log" 2>&1
    log "native_ids v0.4.$v rc=$?"; }
done
python3 "$HERE/native_compare.py" "$OUT/native_ids_v042.json" "$OUT/native_ids_v043.json" "$OUT/NATIVE_IDENTITY.json" 2>&1 | tee -a "$OUT/followup.log"
# C2b: autotune entries after priming vs the v0.4.2 seed
(cd "$P/jax/xla_gpu_per_fusion_autotune_cache_dir" && find . -type f | sort | xargs sha256sum) > "$OUT/autotune_after.sha256"
python3 - "$HERE/autotune_seed_v042.sha256" "$OUT/autotune_after.sha256" "$OUT/AUTOTUNE_SEED_CHECK.json" <<'PY' | tee -a "$OUT/followup.log"
import json, sys
seed = dict(l.split()[::-1] for l in open(sys.argv[1]) if l.strip()); after = dict(l.split()[::-1] for l in open(sys.argv[2]) if l.strip())
res = dict(seed_entries=len(seed), after_entries=len(after), new=sorted(set(after) - set(seed)),
           changed=sorted(k for k in seed if k in after and after[k] != seed[k]), missing=sorted(set(seed) - set(after)))
res["verdict"] = "PASS" if not (res["new"] or res["changed"] or res["missing"]) else "DIFF"
json.dump(res, open(sys.argv[3], "w"), indent=1); print("AUTOTUNE_SEED", res["verdict"], len(seed), len(after), len(res["new"]))
PY
# C3/C4: bytes vs v0.4.2 receipts (+ RW1 within v0.4.3, + O72 determinism control)
python3 "$HERE/bytes_vs_v042.py" "$R" "$P42/receipts" "$OUT/BYTES_VS_V042.json" "$RT/runs" 2>&1 | tee -a "$OUT/followup.log"
if [ -d "$RT/runs" ]; then   # RW3T: pre-registered acceptance + verdict (PREREG.json in this frozen tools dir)
  python3 "$HERE/disk_watch.py" summarize "$RT/disk.jsonl" "$RT" "$OUT/RW3T_DISK.json" > "$OUT/rw3t_disk.log" 2>&1
  python3 "$HERE/accept.py" "$RT" "$OUT/RW3T_ACCEPT.json" "$OUT/RW3T_DISK.json" > "$OUT/rw3t_accept.log" 2>&1
  log "RW3T accept rc=$? $(tail -1 "$OUT/rw3t_accept.log" | cut -c1-200)"
  python3 "$HERE/attrib_detail.py" "$RT" "$OUT/RW3T_DETAIL24.json" N1 O1 N2 O2 N3 O3 > "$OUT/rw3t_detail24.log" 2>&1
  python3 "$HERE/attrib_detail.py" "$RT" "$OUT/RW3T_DETAIL72.json" N72 O72 > "$OUT/rw3t_detail72.log" 2>&1
  python3 "$HERE/rw3t_bytes_check.py" "$RT/runs" "$R/A3_wn3_0227_72h/wrfout_sha.txt" \
    "$P42/receipts/A3_wn3_0227_72h/wrfout_sha.txt" "$OUT/RW3T_BYTES.json" > "$OUT/rw3t_bytes.log" 2>&1
  python3 "$HERE/rw3t_verdict.py" "$OUT/RW3T_ACCEPT.json" "$OUT/RW3T_DETAIL24.json" "$OUT/RW3T_DETAIL72.json" \
    "$OUT/RW3T_BYTES.json" "$OUT/RW3T_VERDICT.json" "$HERE/PREREG.json" "$OUT/RW3T_DISK.json" > "$OUT/rw3t_verdict.log" 2>&1
  log "RW3T verdict rc=$? $(tail -1 "$OUT/rw3t_verdict.log" | cut -c1-200)"
fi
python3 "$HERE/verdict.py" "$S/gpu" "$P" "$OUT/RW_VERDICT.json" --rw3t-root "$RT" --rw3t-verdict "$OUT/RW3T_VERDICT.json" \
  --e41 "$P/e41/wave_gate.json" --aotkey "$L/aotkey_${PIN:0:12}" --disk "$OUT/RW3T_DISK.json" \
  --ref042 <DATA_ROOT>/wrf_gpu2/worktrees/w43-release/proofs/w42_bundle042/receipts/bundle_b979188128e0/RW_VERDICT.json | tee -a "$OUT/followup.log"
python3 "$HERE/identity_verdict.py" "$OUT" "$L/aotkey/COMPONENTS.json" "$L/pin_${PIN:0:10}" "$OUT/IDENTITY_VERDICT.json" 2>&1 | tee -a "$OUT/followup.log"
log "check final: $(check "$OUT/check_final")"
log "FOLLOWUP DONE"
