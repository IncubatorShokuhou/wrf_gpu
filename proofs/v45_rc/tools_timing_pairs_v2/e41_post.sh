#!/usr/bin/env bash
# v45-rc v0.4.5: E41 cubin gate + bundle checker AFTER the GPU lock is released (still inside the entry's RAM slot and
# setsid group), BEFORE any cache/layer cleanup (E166/E177/E201/E220). CPU only.
# usage: e41_post.sh   (env from the entry: W45R_RUN W45R_ENVFILE W45R_MODE W45R_RECEIPTS; smoke: W45R_SMOKE_E41_CACHE
#                       W45R_SMOKE_E41_PROOF = an existing real GPU cache + run proof, read-only)
# Child env of every scan (the w44-warmstart E41 failure 2026-10-24 was a missing PYTHONPATH): JAX_PLATFORMS=cpu
# GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<snapshot>/src, cwd <snapshot>, scripts/wave_cubin_gate.py of the
# snapshot. Passes: (all) every rich JIT + AOT program incl. fused, default discovery; (per prime arm with a nested AOT
# inventory) the same scan REQUIRING that run's programs (--run-proof). Then compile_bundle check (E41 must not write P).
set -uo pipefail
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/lib.sh
: "${W45R_RUN:?}" "${W45R_ENVFILE:?}" "${W45R_MODE:?}" "${W45R_RECEIPTS:?}"
source "$W45R_ENVFILE"
W45R_LOG=$W45R_RUN/post.log
OUT=$W45R_RUN/e41; mkdir -p "$OUT"
CACHE=$W45R_P/jax
INTEGRITY_KIND=sealed
if [ ! -s "$W45R_P/sealed/BUNDLE_MANIFEST.json" ]; then
  [[ ${W45R_ENTRY_NAME:-} == prime_wn3 ]] || { echo 'UNSEALED integrity is only valid after the final cold prime'; exit 6; }
  INTEGRITY_KIND=unsealed
  env JAX_PLATFORMS=cpu "$W45R_PY" -I "$W45R_TOOLS/cache_integrity.py" snapshot "$W45R_P/jax" "$OUT/CACHE_BEFORE.json" > "$OUT/cache_before.log" 2>&1 || exit 6
fi
declare -a PROOFS=()
for pr in "$W45R_RUN"/runs/A*/proof/nested_pipeline_run.json; do [ -s "$pr" ] && PROOFS+=("$pr"); done
if [[ $W45R_MODE == smoke ]]; then
  # a CPU-primed cache holds no CUDA ELF: scan the smoke cache (import/discovery path; refusal expected and recorded),
  # then the real GPU cache named by the smoke launcher (read-only) with its own run proof = the full scan path
  SMOKE_CACHE=$CACHE; CACHE=${W45R_SMOKE_E41_CACHE:?}; PROOFS=("${W45R_SMOKE_E41_PROOF:?}")
fi
scan() {  # scan LABEL CACHE [--run-proof PROOF]
  local label=$1 cache=$2; shift 2
  if w45_quiet; then w45_log "E41 $label: QUIET flag set -> not started"; return 3; fi
  local t0; t0=$(date +%s.%N)
  (cd "$W45R_SNAP" && env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$W45R_SNAP/src" CUOBJDUMP=/usr/local/cuda/bin/cuobjdump \
     /usr/bin/time -v -o "$OUT/$label.time" "$W45R_PY" scripts/wave_cubin_gate.py "$cache" "$OUT/$label" "$@") \
     > "$OUT/$label.log" 2>&1
  local rc=$?
  w45_log "E41 $label rc=$rc $(python3 -c "import sys; print(round(float(sys.argv[2]) - float(sys.argv[1]), 1))" "$t0" "$(date +%s.%N)") s: $(tail -1 "$OUT/$label.log" | cut -c1-220)"
  return $rc
}
[[ $W45R_MODE == smoke ]] && { scan smoke_cpu_cache "$SMOKE_CACHE"; echo "rc=$?" > "$OUT/smoke_cpu_cache.rc"; }
scan all "$CACHE"; ALL_RC=$?
i=0; PRC=()
for pr in "${PROOFS[@]}"; do
  i=$((i + 1))
  env JAX_PLATFORMS=cpu PYTHONPATH="$W45R_SNAP/src" "$W45R_PY" "$W45R_TOOLS/e41_proof_coverage.py" \
    "$W45R_SNAP" "$CACHE" "$OUT/all/wave_gate.json" "$pr" "$OUT/proof$i" > "$OUT/proof$i.log" 2>&1
  PRC+=("$?")
  echo "$pr" > "$OUT/proof$i.which"
done
if [[ $INTEGRITY_KIND == sealed ]]; then
  CHK=$(JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 "$W45R_PY" -I "$W45R_P/tools/compile_bundle.py" check --cache "$W45R_P/jax" \
        --manifest "$W45R_P/sealed/BUNDLE_MANIFEST.json" --out "$OUT/check_after_e41.json" 2>&1); CHECK_RC=$?
  CHK=$(tail -1 <<< "$CHK")
else
  CHK=$(env JAX_PLATFORMS=cpu "$W45R_PY" -I "$W45R_TOOLS/cache_integrity.py" check "$W45R_P/jax" "$OUT/CACHE_BEFORE.json" "$OUT/CACHE_AFTER.json" 2>&1); CHECK_RC=$?
fi
w45_log "check after E41: $CHK"
"$W45R_PY" -I - "$OUT" "$ALL_RC" "${PRC[*]:-}" "$CHK" <<'PY' > "$OUT/E41_POST.json"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
gates = {}
for g in sorted(out.glob("*/wave_gate.json")):
    d = json.loads(g.read_text())
    gates[g.parent.name] = {k: d.get(k) for k in ("e41_gate", "hard_stack_gate", "conservative_flags", "n_entries")} | {
        "coverage": (d.get("coverage") or {}).get("status"), "aot_programs": (d.get("coverage") or {}).get("aot_programs")}
print(json.dumps(dict(all_rc=int(sys.argv[2]), proof_rcs=sys.argv[3].split(), gates=gates, check_after_e41=sys.argv[4],
                      note="E41 rc 2 = gate FAIL or INCOMPLETE (v0.4.4 own gate: INCOMPLETE, compact AOT HLO kInput UNKNOWN, E201)"),
                 indent=1))
PY
mkdir -p "$W45R_RECEIPTS/e41"
cp "$OUT/E41_POST.json" "$OUT"/*.log "$OUT"/*.time "$W45R_RECEIPTS/e41/" 2>/dev/null
for g in "$OUT"/*/wave_gate.json; do [ -s "$g" ] && cp "$g" "$W45R_RECEIPTS/e41/$(basename "$(dirname "$g")")_wave_gate.json"; done
(( CHECK_RC == 0 )) && grep -q -E '^(BUNDLE_OK|UNSEALED_CACHE_OK) missing=0 changed=0 new=0([ ]|$)' <<< "$CHK" || { w45_log "P byte integrity failed after E41 rc=$CHECK_RC"; exit 6; }
"$W45R_PY" -I - "$OUT/E41_POST.json" <<'E41VERDICT'
import json,sys
r=json.load(open(sys.argv[1]));g=r['gates']
assert g and 'all' in g, 'E41 all-program verdict missing'
assert all(int(rc)==0 for rc in r['proof_rcs']),r['proof_rcs']
for label,v in g.items():
    if label=='smoke_cpu_cache':continue
    assert v['hard_stack_gate']=='PASS', (label,v)
    # Full per-executable native/kInput policy is assessed below from the original JSON.
    if v.get('coverage') is not None:assert v['coverage'] in ('PASS','COMPLETE'),(label,v)
print('E41 JSON ACCEPTED; actual GPU scan still required for release')
E41VERDICT
VRC=$?
if (( VRC == 0 )); then
  env JAX_PLATFORMS=cpu "$W45R_PY" "$W45R_TOOLS/e41_policy.py" "$OUT/all/wave_gate.json" "$OUT/RC_POLICY.json" || VRC=$?
fi
if (( VRC == 0 )); then
  [[ $INTEGRITY_KIND == unsealed ]] && cp "$OUT/CACHE_BEFORE.json" "$W45R_P/meta/E41_CACHE_INTEGRITY.json"
  cp "$OUT/RC_POLICY.json" "$W45R_P/meta/E41_RC_POLICY.json"
  cp "$OUT/E41_POST.json" "$W45R_P/meta/E41_POST.json"
  cp "$OUT/all/wave_gate.json" "$W45R_P/meta/E41_wave_gate.json"
  printf '%s\n' "$W45R_MODE" > "$W45R_P/meta/E41_MODE"
fi
exit "$VRC"
