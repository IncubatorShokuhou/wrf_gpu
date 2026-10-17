#!/usr/bin/env bash
# 2x RTX 4090 test matrix for the wrf_gpu multi-GPU runtime (lane o2-multigpu). Run AFTER setup.sh:
#   source <work>/env.sh ; bash scripts/runpod/test_matrix.sh <results-dir> [T1 T2 T3a ...]
# Each test writes its own directory + receipt; failures do not stop later tests. Expected numbers: RUNBOOK.md.
# Needs only the repo (bundled Swiss case). T6 additionally needs PROD inputs in $PROD_CASE (see RUNBOOK.md).
set -uo pipefail
repo=${WRFGPU_REPO:?source env.sh first}
res=${1:?results dir}; shift
tests=${*:-T1 T2 T3a T3b T3c T4 T5}
mkdir -p "$res"
if [ -e "$res/status.tsv" ]; then echo "results directory already used: $res" >&2; exit 2; fi
touch "$res/status.tsv"
swiss=$repo/examples/switzerland_d01
bench="python3 $repo/scripts/runpod/decomposed_bench.py"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$res/matrix.log"; }
run() { local name=$1; shift; log "START $name: $*"; ( "$@" ) > "$res/$name.out" 2>&1; local rc=$?; printf '%s\t%s\n' "$name" "$rc" >> "$res/status.tsv"; log "END $name rc=$rc"; tail -2 "$res/$name.out" | tee -a "$res/matrix.log"; return 0; }

for t in $tests; do case $t in
T1)  # single-GPU baseline through the product CLI (cold compile, then warm) — the reference for every speedup
  run T1_cold python3 -m gpuwrf.cli run --input-dir "$swiss" --output-dir "$res/T1_cold" --domain d01 --hours 6
  run T1_warm python3 -m gpuwrf.cli run --input-dir "$swiss" --output-dir "$res/T1_warm" --domain d01 --hours 6 ;;
T2)  # ensemble: 4 identical members over 2 GPUs (process per GPU); identical inputs -> compare outputs across GPUs
  mkdir -p "$res/T2_members"
  for m in 01 02 03 04; do [ -d "$res/T2_members/member$m" ] || cp -r "$swiss" "$res/T2_members/member$m"; done
  run T2_ensemble "$repo/scripts/run_parallel_cases.sh" --gpus all --out-root "$res/T2" --gpu-log-interval 2 \
      "$res"/T2_members/member0{1,2,3,4} -- --domain d01 --hours 6
  run T2_identity python3 "$repo/scripts/runpod/validate_ensemble.py" "$res/T2" --expected 4 --min-frames 6
  ;;
T3a) # correctness: Swiss mirror 4x2 = 168x84, 2x1 split vs single GPU, 6 steps incl. radiation step 1
  run T3a     $bench --out "$res/T3a" --mirror 4x2 --px 2 --py 1 --steps 6 --start-step 1
  ;;
T3b) # scaling: Swiss mirror 10x10 = 420x420 (the >=400x400 synthetic-from-real domain), both transports, k=1/2
  run T3b_dp_k1 $bench --out "$res/T3b" --mirror 10x10 --px 2 --py 1 --steps 20 --transport device_put
  run T3b_pp_k1 $bench --out "$res/T3b" --mirror 10x10 --px 2 --py 1 --steps 20 --transport ppermute
  run T3b_dp_k2 $bench --out "$res/T3b" --mirror 10x10 --px 2 --py 1 --steps 20 --exchange-every 2
  run T3b_all  $bench --out "$res/T3b" --mirror 10x10 --px 2 --py 1 --steps 20 --exchange all
  ;;
T3c) # 1x2 split (y) of the same domain: message along x instead of y
  run T3c $bench --out "$res/T3c" --mirror 10x10 --px 1 --py 2 --steps 20 ;;
T4)  # big: 840x840 does not fit one 24 GB GPU with temps [I] -> decomposed only (memory-capacity use case)
  run T4 $bench --out "$res/T4" --mirror 20x20 --px 2 --py 1 --steps 10 --host-global ;;
T5)  # nsys of the 420x420 decomposed run (6 steps); open with nsys-ui or `nsys stats`
  if command -v nsys >/dev/null; then
    run T5 nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --force-overwrite=true -o "$res/T5_nsys" \
        $bench --out "$res/T5" --mirror 10x10 --px 2 --py 1 --steps 6 --no-single
  else log "T5 unavailable: nsys not installed"; run T5_missing_nsys false; fi ;;
T6)  # PROD d01 120x70 (needs PROD_CASE=dir with wrfinput_d01, wrfbdy_d01, namelist.input) and d01 mirror 4x6=480x420
  if [ -z "${PROD_CASE:-}" ]; then run T6_missing_input false; else
    run T6a $bench --input "$PROD_CASE" --out "$res/T6a" --px 2 --py 1 --steps 6 --start-step 1
    run T6b $bench --input "$PROD_CASE" --out "$res/T6b" --mirror 4x6 --px 2 --py 1 --steps 20
  fi ;;
*) run "unknown_$t" false ;;
esac; done
run cost_model python3 "$repo/scripts/runpod/cost_model.py" --out "$res/cost_model.json" --markdown "$res/cost_model.md"
log "matrix done: $res (send the whole directory back; RUNBOOK.md lists what to check)"
python3 - "$res" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = [line.split('\t') for line in (root / 'status.tsv').read_text().splitlines()]
failures = [name for name, rc in rows if int(rc) != 0]
report = {'verdict': 'FAIL' if failures or not rows else 'PASS', 'failed_arms': failures,
          'arms': [{'name': name, 'rc': int(rc)} for name, rc in rows]}
(root / 'matrix_status.json').write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps(report))
sys.exit(1 if report['verdict'] == 'FAIL' else 0)
PY
