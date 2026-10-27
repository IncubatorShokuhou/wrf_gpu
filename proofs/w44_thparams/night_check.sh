#!/usr/bin/env bash
# CPU-only E201 inventory and E41 after all four preparations released the GPU.
# v5: own user systemd scopes (no shared RAM-slot FIFO, E233); the E41 verdict is parsed from
# wave_gate.json by e41_verdict.py and RECORDED (E230: tool rc 2 = FAIL or INCOMPLETE) -- it never
# aborts the later warm measurement; the lane kill switch acts on the recorded verdict.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
smoke=${THPARAMS_LAUNCH_SMOKE:-0}
[[ $smoke == 0 || $smoke == 1 ]]
# Run dirs/caches on the <DATA_ROOT> lane scratch (root fs is at 97 %).
RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/gpu_v5
expect=gpu
# Smoke: the SAME inventory/E41 commands on the CPU launch-smoke receipts/cache (E220).
if [[ $smoke == 1 ]]; then RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/launch_smoke_v5; expect=cpu; fi
# CUDA mini (manager 21:48Z, 5-column Thompson MODEL SWAP, not a grid crop): blobs are named by domain
# in their own cache, so the same inventory and E41 scan run on REAL cubins. smoke=1 + mini=1 = CPU probe.
mini=${THPARAMS_CUDA_MINI:-0}
[[ $mini == 0 || $mini == 1 ]]
if [[ $mini == 1 ]]; then RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/cuda_mini_v5; fi
if [[ $mini == 1 && $smoke == 1 ]]; then RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/cuda_mini_probe_v5; fi
[[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$HERE:$TREE/src"
inv_cap=60; if [[ $mini == 1 ]]; then inv_cap=20; fi  # receipts + blob hashes only
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 8,9,12,24,25,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 -- timeout $inv_cap python - "$RUN" "$TREE" "$expect" "$mini" <<'PY'
import hashlib, json, pathlib, subprocess, sys
from prepared_inventory import artifact_bindings
run,tree=map(pathlib.Path,sys.argv[1:3]);expect=sys.argv[3];mini=sys.argv[4]=='1';pin=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()
selected={}
for path in sorted(run.glob('*-d??-*/receipt.json')):
    row=json.loads(path.read_text())
    if row['pin']!=pin or row['status']!='PASS': continue
    # Smoke/mini receipts execute the 5-column mini model; product preparations never step.
    assert row['platform']==expect and row['cpu_smoke']==(expect=='cpu') and row.get('mini',False)==mini
    assert row['stepping_executed']==(expect=='cpu' or mini)
    assert pathlib.Path(row['cache']).resolve()==(run/'cache/jax').resolve()
    bindings=artifact_bindings(row['aot'],run/'cache/jax')
    program=row['domain'] if (expect=='gpu' or mini) else 'launch_smoke_'+row['variant']  # CPU smoke names the mini by variant
    assert all(pathlib.Path(b['blob_path']).parent.name==program for b in bindings)
    selected[row['variant'],row['domain']]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'artifacts':bindings}
assert set(selected)=={(v,d) for v in ('off','on') for d in ('d01','d02')},selected
prepared={'pin':pin,'preparations':{':'.join(k):v for k,v in selected.items()},
          'metadata':{'nested_aot':{'domains':{'d01':{},'d02':{}}}}}
(run/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
PY
e41_cap=600; if [[ $mini == 1 ]]; then e41_cap=80; fi  # mini cache: 4 small programs
mkdir -p "$RUN/e41"; rm -f "$RUN/e41/verdict.json"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 8,9,12,24,25,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- timeout $e41_cap python "$TREE/scripts/wave_cubin_gate.py" "$RUN/cache/jax" "$RUN/e41" 'jit__advance_chunk*' --run-proof "$RUN/prepared.json" > "$RUN/e41.log" 2>&1 || e41_rc=$?
e41_rc=${e41_rc:-0}
timeout 30 python3 -I "$HERE/e41_verdict.py" "$RUN/e41" "$RUN/e41/verdict.json" || { echo "E41 verdict unreadable (tool rc $e41_rc)"; exit 6; }
if [[ $smoke == 1 ]]; then
  # A CPU smoke cache holds no CUDA ELF: the real gate must reach its coverage refusal, never an import/env error.
  # Defined wave_cubin_gate refusals for a cache without CUDA executables (measured SM05c: production smoke
  # blobs are not step programs -> 'no stepping executable entries'; mini probe blobs are CPU format).
  [[ $e41_rc == 1 ]] && grep -Eq "ValueError: (no stepping executable entries; cannot pass gate|not a supported serialized GPU executable|no native CUDA ELF in executable|unscanned stepping executable|run program\(s\) absent from scan)" "$RUN/e41.log" \
    && ! grep -Eq "ModuleNotFoundError|ImportError" "$RUN/e41.log" || { echo "E41 smoke traversal FAILED rc=$e41_rc"; exit 4; }
  python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); assert v["e41_gate"]=="MISSING" and not v["acceptable"],v' "$RUN/e41/verdict.json"
  # The same parser on the sealed v0.4.3 bundle receipt (real GPU cache, compact AOT -> INCOMPLETE/STACK PASS).
  timeout 30 python3 -I "$HERE/e41_verdict.py" <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/e41 "$RUN/e41/verdict_bundle_probe.json"
  python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); assert v["e41_gate"]=="INCOMPLETE" and v["acceptable"],v' "$RUN/e41/verdict_bundle_probe.json"
  echo "LAUNCH SMOKE PASS w44-thparams check (inventory + E41 coverage refusal on CPU cache + verdict recorded) $RUN/e41.log"
  exit 0
fi
cat "$RUN/e41.log" "$RUN/e41/verdict.json"
echo "E41 RECORDED tool_rc=$e41_rc $(python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); print(v["e41_gate"], "acceptable" if v["acceptable"] else "NOT-acceptable")' "$RUN/e41/verdict.json")"
exit 0
