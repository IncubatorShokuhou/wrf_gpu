#!/usr/bin/env bash
# GPU arm (lane w41-da-cycle), ONE official-lock session: the DA cycle loop with N=10 PROD d01 members x 2 windows
# x 6 h in ONE process (zero-update arms: file seam + restart seam), then 2 separate `gpuwrf run` processes of the
# same cases for the in-process == separate bitwise check on the GPU (shared session cache), then the E41 gate.
#   gpu_arm.sh prepare SESSION   CPU only, outside the lock: member wrfinputs + cycle config
#   gpu_arm.sh run SESSION       needs the manager lease /tmp/wrf_gpu2_run_approvals/w41-da-cycle.lease
# Run from a CLEAN DETACHED snapshot of lane/w41-da-cycle (the measured source).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TREE=$(cd -- "$HERE/../.." && pwd)
MODE=${1:?prepare|run}; SESSION=${2:?session dir}
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
N=${W41_MEMBERS:-10}
LEASE=/tmp/wrf_gpu2_run_approvals/w41-da-cycle.lease
cpu_env() { env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$TREE/src" PYTHONDONTWRITEBYTECODE=1 "$@"; }

if [[ $MODE == prepare ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  [[ -e $SESSION ]] && { echo "REFUSED: $SESSION exists" >&2; exit 2; }
  mkdir -p "$SESSION/members"
  cpu_env taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python3 - "$PROD" "$SESSION" "$N" <<'PY'
import json, shutil, sys
from pathlib import Path
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
prod, session, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
members = {}
for k in range(n):
    out = session / "members" / f"m{k:02d}"
    out.mkdir(parents=True)
    shutil.copy2(prod / "wrfinput_d01", out / "wrfinput_d01")
    if k:
        rng = np.random.default_rng(4100 + k)
        with Dataset(out / "wrfinput_d01", "a") as ds:  # interior T noise (outside the 5-row specified zone)
            t = ds["T"][:]
            nz, ny, nx = t.shape[1:]
            w = 6
            pert = np.zeros((nz, ny, nx), np.float32)
            pert[:20, w:ny - w, w:nx - w] = rng.normal(0.0, 0.3, (20, ny - 2 * w, nx - 2 * w)).astype(np.float32)
            ds["T"][:] = t + pert[None]
    members[f"m{k:02d}"] = str(out / "wrfinput_d01")
# Window-0 cases are written here on the CPU (not inside the lock), with the loop's forcing_hours (12 h).
spec = {"source_case": str(prod), "t_a": "2026-07-26_00:00:00", "hours": 12, "members": members,
        "out": str(session / "cases0")}
(session / "cases0_spec.json").write_text(json.dumps(spec, indent=1) + "\n")
cfg = {"schema": "gpuwrf.da.cycle.v1", "source_case": str(prod), "t0": "2026-07-26T00:00:00Z", "window_hours": 6,
       "cycles": 2, "arms": {"zero_file": {"kind": "zero_update"},
                             "zero_restart": {"kind": "zero_update", "seam": "restart"}},
       "initial_cases": {k: str(session / "cases0" / k) for k in members}, "forcing_hours": 12,
       "share_executables": True, "require_platform": "gpu"}
(session / "cycle.json").write_text(json.dumps(cfg, indent=1) + "\n")
print(len(members), "members ->", session / "cycle.json")
PY
  cpu_env taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python3 -m gpuwrf.assimilation.letkf.cycle_loop first-window \
    "$SESSION/cases0_spec.json" || exit $?
  cpu_env python3 -m gpuwrf da cycle --config "$SESSION/cycle.json" --out "$SESSION/loop" --dry-run
  exit $?
fi

[[ $MODE == run ]] || { echo "unknown mode $MODE" >&2; exit 2; }
[[ -s $LEASE ]] || { echo "REFUSED: $LEASE missing (manager GPU grant)" >&2; exit 126; }
if git -C "$TREE" symbolic-ref -q HEAD >/dev/null; then echo 'REFUSED: model source must be a detached snapshot' >&2; exit 2; fi
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]] || { echo 'REFUSED: dirty snapshot' >&2; exit 2; }
[[ -s $SESSION/cycle.json && ! -e $SESSION/loop ]] || { echo "REFUSED: prepare $SESSION first (fresh loop dir)" >&2; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi

if [[ ${3:-} != --locked ]]; then
  rc=0
  "$TREE/scripts/with_gpu_lock.sh" --label w41-da-cycle --timeout 3600 -- \
    timeout --signal=TERM --kill-after=15s "${W41_GPU_BOUND_S:-900}s" bash "$0" run "$SESSION" --locked || rc=$?
  echo "$rc" > "$SESSION/gpu_session_rc.txt"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: E41 pending at $SESSION"; exit 3; fi
  cpu_env "$TREE/scripts/with_ram_slot.sh" --label w41-da-cycle-e41 --max 6G --timeout 900 -- nice -n 19 \
    taskset -c 8,9,12,13,24,25,28,29 timeout 900 python3 "$TREE/scripts/wave_cubin_gate.py" "$SESSION/cache" \
    "$SESSION/e41" > "$SESSION/e41.log" 2>&1
  e41=$?
  cpu_env python3 "$HERE/compare_gpu_arm.py" "$SESSION" > "$SESSION/compare.log" 2>&1
  echo "GPU_RC=$rc E41_RC=$e41 COMPARE_RC=$? session=$SESSION"
  exit "$rc"
fi

# --- inside the official lock -------------------------------------------------------------------
[[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]] || { echo 'REFUSED: official GPU lock absent' >&2; exit 126; }
mkdir -p "$SESSION/cache" "$SESSION/triton_cache" "$SESSION/cuda_cache" "$SESSION/tmp"
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${W41_POOL_FRACTION:-0.25}
export GPUWRF_JAX_CACHE_DIR="$SESSION/cache" JAX_COMPILATION_CACHE_DIR="$SESSION/cache"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$SESSION/cache/xla_gpu_per_fusion_autotune_cache_dir"
export TMPDIR="$SESSION/tmp" TRITON_CACHE_DIR="$SESSION/triton_cache" CUDA_CACHE_PATH="$SESSION/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$TREE/src"
cd -- "$TREE"
s=$SECONDS
/usr/bin/time -v python3 -m gpuwrf da cycle --config "$SESSION/cycle.json" --out "$SESSION/loop" \
  > "$SESSION/loop.stdout" 2> "$SESSION/loop.stderr"
echo "loop rc=$? wall=$((SECONDS - s))s" | tee "$SESSION/loop.rc"
# Separate fresh processes (same session cache): 'cont' = the window-0 case of m01 run 12 h continuously (frames
# 00-06Z == in-process window 0; 07-12Z == the restart-seam stream after its zero-increment injection);
# 'c01' = the window-1 file-seam case of m01 (== in-process window 1 of zero_file).
for pair in "cont:cases0/m01:12" "c01:loop/c01/zero_file/prep/cases/m01:6"; do
  IFS=: read -r tag rel hours <<< "$pair"; case_dir=$SESSION/$rel
  [[ -d $case_dir ]] || { echo "missing $case_dir" >> "$SESSION/separate.rc"; continue; }
  s=$SECONDS
  /usr/bin/time -v python3 -m gpuwrf run --input-dir "$case_dir" --output-dir "$SESSION/separate/$tag" \
    --scratch-dir "$SESSION/tmp/sep_$tag" --domain d01 --max-dom 1 --hours "$hours" \
    > "$SESSION/separate_$tag.stdout" 2> "$SESSION/separate_$tag.stderr"
  echo "separate $tag rc=$? wall=$((SECONDS - s))s" | tee -a "$SESSION/separate.rc"
done
exit 0
