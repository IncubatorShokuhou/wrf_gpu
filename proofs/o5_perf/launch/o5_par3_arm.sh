#!/usr/bin/env bash
# o5-perf ONE WN3 3-nest parallel-cases arm (product runner of the sealed bundle, release defaults, C-auto per case).
# usage: o5_par3_arm.sh BUNDLE NAME OUT_DIR BUDGET_S HOURS CASE_DIR...
# OUT_DIR gets env.txt source_head.txt device.txt t0_unix t_end_unix run.log rc.txt cases/<case>/... mtimes.json wrfout_sha.txt
# wrfouts are hashed then deleted (disk rule 2026-10-24); parallel_run.json and per-case receipts/proofs are kept.
set -uo pipefail
(( $# >= 6 )) || { sed -n 2,5p "$0"; exit 2; }
P=$1 NAME=$2 OUT=$3 BUDGET=$4 HOURS=$5; shift 5
case "$OUT" in <DATA_ROOT>/wrf_gpu2/scratch/o5-perf/*) ;; *) echo "refuse: OUT outside o5-perf scratch"; exit 2;; esac
source "$P/bundle.env"
export TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/tmp/$NAME
if [[ "${O5_LAUNCH_SMOKE:-0}" == 1 ]]; then export JAX_PLATFORMS=cpu; else export JAX_PLATFORMS=cuda; fi
mkdir -p "$TMPDIR" "$OUT"; cd "$OUT" || exit 2
if env | grep -E "^GPUWRF_" | grep -v -E "^GPUWRF_(JAX_CACHE_DIR|XLA_AUTOTUNE_CACHE_DIR|GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)=" > foreign_flags.txt; then
  echo "refuse: GPUWRF_* model flags in the environment"; cat foreign_flags.txt; exit 2; fi
[[ -z "${XLA_FLAGS:-}" && -z "${CUDA_ROOT:-}" ]] || { echo "refuse: XLA_FLAGS/CUDA_ROOT set"; exit 2; }
env | grep -E "^(GPUWRF|JAX|XLA|CUDA|TRITON|PYTHONPATH|TF_CPP)" | grep -v TOKEN | sort > env.txt
git -C "${PYTHONPATH%/src}" rev-parse HEAD > source_head.txt
if [[ "${O5_LAUNCH_SMOKE:-0}" != 1 ]]; then
  XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 120 python3 -c "import jax; d = jax.devices()[0]; assert d.platform == 'gpu', d; print(d)" > device.txt 2>device.err || exit 5
else
  echo "smoke: device assert skipped (CPU)" > device.txt
fi
date +%s.%N > t0_unix
timeout --signal=TERM --kill-after=30 "$BUDGET" bash "${PYTHONPATH%/src}/scripts/run_parallel_cases.sh" --out-root "$OUT/cases" \
  --max-parallel 3 --host-gb-per-case 8 "$@" -- --domains-from-namelist --emit-initial-history --hours "$HOURS" > run.log 2>&1
rc=$?
date +%s.%N > t_end_unix
echo "$rc" > rc.txt
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
m = {c.name: {p.name: p.stat().st_mtime for p in sorted(c.rglob("wrfout_d0*")) if p.is_file()} for c in sorted((out / "cases").glob("*")) if c.is_dir()}
print(json.dumps(m, indent=1), file=open(out / "mtimes.json", "w"))
PY
(cd "$OUT/cases" 2>/dev/null && find . -type f -name 'wrfout_d0*' -print0 | sort -z | xargs -0 -r sha256sum > ../wrfout_sha.txt)
[[ -d "$OUT/cases" ]] && find "$OUT/cases" -type f -name 'wrfout_d0*' -delete
echo "par3 arm $NAME rc=$rc"
exit "$rc"
