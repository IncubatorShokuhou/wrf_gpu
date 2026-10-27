#!/usr/bin/env bash
# o5-perf ONE product-CLI arm from a sealed bundle (release defaults) + an explicit extra-env file (logging only).
# Adapted from P/meta/arm.sh (w43-release); changes: TMPDIR under o5-perf scratch, EXTRA env file, smoke mode.
# usage: o5_arm.sh BUNDLE NAME INPUT_DIR OUT_DIR BUDGET_S EXTRA_ENV -- CLI_ARGS...
# OUT_DIR gets env.txt source_head.txt device.txt t0_unix t_end_unix run.log time.txt rc.txt proof/ wrfout/ mtimes.json wrfout_sha.txt
# O5_LAUNCH_SMOKE=1: no device assert; JAX_PLATFORMS=cpu; the real CLI must start and refuse at the production GPU guard.
set -uo pipefail
(( $# >= 7 )) && [[ $7 == -- ]] || { sed -n 2,7p "$0"; exit 2; }
P=$1 NAME=$2 IN=$3 OUT=$4 BUDGET=$5 EXTRA=$6; shift 7
case "$OUT" in <DATA_ROOT>/wrf_gpu2/scratch/o5-perf/*) ;; *) echo "refuse: OUT outside o5-perf scratch"; exit 2;; esac
source "$P/bundle.env"
export TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/tmp/$NAME
if [[ "${O5_LAUNCH_SMOKE:-0}" == 1 ]]; then export JAX_PLATFORMS=cpu; else export JAX_PLATFORMS=cuda; fi
mkdir -p "$TMPDIR" "$OUT"; cd "$OUT" || exit 2
if env | grep -E "^GPUWRF_" | grep -v -E "^GPUWRF_(JAX_CACHE_DIR|XLA_AUTOTUNE_CACHE_DIR|GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)=" > foreign_flags.txt; then
  echo "refuse: GPUWRF_* model flags in the environment"; cat foreign_flags.txt; exit 2; fi
[[ -z "${XLA_FLAGS:-}" && -z "${CUDA_ROOT:-}" ]] || { echo "refuse: XLA_FLAGS/CUDA_ROOT set"; exit 2; }
# EXTRA may only set TF_CPP_* logging variables (not in any cheap key / exec key: aot_cheap_key._EXEC_ENV_FLAGS).
grep -v -E '^(#.*|export TF_CPP_[A-Z_]+=[A-Za-z0-9_,=.]+|)$' "$EXTRA" && { echo "refuse: EXTRA env sets non-logging vars"; exit 2; }
source "$EXTRA"
env | grep -E "^(GPUWRF|JAX|XLA|CUDA|TRITON|PYTHONPATH|TF_CPP)" | grep -v TOKEN | sort > env.txt
git -C "${PYTHONPATH%/src}" rev-parse HEAD > source_head.txt
if [[ "${O5_LAUNCH_SMOKE:-0}" != 1 ]]; then
  XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 120 python3 -c "import jax; d = jax.devices()[0]; assert d.platform == 'gpu', d; print(d)" > device.txt 2>device.err || exit 5
else
  echo "smoke: device assert skipped (CPU)" > device.txt
fi
date +%s.%N > t0_unix
/usr/bin/time -v -o time.txt timeout --signal=TERM --kill-after=15 "$BUDGET" python3 -m gpuwrf.cli run --input-dir "$IN" \
  --output-dir "$OUT/wrfout" --proof-dir "$OUT/proof" --scratch-dir "$OUT/scratch" "$@" > run.log 2>&1
rc=$?
date +%s.%N > t_end_unix
echo "$rc" > rc.txt
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(json.dumps({p.name: p.stat().st_mtime for p in sorted((out / "wrfout").glob("wrfout_d0*"))}, indent=1),
      file=open(out / "mtimes.json", "w"))
PY
(cd wrfout 2>/dev/null && sha256sum wrfout_d0* > ../wrfout_sha.txt)
echo "arm $NAME rc=$rc"
exit "$rc"
