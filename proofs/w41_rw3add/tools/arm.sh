#!/usr/bin/env bash
# w41-rw3add copy of w41-bundle041 48a7e341f3 arm.sh (only TMPDIR changed). ONE product-CLI arm (release defaults, no GPUWRF_*
# model flag). Runs inside a systemd scope of the session,
# directly (priming: writes the bundle cache) or as the command of bundle_consumer.sh (replay: P read-only, writes -> layer).
# usage: arm.sh ENVFILE NAME INPUT_DIR OUT_DIR BUDGET_S -- CLI_ARGS...
# OUT_DIR (outside every bundle) gets env.txt source_head.txt device.txt t0_unix run.log time.txt rc.txt wrfout/ proof/
# mtimes.json (original wrfout mtimes, E162) wrfout_sha.txt. Exit = the CLI's rc (5 = device assert failed).
set -uo pipefail
(( $# >= 6 )) && [[ $6 == -- ]] || { sed -n 2,7p "$0"; exit 2; }
ENVFILE=$1 NAME=$2 IN=$3 OUT=$4 BUDGET=$5; shift 6
source "$ENVFILE"
export JAX_PLATFORMS=cuda TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w41-rw3add/tmp/$NAME
mkdir -p "$TMPDIR" "$OUT"; cd "$OUT" || exit 2
env | grep -E "^(GPUWRF|JAX|XLA|CUDA|TRITON|PYTHONPATH)" | grep -v TOKEN | sort > env.txt
if env | grep -E "^GPUWRF_" | grep -v -E "^GPUWRF_(JAX_CACHE_DIR|XLA_AUTOTUNE_CACHE_DIR|GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)=" > foreign_flags.txt; then
  echo "refuse: GPUWRF_* model flags in the environment"; cat foreign_flags.txt; exit 2; fi
git -C "${PYTHONPATH%/src}" rev-parse HEAD > source_head.txt
XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 120 python3 -c "import jax; d = jax.devices()[0]; assert d.platform == 'gpu', d; print(d)" > device.txt 2>device.err || exit 5
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
