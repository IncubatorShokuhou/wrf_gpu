#!/usr/bin/env bash
# v45-rc v0.4.5: ONE product-CLI arm (release defaults, no GPUWRF_* model flag) = the exact child of every GPU arm.
# Runs under arm_timer.py, directly (priming: writes the bundle cache) or as the command of P/tools/bundle_consumer.sh
# (replay: P read-only, writes go to the consumer layer). No GNU timeout (the cap is arm_timer's).
# usage: arm.sh ENVFILE NAME INPUT_DIR OUT_DIR -- CLI_ARGS...
# OUT_DIR (outside every bundle) gets env_product.txt env_child.txt source_head.txt device.txt t0_unix run.log time.txt
# rc.txt t_end_unix wrfout/ proof/ scratch/. Exit = the CLI's rc (5 = device assertion failed, 2 = refused).
# Device selection: product child = JAX_PLATFORMS=cuda (never 'gpu': it expands to cuda+rocm). CPU launch smoke ONLY
# (W45R_LAUNCH_SMOKE=1, no GPU lock held): the product env is recorded first, then JAX_PLATFORMS=cpu and
# CUDA_VISIBLE_DEVICES='' are substituted, and the duration arguments are replaced by W45R_SMOKE_DURATION (or --dry-run is
# appended for W45R_SMOKE_ARM_MODE=dryrun). Nothing else differs.
set -uo pipefail
(( $# >= 5 )) && [[ $5 == -- ]] || { sed -n 2,11p "$0"; exit 2; }
ENVFILE=$1 NAME=$2 IN=$3 OUT=$4; shift 5
export PATH=<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C
source "$ENVFILE" || exit 2
# The official lock token is a non-secret process capability, required by nested preflight.
for _secret in $(compgen -e | grep -iE "TOKEN|PASSW|SECRET|API_?KEY|EARTHDATA|CREDENTIAL|_COOKIE$"); do
  [[ $_secret == GPUWRF_GPU_LOCK_TOKEN ]] || unset "$_secret"
done
unset _secret JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda TMPDIR=${W45R_TMP_ROOT:?}/$NAME
mkdir -p "$TMPDIR" "$OUT" || exit 2
cd "$OUT" || exit 2
redact() { grep -v -i -E '(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|PRIVATE_?KEY)=' ; }
printf 'device=cuda\nsource=%s\ncache=%s\n' "$PYTHONPATH" "$GPUWRF_JAX_CACHE_DIR" > env_product.txt
if compgen -e | grep -E '^GPUWRF_' | grep -v -E '^GPUWRF_(JAX_CACHE_DIR|XLA_AUTOTUNE_CACHE_DIR|GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$' > foreign_flags.txt; then
  echo "refuse: GPUWRF_* model flags in the environment"; true; exit 2; fi
if compgen -e | grep -E '^(XLA_FLAGS$|CUDA_ROOT$|TF_)' > foreign_xla.txt; then echo "refuse: XLA_FLAGS/CUDA_ROOT/TF_* set (E16/E107)"; exit 2; fi
git -C "${PYTHONPATH%/src}" rev-parse HEAD > source_head.txt || exit 2
ARGS=("$@")
EXPECT=gpu
if [[ ${W45R_LAUNCH_SMOKE:-0} == 1 ]]; then
  [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]] || { echo "refuse: CPU smoke under the official GPU lock"; exit 2; }
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
  EXPECT=cpu
  if [[ "$IN" != "${W45R_CPU_SWISS_INPUT:?}" && ${W45R_PROFILE:-0} != 1 ]]; then
    ARGS+=(--dry-run)
  elif [[ ${W45R_SMOKE_ARM_MODE:-model} == dryrun ]]; then ARGS+=(--dry-run)
  else
    # Preserve the exact requested duration/argv. cli_driver bounds the actual
    # model to its first root only AFTER validating that full duration (E225).
    :
  fi
fi
printf 'device=%s\nsource=%s\ncache=%s\n' "$JAX_PLATFORMS" "$PYTHONPATH" "$GPUWRF_JAX_CACHE_DIR" > env_child.txt
printf '%s\n' "python3 -m gpuwrf.cli run --input-dir $IN --output-dir $OUT/wrfout --proof-dir $OUT/proof --scratch-dir $OUT/scratch ${ARGS[*]}" > command.txt
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -c "import jax, sys; d = jax.devices()[0]; print(d, d.platform); sys.exit(0 if d.platform == sys.argv[1] else 5)" "$EXPECT" > device.txt 2> device.err || { echo "device assertion failed (want $EXPECT)"; exit 5; }
date +%s.%N > t0_unix
MODEL=(python3 <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/cli_driver.py run --input-dir "$IN" --output-dir "$OUT/wrfout" --proof-dir "$OUT/proof" --scratch-dir "$OUT/scratch" "${ARGS[@]}")
if [[ ${W45R_PROFILE:-0} == 1 ]]; then
  umask 077
  if [[ ${W45R_LAUNCH_SMOKE:-0} == 1 ]]; then
    MODEL=(python3 <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/cpu_profile_driver.py run --input-dir "$IN" --output-dir "$OUT/wrfout" --proof-dir "$OUT/proof" --scratch-dir "$OUT/scratch" "${ARGS[@]}")
  fi
  mkdir -p "$OUT/profile"
  TRACE=cuda,nvtx; NSYS_EXTRA=(--cuda-graph-trace=node)
  if [[ ${W45R_LAUNCH_SMOKE:-0} == 1 ]]; then TRACE=nvtx; NSYS_EXTRA=(); fi
  MODEL=(nsys profile --trace="$TRACE" "${NSYS_EXTRA[@]}" --sample=none --cpuctxsw=none --force-overwrite=false --output="$OUT/profile/roots" "${MODEL[@]}")
fi
/usr/bin/time -v -o time.txt "${MODEL[@]}" > run.log 2>&1
rc=$?
date +%s.%N > t_end_unix
echo "$rc" > rc.txt
if [[ ${W45R_LAUNCH_SMOKE:-0} == 1 && "$IN" != "${W45R_CPU_SWISS_INPUT:?}" && $rc == 0 ]]; then
  JAX_PLATFORMS=cpu python3 <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/cpu_plan_record.py "$OUT" "$IN" || exit 6
fi
echo "arm $NAME rc=$rc"
exit "$rc"
