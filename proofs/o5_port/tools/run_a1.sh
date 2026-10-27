#!/usr/bin/env bash
# usage: run_a1.sh LABEL CASE TARGET.pbtxt...   XLA deviceless (A1) census on CPU (lane o5-port)
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi
avail=$(df --output=avail -BG <DATA_ROOT> | tail -1 | tr -dc 0-9); if [ "$avail" -lt 365 ]; then echo "DISK_LOW ${avail}G"; exit 3; fi
LABEL=$1; CASE=$2; shift 2
L=<USER_HOME>/wrf_gpu2_lanes/o5-port; S=<DATA_ROOT>/wrf_gpu2/scratch/o5-port
TOOLS=$(cd "$(dirname "$0")" && pwd)
TARGETS=(); for t in "$@"; do TARGETS+=(--target "$(readlink -f "$t")"); done
echo "alias -> $(readlink $L/alias) @ $(git -C $L/alias rev-parse HEAD 2>/dev/null || echo NOGIT-see-base_hl_provenance.json) dirty=$(git -C $L/alias status --porcelain -- src 2>/dev/null | wc -l)" > $L/$LABEL.log
# Product CUDA command-buffer policy (gpuwrf/__init__ -> configure_command_buffers(default_on=True)) is skipped
# on a CPU process, so it is set explicitly; autotune level 0 because a CompileOnly client cannot autotune.
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 CUDA_VISIBLE_DEVICES= \
  XLA_FLAGS="--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE --xla_enable_command_buffers_during_profiling=true --xla_disable_hlo_passes=fusion-dynamic-memcpy-rewriter --xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=6"
scripts_dir=$L/alias/scripts
exec >>$L/$LABEL.log 2>&1
"$scripts_dir/with_ram_slot.sh" --label o5-port --max 12G --timeout 7200 -- \
  taskset -c 8,9,12,24,25,28 nice -n 19 timeout 5400 \
  python "$TOOLS/a1_deviceless.py" --alias $L/alias --case "$CASE" --out $S/$LABEL "${TARGETS[@]}"
echo "rc=$?"
