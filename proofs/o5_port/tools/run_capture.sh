#!/usr/bin/env bash
# usage: run_capture.sh LABEL CASE [capture_real.py args...]   (CPU only; lane o5-port)
# Swiss = <USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu ; PROD = <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi
avail=$(df --output=avail -BG <DATA_ROOT> | tail -1 | tr -dc 0-9); if [ "$avail" -lt 365 ]; then echo "DISK_LOW ${avail}G"; exit 3; fi
LABEL=$1; CASE=$2; shift 2
L=<USER_HOME>/wrf_gpu2_lanes/o5-port; S=<DATA_ROOT>/wrf_gpu2/scratch/o5-port
AL=${O5_ALIAS:-$L/alias}  # stable import alias path (E58/E94); base and candidate must share it
TOOLS=$(cd "$(dirname "$0")" && pwd)
echo "alias -> $(readlink $AL) @ $(git -C $AL rev-parse HEAD 2>/dev/null || echo NOGIT-see-base_hl_provenance.json) dirty=$(git -C $AL status --porcelain -- src 2>/dev/null | wc -l)" > $L/$LABEL.log
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 CUDA_VISIBLE_DEVICES=
scripts_dir=$AL/scripts
exec >>$L/$LABEL.log 2>&1
"$scripts_dir/with_ram_slot.sh" --label o5-port --max 12G --timeout 5400 -- \
  taskset -c 8,9,12,24,25,28 nice -n 19 timeout 2400 \
  python "$TOOLS/capture_real.py" --alias $AL --case "$CASE" --out $S/$LABEL "$@"
echo "rc=$?"
