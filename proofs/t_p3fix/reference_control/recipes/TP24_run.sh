#!/usr/bin/env bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
mkdir -p <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24/{triton,cuda,tmp}
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24/triton
export CUDA_CACHE_PATH=<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24/cuda
export TMPDIR=<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24/tmp
export TRITON_ALWAYS_COMPILE=1
taskset -c 16 nice -n 19 timeout --kill-after=5 40 python <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24_decode.py || exit $?
exec taskset -c 16 nice -n 19 timeout --kill-after=5 100 gdb -q -batch -x <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24_gdb.cmd --args <USER_HOME>/miniconda3/bin/python <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP05_passprobe.py <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24/actual.ttir <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24/compile
