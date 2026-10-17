#!/usr/bin/env bash
# Caller supplies QUIET, CPU-only env, affinity/nice and timeout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$1"
SOURCE="<USER_HOME>/src/wrf_pristine/WRF/phys/module_cu_nsas.F"
export PATH="<USER_HOME>/miniconda3/envs/wrfbuild/bin:$PATH"
ulimit -c 0
for MODE in r4 r8; do
  BUILD="$WORK/build_$MODE"
  mkdir -p "$BUILD"
  cp "$SOURCE" "$BUILD/module_cu_nsas.F"
  cp "$HERE/nsas96_oracle.F90" "$BUILD/"
  cd "$BUILD"
  FLAGS="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0"
  if [ "$MODE" = r8 ]; then FLAGS="$FLAGS -fdefault-real-8 -fdefault-double-8"; fi
  printf '%s\n' "$FLAGS" > flags.txt
  gfortran $FLAGS -c module_cu_nsas.F
  gfortran $FLAGS -c nsas96_oracle.F90
  gfortran -O2 -o nsas96_oracle module_cu_nsas.o nsas96_oracle.o
  sha256sum "$SOURCE" module_cu_nsas.F > source.sha256
  for CASE in 1 2 3 4 5; do
    ./nsas96_oracle "$CASE" > "$WORK/case${CASE}_${MODE}.txt"
  done
done
