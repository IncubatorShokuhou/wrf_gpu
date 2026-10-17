#!/bin/bash
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
WORK=${1:?lane build path}
HERE=$(cd "$(dirname "$0")" && pwd)
PRISTINE=${PRISTINE:-<USER_HOME>/src/wrf_pristine/WRF}
FC=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}
mkdir -p "$WORK"
# Standalone first module: byte-verbatim bodies, no instrumented full-WRF executable.
sed '/^END MODULE my2_mod/q' "$PRISTINE/phys/module_mp_milbrandt2mom.F" > "$WORK/my2.f90"
sha256sum "$PRISTINE/phys/module_mp_milbrandt2mom.F" "$WORK/my2.f90" > "$WORK/source.sha256"
cd "$WORK"
"$FC" -O2 -ftree-vectorize -funroll-loops -ffree-form -ffree-line-length-none -fcheck=bounds -finit-real=zero -finit-integer=0 -finit-logical=false -c my2.f90
"$FC" -O2 -ffree-form -ffree-line-length-none -fcheck=bounds "$HERE/oracle.F90" my2.o -o oracle
cp "$PRISTINE/phys/module_mp_milbrandt2mom.F" full_module.f90
printf 'module module_wrf_error\nend module\n' > error_stub.f90
"$FC" -O2 -ftree-vectorize -funroll-loops -ffree-form -ffree-line-length-none -fcheck=bounds -finit-real=zero -finit-integer=0 -finit-logical=false -c error_stub.f90 full_module.f90
"$FC" -O2 -ffree-form -ffree-line-length-none -fcheck=bounds "$HERE/wrapper.F90" full_module.o -o wrapper
