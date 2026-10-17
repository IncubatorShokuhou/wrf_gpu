#!/usr/bin/env bash
# Build the two whole-patch drivers against the UNMODIFIED WRF module_cu_tiedtke.F with the WRF
# configure.wrf optimization flags (FCOPTIM -O2 -ftree-vectorize -funroll-loops), REAL*4.
set -eo pipefail
OUT=${1:?build dir}
WRF=<USER_HOME>/src/wrf_pristine/WRF
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"; cd "$OUT"
cp "$WRF/phys/module_cu_tiedtke.F" "$WRF/share/module_model_constants.F" "$HERE/tied_grid.f90" "$HERE/tied_grid1.f90" .
set +u; source <USER_HOME>/miniconda3/etc/profile.d/conda.sh; conda activate wrfbuild; set -u
F="-O2 -ftree-vectorize -funroll-loops -w -ffree-line-length-none -ffpe-summary=none"
gfortran $F -ffree-form -cpp -c module_model_constants.F
gfortran $F -ffree-form -cpp -c module_cu_tiedtke.F
for p in tied_grid tied_grid1; do gfortran $F -c $p.f90; gfortran -O2 -o $p module_model_constants.o module_cu_tiedtke.o $p.o; done
sha256sum module_cu_tiedtke.F "$WRF/phys/module_cu_tiedtke.F" tied_grid tied_grid1
