#!/usr/bin/env bash
# Build the t-wsmdz real-column WSM5/WSM6 oracle from UNMODIFIED pristine WRF
# sources (copied verbatim into a lane-dir build, E104) with the WRF
# production optimisation flags of configure.wrf (FCOPTIM, REAL4, no PROMOTION).
# Usage: build_wsm_cols.sh <build_dir>
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
B="${1:?build dir}"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
export PATH="<USER_HOME>/miniconda3/envs/wrfbuild/bin:${PATH}"
rm -rf "${B}"; mkdir -p "${B}"
cp "${WRF}/share/module_model_constants.F" "${WRF}/frame/libmassv.F" \
   "${WRF}/phys/module_mp_radar.F" "${WRF}/phys/module_mp_wsm5.F" \
   "${WRF}/phys/mic-wsm5-3-5-code.h" "${WRF}/phys/mic-wsm5-3-5-locvar.h" \
   "${WRF}/phys/mic-wsm5-3-5-callsite.h" "${WRF}/phys/ccpp_kind_types.f90" \
   "${WRF}/phys/physics_mmm/module_libmassv.F90" "${WRF}/phys/physics_mmm/mp_radar.F90" \
   "${WRF}/phys/physics_mmm/mp_wsm6.F90" "${WRF}/phys/physics_mmm/mp_wsm6_effectRad.F90" \
   "${WRF}/phys/module_mp_wsm6.F" "${B}/"
cp "${HERE}/../v060/oracle/morrison_stub_modules.f90" "${HERE}/wsm_cols_driver.f90" "${B}/"
cd "${B}"
sha256sum *.F *.F90 *.f90 *.h > source_sha256.txt
OPT="-O2 -ftree-vectorize -funroll-loops -w -ffree-line-length-none -ffpe-summary=none -cpp -I."
FIX="-O2 -ftree-vectorize -funroll-loops -w -ffixed-line-length-none -ffpe-summary=none -cpp -ffixed-form -I."
set -e
${FC} ${OPT} -c morrison_stub_modules.f90
${FC} ${OPT} -ffree-form -c module_model_constants.F
${FC} ${FIX} -c libmassv.F
${FC} ${OPT} -ffree-form -c module_mp_radar.F
${FC} ${OPT} -ffree-form -c module_mp_wsm5.F
${FC} ${OPT} -c ccpp_kind_types.f90
${FC} ${OPT} -c module_libmassv.F90
${FC} ${OPT} -c mp_radar.F90
${FC} ${OPT} -c mp_wsm6.F90
${FC} ${OPT} -c mp_wsm6_effectRad.F90
${FC} ${OPT} -ffree-form -c module_mp_wsm6.F
${FC} ${OPT} -c wsm_cols_driver.f90
${FC} ${OPT} -o wsm_cols morrison_stub_modules.o module_model_constants.o libmassv.o \
   module_mp_radar.o module_mp_wsm5.o ccpp_kind_types.o module_libmassv.o mp_radar.o \
   mp_wsm6.o mp_wsm6_effectRad.o module_mp_wsm6.o wsm_cols_driver.o
echo BUILD_OK
