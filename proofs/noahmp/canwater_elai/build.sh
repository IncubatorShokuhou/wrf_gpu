#!/usr/bin/env bash
# Build the pristine CANWATER oracle (V033-W2) in a LANE build dir: build.sh <build_dir> <cwe_routines.inc>
# Links the COMPILED pristine WRF Noah-MP objects (no WRF rebuild), WRF gfortran flags (REAL = 4 bytes).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$1"; INC="$2"
PHYS=<USER_HOME>/src/wrf_pristine/WRF/phys
FFLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2 -I${PHYS}"
set +eu
# shellcheck disable=SC1091
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -eu
mkdir -p "${B}"; cd "${B}"
cp "${INC}" cwe_routines.inc
cp "${HERE}/canwater_oracle.F90" .
# WRF runtime stubs never executed on this path (same as proofs/noahmp/build_driver.sh)
cat > _wrfstubs.F90 <<'EOS'
subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(*,'(A)') trim(msg)
end subroutine wrf_message
subroutine wrf_error_fatal3(fname, line, msg)
  character(len=*), intent(in) :: fname, msg
  integer, intent(in) :: line
  write(0,'(A)') 'FATAL: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal3
subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
EOS
cat > _modstubs.F90 <<'EOS'
module module_sf_urban
contains
  subroutine urban(); end subroutine urban
  subroutine iri_scheme(); end subroutine iri_scheme
end module module_sf_urban
module module_sf_bep
contains
  subroutine bep(); end subroutine bep
end module module_sf_bep
module module_sf_bep_bem
contains
  subroutine bep_bem(); end subroutine bep_bem
end module module_sf_bep_bem
module module_ra_gfdleta
contains
  subroutine cal_mon_day(); end subroutine cal_mon_day
end module module_ra_gfdleta
EOS
gfortran ${FFLAGS} -c _wrfstubs.F90 -o _wrfstubs.o
gfortran ${FFLAGS} -c _modstubs.F90 -o _modstubs.o
gfortran ${FFLAGS} -c canwater_oracle.F90 -o canwater_oracle.o
gfortran canwater_oracle.o _wrfstubs.o _modstubs.o \
  "${PHYS}/module_sf_noahmplsm.o" "${PHYS}/module_sf_noahmpdrv.o" \
  "${PHYS}/module_sf_gecros.o" "${PHYS}/module_sf_noahmp_glacier.o" \
  "${PHYS}/module_sf_noahmp_groundwater.o" \
  -o canwater_oracle.exe
for t in MPTABLE.TBL SOILPARM.TBL GENPARM.TBL; do ln -sf "<USER_HOME>/src/wrf_pristine/WRF/run/${t}" "${t}"; done
echo "BUILT ${B}/canwater_oracle.exe"
