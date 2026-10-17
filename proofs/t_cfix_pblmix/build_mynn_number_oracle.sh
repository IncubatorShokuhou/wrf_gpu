#!/usr/bin/env bash
# t-cfix-pblmix F8: WRF MYNN DMP_mf + mynn_tendencies number-scalar oracle on real c19 (mp=28) columns.
# (a) driver linked to the PRISTINE build objects (phys/module_bl_mynnedmf.o, WRF FCFLAGS) -> out_pristine.txt
# (b) driver linked to a private WRITE-only copy of the preprocessed module (capture of the internal
#     post-solve sqc2/sqi2/qnc2/qni2/qnwfa2/qnifa2, compiled with the same WRF FCFLAGS) -> out_capture.txt
# The MF/TEN lines of (a) and (b) must be byte-identical (capture is numerically inert).
# Usage: build_mynn_number_oracle.sh <build_dir> <wrfout[,...]> <ncol per file>
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
PHYS=<USER_HOME>/src/wrf_pristine/WRF/phys; SHARE=<USER_HOME>/src/wrf_pristine/WRF/share
B="${1:?build dir}"; WRFOUT="${2:?wrfout}"; NCOL="${3:-6}"
rm -rf "$B"; mkdir -p "$B/pristine" "$B/capture"
cd "$B"; set -e
( cd "$HERE" && python3 mynn_number_columns.py "$WRFOUT" "$NCOL" "$B/cols.txt" "$B/inputs.json" )
WRFF="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
OBJS="$PHYS/module_bl_mynnedmf_common.o $SHARE/module_model_constants.o $PHYS/ccpp_kind_types.o"
( cd pristine
  $FC -cpp $WRFF -I$PHYS -c "$HERE/mynn_number_driver.F90" -o driver.o
  $FC driver.o $PHYS/module_bl_mynnedmf.o $OBJS -o oracle
  ./oracle ../cols.txt > out_pristine.txt )
( cd capture
  python3 - "$PHYS/module_bl_mynnedmf.f90" module_bl_mynnedmf.f90 <<'PY'
import sys
src = open(sys.argv[1]).read()
decl = "\n IMPLICIT NONE\n"
assert src.count(decl) >= 1
src = src.replace(decl, decl + " real(kind_phys), dimension(500), public :: cap_sqc2=0., cap_sqi2=0., cap_qnc2=0.,"
                  " cap_qni2=0., cap_qnwfa2=0., cap_qnifa2=0.  ! t-cfix-pblmix WRITE-only capture\n", 1)
lines = src.split("\n")
start = next(i for i, l in enumerate(lines) if "SUBROUTINE mynn_tendencies(" in l)
end = next(i for i, l in enumerate(lines) if "END SUBROUTINE mynn_tendencies" in l)
qnc = next(i for i in range(start, end) if "IF (bl_mynn_cloudmix > 0 .AND. FLAG_QNC .AND." in lines[i])
lines.insert(end, "  cap_qnc2(kts:kte)=qnc2(kts:kte); cap_qni2(kts:kte)=qni2(kts:kte); "
             "cap_qnwfa2(kts:kte)=qnwfa2(kts:kte); cap_qnifa2(kts:kte)=qnifa2(kts:kte)")
lines.insert(qnc, "  cap_sqc2(kts:kte)=sqc2(kts:kte); cap_sqi2(kts:kte)=sqi2(kts:kte)")
open(sys.argv[2], "w").write("\n".join(lines))
PY
  $FC $WRFF -I$PHYS -c module_bl_mynnedmf.f90 -o module_bl_mynnedmf.o
  $FC -cpp -DCAPTURE $WRFF -I. -I$PHYS -c "$HERE/mynn_number_driver.F90" -o driver.o
  $FC driver.o module_bl_mynnedmf.o $OBJS -o oracle
  ./oracle ../cols.txt > out_capture.txt )
sha256sum "$PHYS/module_bl_mynnedmf.f90" "$PHYS/module_bl_mynnedmf.o" "$HERE/mynn_number_driver.F90" cols.txt > sources.sha256
cmp <(grep -v '^CAP' capture/out_capture.txt) pristine/out_pristine.txt && echo "CAPTURE_INERT byte-identical MF/TEN"
python3 "$HERE/make_mynn_fixture.py" "$B" "$HERE/mynn_number_fixture.json"
echo OK
