#!/usr/bin/env bash
# t-wsmdz: UNMODIFIED WRF phys/module_bl_acm.F (ACMPBL) on the v060 ACM2 oracle driver
# extended with two condensate cases (7: elevated ice, 8: elevated cloud water) that
# exercise EDDYX's moist Richardson branch (imvdif=1) and RQCBLTEN/RQIBLTEN.
# Usage: build_acm2_moist_oracle.sh <build_dir>
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
B="${1:?build dir}"; rm -rf "$B"; mkdir -p "$B"
cp <USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_acm.F "$HERE/acm2_moist_oracle_driver.f90" "$B/"
cd "$B"; set -e
sha256sum module_bl_acm.F acm2_moist_oracle_driver.f90 > "$HERE/acm2_moist_sources.sha256"
F="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none"
$FC $F -c module_bl_acm.F; $FC $F -c acm2_moist_oracle_driver.f90; $FC $F -o acm2_oracle module_bl_acm.o acm2_moist_oracle_driver.o
for c in 7 8; do ./acm2_oracle "$c" > "case_${c}.txt"; python3 "$HERE/../v060/oracle/acm2_dump_to_json.py" "case_${c}.txt" "$HERE/acm2_moist_case_${c}.json"; done
echo OK
