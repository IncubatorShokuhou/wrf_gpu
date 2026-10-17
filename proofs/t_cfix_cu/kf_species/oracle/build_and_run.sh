#!/usr/bin/env bash
# F11 KF oracle: proofs/p0_4/oracle/kf_oracle_driver.f90 with ONE change -- a flag mode from argv[2]:
#   0 = mixed phase (original; reproduces proofs/p0_4/savepoints byte-for-byte, E83)
#   1 = Kessler mp1: warm_rain=.TRUE., F_QI=F_QS=.FALSE. (module_physics_init.F:4479-4481)
#   2 = no-ice MP with rain (WSM3-class): F_QI=F_QS=.FALSE.
# against the UNMODIFIED pristine WRF 4.7.1 module_cu_kfeta.F.  Usage: build_and_run.sh <build_dir>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "${HERE}/../../../.." && pwd)"  # repo root: oracle -> group -> t_cfix_cu -> proofs -> root
B="$1"; FC=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}
mkdir -p "$B"; cd "$B"
cp <USER_HOME>/src/wrf_pristine/WRF/phys/module_cu_kfeta.F "$ROOT/proofs/p0_4/oracle/module_wrf_error.f90" "$HERE/kf_oracle_driver_flags.f90" .
"$FC" -O2 -ffree-line-length-none -ffpe-summary=none -c module_wrf_error.f90
"$FC" -O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -c module_cu_kfeta.F
"$FC" -O2 -ffree-line-length-none -ffpe-summary=none -c kf_oracle_driver_flags.f90
"$FC" -O2 -o kf_oracle module_wrf_error.o module_cu_kfeta.o kf_oracle_driver_flags.o
for c in 1 2 3 4; do for m in 0 1 2; do ./kf_oracle "$c" "$m" > "case_${c}_m${m}.txt"
  python3 "$ROOT/proofs/p0_4/oracle/dump_to_json.py" "case_${c}_m${m}.txt" "kf_case_${c}_m${m}.json"; done; done
