#!/usr/bin/env bash
# F10 oracle: the v060 GF single-column driver (proofs/v060/oracle/gf_oracle_driver.f90) with ONE change:
# ISHALLOW_G3 read from argv[2] (default 1) instead of the hard-coded 1, and an argv[3] forcing mode (1 = the
# WRF caller's nonzero RTHFTEN/RQVFTEN/RTHRATEN and P8W(1) != P(1), t-cfix-cu F18), against UNMODIFIED pristine
# WRF 4.7.1 GF sources.  ishallow=1 output reproduces proofs/v060/savepoints/gf_case_*.json byte-for-byte
# (E83 provenance check, tests/test_t_cfix_cu_gf_ishallow.py).  Usage: build_and_run.sh <build_dir>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "${HERE}/../../../.." && pwd)"  # repo root: oracle -> group -> t_cfix_cu -> proofs -> root
B="$1"; WP=<USER_HOME>/src/wrf_pristine/WRF/phys
FC=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}
mkdir -p "$B"; cd "$B"
for f in module_gfs_machine module_gfs_physcons module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
  cp "$WP/$f.F" .; "$FC" -O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -DWRF_CHEM=0 -DWRF_DFI_RADAR=0 -DNMM_CORE=0 -c "$f.F"; done
cp "$HERE/gf_oracle_driver_ishallow.f90" .
"$FC" -O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -c gf_oracle_driver_ishallow.f90
"$FC" -O2 -o gf_oracle module_gfs_machine.o module_gfs_physcons.o module_cu_gf_deep.o module_cu_gf_sh.o module_cu_gf_wrfdrv.o gf_oracle_driver_ishallow.o
for c in 1 2 3 4 5; do for i in 0 1; do ./gf_oracle "$c" "$i" 0 > "case_${c}_ish${i}.txt"
  python3 "$ROOT/proofs/v060/oracle/gf2_dump_to_json.py" "case_${c}_ish${i}.txt" "gf_case_${c}_ish${i}.json"
  ./gf_oracle "$c" "$i" 1 > "case_${c}_ish${i}_f1.txt"
  python3 "$ROOT/proofs/v060/oracle/gf2_dump_to_json.py" "case_${c}_ish${i}_f1.txt" "gf_case_${c}_ish${i}_f1.json"; done; done
