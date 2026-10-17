#!/usr/bin/env bash
# v0.40 SLUCM (sf_urban_physics=1) pristine-WRF oracle: build + run.
#  * UNMODIFIED: phys/module_sf_urban.F, module_sf_bep.F, module_sf_bep_bem.F,
#    module_sf_bem.F, module_bep_bem_helper.F, share/module_model_constants.F,
#    frame/module_wrf_error.F  (sha256 checked against the pristine tree)
#  * VERBATIM EXTRACTS (sha256 of the extracted line ranges recorded):
#      noahmp_urban  = phys/module_sf_noahmpdrv.F lines 2994-3783
#      CAL_MON_DAY   = phys/module_ra_gfdleta.F   (SUBROUTINE..END SUBROUTINE)
#  * DATA STUB: NOAHMP_TABLES category ids from run/MPTABLE.TBL (MODIS section:
#    ISURBAN=13, LCZ_1..LCZ_11 = 51..61), which read_mp_veg_parameters sets.
# Precisions: fp64 (-fdefault-real-8 -fdefault-double-8) and fp32 (WRF REAL).
# Tables: default = pristine run/URBPARM.TBL; alt1 = IMP_SCHEME 2 + AHOPTION 1 +
# ALHOPTION 1 + BOUNDR/B/G 2; alt2 = TS_SCHEME 2 (force-restore).
# CPU only, pinned to the lane core (taskset -c 8 nice -n 19).
set -o pipefail
set -u
quiet_guard() { if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; }
quiet_guard

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
BUILD_ROOT="${URBAN_ORACLE_BUILD:-<USER_HOME>/wrf_gpu2_lanes/o2-urban/oracle_build}"
PIN="taskset -c 8 nice -n 19"
SAVE="${HERE}/savepoints"

set +u
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
set -e
export OMP_NUM_THREADS=1
export LC_ALL=C

BASEFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none -fallow-argument-mismatch -finit-integer=0"
FFLAGS64="${BASEFLAGS} -fdefault-real-8 -fdefault-double-8"
FFLAGS32="${BASEFLAGS}"

STAGE="${BUILD_ROOT}/stage"
rm -rf "${STAGE}"; mkdir -p "${STAGE}" "${SAVE}"
for f in phys/module_sf_urban.F phys/module_sf_bep.F phys/module_sf_bep_bem.F phys/module_sf_bem.F \
         phys/module_bep_bem_helper.F share/module_model_constants.F frame/module_wrf_error.F; do
  cp "${WRF}/${f}" "${STAGE}/$(basename "$f")"
  a=$(sha256sum "${STAGE}/$(basename "$f")" | cut -d' ' -f1); b=$(sha256sum "${WRF}/${f}" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "FATAL: staged $f differs"; exit 4; }
done
cp "${WRF}/run/URBPARM.TBL" "${STAGE}/URBPARM.TBL.default"
cp "${HERE}/urban_oracle.f90" "${HERE}/regimes.txt" "${HERE}/regimes_branch.txt" "${STAGE}/"

# alt tables (data only; code untouched)
sed -e 's/^IMP_SCHEME: .*/IMP_SCHEME: 2/' -e 's/^AHOPTION: .*/AHOPTION: 1/' \
    -e 's/^ALHOPTION: .*/ALHOPTION: 1/' -e 's/^BOUNDR: .*/BOUNDR: 2/' -e 's/^BOUNDB: .*/BOUNDB: 2/' \
    -e 's/^BOUNDG: .*/BOUNDG: 2/' "${STAGE}/URBPARM.TBL.default" > "${STAGE}/URBPARM.TBL.alt1"
sed -e 's/^TS_SCHEME: .*/TS_SCHEME: 2/' "${STAGE}/URBPARM.TBL.default" > "${STAGE}/URBPARM.TBL.alt2"
for t in default alt1 alt2; do cp "${STAGE}/URBPARM.TBL.${t}" "${SAVE}/URBPARM.TBL.${t}"; done

# verbatim extracts
{
  echo "MODULE noahmp_urban_extract"
  echo "CONTAINS"
  sed -n '2994,3783p' "${WRF}/phys/module_sf_noahmpdrv.F"
  echo "END MODULE noahmp_urban_extract"
} > "${STAGE}/noahmp_urban_extract.F"
EXTRACT_SHA=$(sed -n '2994,3783p' "${WRF}/phys/module_sf_noahmpdrv.F" | sha256sum | cut -d' ' -f1)
head -1 <(sed -n '2994p' "${WRF}/phys/module_sf_noahmpdrv.F") | grep -q "SUBROUTINE noahmp_urban" || { echo "FATAL: extract start"; exit 5; }
sed -n '3783p' "${WRF}/phys/module_sf_noahmpdrv.F" | grep -q "END SUBROUTINE noahmp_urban" || { echo "FATAL: extract end"; exit 5; }
{
  echo "MODULE module_ra_gfdleta"
  echo "CONTAINS"
  awk '/SUBROUTINE CAL_MON_DAY/{p=1} p{print} /END SUBROUTINE CAL_MON_DAY/{exit}' "${WRF}/phys/module_ra_gfdleta.F"
  echo "END MODULE module_ra_gfdleta"
} > "${STAGE}/module_ra_gfdleta_extract.F"
CALMON_SHA=$(awk '/SUBROUTINE CAL_MON_DAY/{p=1} p{print} /END SUBROUTINE CAL_MON_DAY/{exit}' "${WRF}/phys/module_ra_gfdleta.F" | sha256sum | cut -d' ' -f1)
cat > "${STAGE}/noahmp_tables_stub.f90" <<'EOF'
MODULE NOAHMP_TABLES
  ! category ids as set by read_mp_veg_parameters from run/MPTABLE.TBL (MODIS)
  INTEGER :: ISURBAN_TABLE = 13
  INTEGER :: LCZ_1_TABLE = 51, LCZ_2_TABLE = 52, LCZ_3_TABLE = 53, LCZ_4_TABLE = 54
  INTEGER :: LCZ_5_TABLE = 55, LCZ_6_TABLE = 56, LCZ_7_TABLE = 57, LCZ_8_TABLE = 58
  INTEGER :: LCZ_9_TABLE = 59, LCZ_10_TABLE = 60, LCZ_11_TABLE = 61
END MODULE NOAHMP_TABLES
EOF
cat > "${STAGE}/_wrfstubs.f90" <<'EOF'
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor
subroutine wrf_dm_bcast_string(buf, n)
  character(len=*), intent(inout) :: buf
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_string
subroutine wrf_dm_bcast_integer(buf, n)
  integer, intent(in) :: n
  integer, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_integer
subroutine wrf_dm_bcast_real(buf, n)
  integer, intent(in) :: n
  real, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_real
subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
subroutine wrf_abort()
  stop 1
end subroutine wrf_abort
EOF

build() {  # name flags
  local name="$1"; shift
  local dir="${BUILD_ROOT}/build_${name}"
  quiet_guard
  rm -rf "${dir}"; mkdir -p "${dir}"; cp "${STAGE}"/* "${dir}/"
  ( cd "${dir}"
    for f in module_model_constants.F module_wrf_error.F _wrfstubs.f90 module_sf_urban.F \
             module_bep_bem_helper.F module_sf_bem.F module_sf_bep.F module_sf_bep_bem.F \
             noahmp_tables_stub.f90 module_ra_gfdleta_extract.F noahmp_urban_extract.F urban_oracle.f90; do
      quiet_guard
      ${PIN} gfortran "$@" -c "$f"
    done
    ${PIN} gfortran "$@" -o urban_oracle urban_oracle.o noahmp_urban_extract.o module_ra_gfdleta_extract.o \
        noahmp_tables_stub.o module_sf_bep_bem.o module_sf_bep.o module_sf_bem.o module_bep_bem_helper.o \
        module_sf_urban.o module_wrf_error.o module_model_constants.o _wrfstubs.o
    echo "$*" > fflags.txt )
}

run() {  # name prec table [regimes-file suffix]  (suffix "" = regimes.txt, "_branch" = regimes_branch.txt)
  local dir="${BUILD_ROOT}/build_$1" prec="$2" tbl="$3" sfx="${4:-}"
  local rd="${dir}/run_${tbl}${sfx}"
  quiet_guard
  rm -rf "${rd}"; mkdir -p "${rd}"
  cp "${dir}/URBPARM.TBL.${tbl}" "${rd}/URBPARM.TBL"
  ( cd "${rd}" && ${PIN} "${dir}/urban_oracle" "${prec}" "${dir}/regimes${sfx}.txt" > dump.txt 2> stderr.txt )
  ${PIN} <USER_HOME>/miniconda3/bin/python -I "${HERE}/dump_to_npz.py" "${rd}/dump.txt" "${SAVE}/slucm_${prec}_${tbl}${sfx}.npz"
}

build fp64 ${FFLAGS64}
build fp32 ${FFLAGS32}
for t in default alt1 alt2; do
  for sfx in "" _branch; do
    run fp64 fp64 "$t" "$sfx"
    run fp32 fp32 "$t" "$sfx"
  done
done

{
  echo "wrf_root=${WRF}"
  ( cd "${STAGE}" && sha256sum module_sf_urban.F module_sf_bep.F module_sf_bep_bem.F module_sf_bem.F \
      module_bep_bem_helper.F module_model_constants.F module_wrf_error.F URBPARM.TBL.default )
  echo "noahmp_urban_extract module_sf_noahmpdrv.F:2994-3783 sha256=${EXTRACT_SHA}"
  echo "cal_mon_day_extract module_ra_gfdleta.F sha256=${CALMON_SHA}"
  ( cd "${HERE}" && sha256sum urban_oracle.f90 regimes.txt regimes_branch.txt build_and_run.sh dump_to_npz.py make_urban_oracle_inputs.py make_branch_regimes.py )
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags64=${FFLAGS64}"
  echo "fflags32=${FFLAGS32}"
  echo "libm=$(ldd "${BUILD_ROOT}/build_fp64/urban_oracle" | awk '/libm\.so/{print $3}') $(ldd --version | head -n 1)"
  echo "pin=${PIN}"
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${SAVE}/build_manifest.txt"
echo "OK: SLUCM oracle savepoints under ${SAVE}"
