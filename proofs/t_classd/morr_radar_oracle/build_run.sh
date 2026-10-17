#!/bin/bash
# Usage: build_run.sh <workdir> <input_fp32.bin> <input_fp64.bin>
# Builds pristine module_model_constants + module_mp_radar + module_mp_morr_two_moment[_aero]
# (verbatim copies) with the v060 stub module_wrf_error; REAL4 and REAL8 (-fdefault-real-8) arms.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WRF=<USER_HOME>/src/wrf_pristine/WRF
W="$1"; IN32="$2"; IN64="$3"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
[ -x "$FC" ] || FC=gfortran
mkdir -p "$W"
for mode in fp32 fp64; do
  for mod in morr morr_aero; do
    B="$W/${mode}_${mod}"; rm -rf "$B"; mkdir -p "$B"
    cp "$WRF/share/module_model_constants.F" "$WRF/phys/module_mp_radar.F" "$B/"
    src=module_mp_morr_two_moment.F; [ $mod = morr_aero ] && src=module_mp_morr_two_moment_aero.F
    cp "$WRF/phys/$src" "$B/"
    cp "$HERE/../../v060/oracle/morrison_stub_modules.f90" "$HERE/morr_radar_driver.F90" "$B/"
    F="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp"
    # Standing init policy flags; gfortran 14 rejects -finit-real on the aero module's SAVEd
    # automatic arrays, and refl10cm_hm reads no uninitialized storage in either module.
    [ $mod = morr ] && F="$F -finit-real=zero -finit-integer=0 -finit-logical=false"
    [ $mode = fp64 ] && F="$F -fdefault-real-8 -fdefault-double-8"
    D=""; [ $mod = morr_aero ] && D="-DAERO"
    ( cd "$B" && sha256sum module_model_constants.F module_mp_radar.F $src morrison_stub_modules.f90 morr_radar_driver.F90 > SOURCES.sha256 \
      && $FC $F -c morrison_stub_modules.f90 && $FC $F -c module_model_constants.F && $FC $F -c module_mp_radar.F \
      && $FC $F -c $src && $FC $F $D -c morr_radar_driver.F90 \
      && $FC $F -o oracle morrison_stub_modules.o module_model_constants.o module_mp_radar.o ${src%.F}.o morr_radar_driver.o ) || exit 10
    in=$IN32; [ $mode = fp64 ] && in=$IN64
    ( cd "$B" && ./oracle "$in" > out.txt ) || exit 20
    echo "$mode $mod $(grep -c '^DBZ' $B/out.txt) levels"
  done
done
