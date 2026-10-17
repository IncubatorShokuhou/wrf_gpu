#!/usr/bin/env bash
# Compile unchanged WRF sources in the caller's lane directory (E104).
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
out=${1:?absolute lane build directory}
wrf=${WRF_PRISTINE_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}
fc=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
mkdir -p "$out/tmp"
export TMPDIR="$out/tmp"
cp "$wrf/share/module_model_constants.F" "$out/"
for name in module_mp_etanew module_ra_flg module_ra_gfdleta; do
  cp "$wrf/phys/$name.F" "$out/"
done
cp "$here/"*.F90 "$out/"
if [ "${FLG_SAVEPOINTS:-0}" = 1 ]; then
  python "$here/instrument.py" "$out/module_ra_flg.F"
fi
if [ "${GFDL_SAVEPOINTS:-0}" = 1 ]; then
  python "$here/gfdl_instrument.py" "$out/module_ra_gfdleta.F"
fi
if [ "${GFDL_SW_SAVEPOINTS:-0}" = 1 ]; then
  python "$here/gfdl_sw_instrument.py" "$out/module_ra_gfdleta.F"
fi
if [ "${GFDL_LW_SAVEPOINTS:-0}" = 1 ]; then
  python "$here/gfdl_lw_instrument.py" "$out/module_ra_gfdleta.F"
fi
if [ "${GFDL_RADFS_SAVEPOINTS:-0}" = 1 ]; then
  python "$here/gfdl_radfs_instrument.py" "$out/module_ra_gfdleta.F"
fi
if [ "${GFDL_RADTN_SAVEPOINTS:-0}" = 1 ]; then
  python "$here/gfdl_radtn_instrument.py" "$out/module_ra_gfdleta.F"
fi
cd "$out"
sha256sum ./*.F ./*.F90 > sources.sha256
flags=(-O2 -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -DRWORDSIZE=4 -DIWORDSIZE=4 -finit-real=zero -finit-integer=0 -finit-logical=false)
if [ "${FLG_SAVEPOINTS:-0}" = 1 ]; then flags+=(-DFLG_SAVEPOINTS=1); fi
if [ "${GFDL_SAVEPOINTS:-0}" = 1 ]; then flags+=(-DGFDL_SAVEPOINTS=1); fi
if [ "${GFDL_SW_SAVEPOINTS:-0}" = 1 ]; then flags+=(-DGFDL_SW_SAVEPOINTS=1); fi
if [ "${GFDL_LW_SAVEPOINTS:-0}" = 1 ]; then flags+=(-DGFDL_LW_SAVEPOINTS=1); fi
if [ "${GFDL_RADFS_SAVEPOINTS:-0}" = 1 ]; then flags+=(-DGFDL_RADFS_SAVEPOINTS=1); fi
if [ "${GFDL_RADTN_SAVEPOINTS:-0}" = 1 ]; then flags+=(-DGFDL_RADTN_SAVEPOINTS=1); fi
read -r -a variant_flags <<< "${FLG_VARIANT_FLAGS:-}"
flags+=("${variant_flags[@]}")
"$fc" "${flags[@]}" -c stubs.F90 module_model_constants.F module_mp_etanew.F module_ra_flg.F module_ra_gfdleta.F oracle.F90
"$fc" -o oracle stubs.o module_model_constants.o module_mp_etanew.o module_ra_flg.o module_ra_gfdleta.o oracle.o
for name in tables leaves transport gfdl_support gfdl_tables gfdl_emission ferrier_tables; do
  if [ -f "$name.F90" ]; then
    "$fc" "${flags[@]}" "$name.F90" stubs.o module_model_constants.o module_mp_etanew.o module_ra_flg.o module_ra_gfdleta.o -o "$name"
  fi
done
for name in tr49t85 tr49t67 tr67t85 co2_trans; do
  ln -sf "$wrf/run/$name" "$name"
done
"$fc" --version > compiler.txt
