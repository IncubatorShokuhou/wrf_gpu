"""Radiation-only MYNN cloud merge versus the literal pristine WRF block."""
import ctypes
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).parents[1]), str(Path(__file__).parents[1] / "v041")]
from test_v022_camuw_pbl import _namelist, _grid
from test_icloud import _real_column_state, FIXTURE, SAVEPOINT, NAMES
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.io.namelist_binding import bound_options
from gpuwrf.physics import cloud_fraction as cf
from gpuwrf.runtime import operational_mode as om


@pytest.fixture(scope="module")
def merge_oracle(tmp_path_factory):
    text = Path("data/wrf_pristine/WRF/phys/module_radiation_driver.F").read_text()
    start = text.index("        IF ( PRESENT ( CLDFRA_BL ) .AND.  PRESENT ( QC_BL ) ) THEN")
    end = text.index("        IF ( PRESENT (cldfra_mp_all) ) THEN", start)
    block = text[start:end].replace("PRESENT ( CLDFRA_BL )", ".true.").replace("PRESENT ( QC_BL )", ".true.")
    code = """subroutine merge(n,nz,itimestep,icloud_bl,cldfra,qc,qi,cldfra_bl,qc_bl,qi_bl) bind(C)
use iso_c_binding
implicit none
integer(c_int),value :: n,nz,itimestep,icloud_bl
real(c_float),intent(inout)::cldfra(n,nz,1),qc(n,nz,1),qi(n,nz,1)
real(c_float),intent(in)::cldfra_bl(n,nz,1),qc_bl(n,nz,1),qi_bl(n,nz,1)
integer::i,j,k,its,ite,jts,jte,kts,kte
its=1;ite=n;jts=1;jte=1;kts=1;kte=nz
""" + block + """
contains
subroutine wrf_debug(level,message)
integer,intent(in)::level
character(*),intent(in)::message
end subroutine
end subroutine
"""
    directory = tmp_path_factory.mktemp("pristine_merge")
    source, library = directory / "merge.f90", directory / "merge.so"
    source.write_text(code)
    compiler = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    subprocess.run([compiler, "-O2", "-ffp-contract=off", "-fPIC", "-shared", str(source), "-o", str(library)],
                   check=True, timeout=30, capture_output=True)
    fn = ctypes.CDLL(str(library)).merge
    fn.argtypes = [ctypes.c_int] * 4 + [np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")] * 6
    return fn


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("lead", [0., 3600.])
def test_literal_wrf_merge_and_icloud_bl_zero(merge_oracle, monkeypatch, enabled, lead):
    with np.load(FIXTURE) as data, np.load(SAVEPOINT) as saved:
        grid, state = _real_column_state(({k: data[k] for k in (*NAMES, "xland", "gridkm")}, saved))
    # Real winter columns with additional real-structured active BL ice/water
    # exercise WRF's threshold branches, including the first-step fraction hold.
    state = state.replace(cldfra_bl=jnp.full_like(state.qv, .4), qc_bl=jnp.full_like(state.qv, 2.e-5),
                          qi_bl=jnp.full_like(state.qv, 3.e-6))
    monkeypatch.setenv("GPUWRF_MYNN_SGS_CLOUD", "1")
    before = {name: np.asarray(getattr(state, name)).copy() for name in ("qsq", "qc_bl", "qi_bl", "cldfra_bl")}
    nml = _namelist(grid, icloud_bl=enabled)
    default_columns = pc._rrtmg_column_inputs(state, grid, time_utc="2023-01-15T04:00:00Z",
                                             lead_seconds=lead, use_mp_re=0,
                                             use_mynn_sgs_cloud=False, cloud_mp_physics=8)[0]
    cols = lambda a: np.asfortranarray(np.moveaxis(np.asarray(a), 0, -1).reshape(-1, 44), dtype=np.float32)
    # cal_cldfra1 is separately oracle-qualified; execute the changed BL merge
    # from the pristine source on exactly those caller cloud fractions.
    fraction = np.asfortranarray(default_columns.cloud_fraction.reshape(-1, 44), dtype=np.float32)
    if enabled:
        # Qualify the literal merge itself. The retained default first-call
        # fraction is a separate pre-existing producer (OFF identity forbids
        # changing it in this feature).
        if pc._rrtmg_real_enabled():
            # Release GPUWRF_RRTMG_REAL path: the caller forms that first-call fraction from the
            # REAL-cast state in REAL (physics_couplers._rrtmg_column_inputs); feed the oracle the same.
            with jax.enable_x64(False):
                first = pc._cloud_fraction_columns(pc._rrtmg_real_tree(state))
        else:
            first = pc._cloud_fraction_columns(state)
        fraction = np.asfortranarray(np.asarray(first).reshape(-1, 44), dtype=np.float32)
    qc, qi = cols(state.qc), cols(state.qi)
    merge_oracle(9, 44, 1 if lead == 0 else 2, enabled, fraction, qc, qi,
                 cols(state.cldfra_bl), cols(state.qc_bl), cols(state.qi_bl))
    actual = pc._rrtmg_column_inputs(state, grid, time_utc="2023-01-15T04:00:00Z",
                                     lead_seconds=lead, use_mp_re=0,
                                     **om._rrtmg_pbl_cloud_kwargs(nml))[0]
    for name, expected in (("qc", qc), ("qi", qi), ("cloud_fraction", fraction)):
        got = np.asarray(getattr(actual, name), np.float32).reshape(-1, 44)
        assert np.isfinite(got).all() and np.isfinite(expected).all()
        np.testing.assert_array_equal(got, expected, err_msg=name)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name), value)
    if not enabled:
        merged = pc._rrtmg_column_inputs(state, grid, time_utc="2023-01-15T04:00:00Z",
                                        lead_seconds=3600., use_mp_re=0)[0]
        assert np.any(np.asarray(merged.qc, np.float32) != qc.reshape(3, 3, 44))
        assert np.any(np.asarray(merged.qi, np.float32) != qi.reshape(3, 3, 44))


def test_icloud_bl_binding_default_static_identity_and_icloud3_bypass():
    assert bound_options({"physics": {"icloud_bl": 0}}, "d03")["icloud_bl"] == 0
    nml = _namelist(_grid())
    assert om._rrtmg_pbl_cloud_kwargs(nml) == {}
    assert jax.tree.structure(nml) == jax.tree.structure(replace(nml, icloud_bl=1))
    off = replace(nml, icloud_bl=0, rad_diag=1)
    leaves, structure = jax.tree.flatten(off)
    restored = jax.tree.unflatten(structure, leaves)
    assert restored.icloud_bl == 0 and restored.rad_diag == 1
    assert om._rrtmg_pbl_cloud_kwargs(off)["use_mynn_sgs_cloud"] is False
    assert om._rrtmg_pbl_cloud_kwargs(replace(nml, icloud=3)) == om._rrtmg_pbl_cloud_kwargs(replace(nml, icloud=3, icloud_bl=0))
