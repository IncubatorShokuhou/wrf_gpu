"""Shin-Hong (bl_pbl_physics=11) TKE_PBL / EL_PBL history vs original WRF.

Original CPU-WRF c07 (Swiss, bl=11, shinhong_tke_diag=0) history holds
TKE_PBL = EPSQ2L/2 = 0.005 on every mass level from lead zero on and 0 at the
unused Z top (shinhonginit, module_bl_shinhong.F:2270-2277); the port wrote 0
everywhere (proofs/w41_shinhongtke/C07_TKE_PBL_HISTORY.json). diag=1 rewrites
TKE kts:kte and EL_PBL k>kts every call (:1601-1604).
"""
import dataclasses
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_m7_netcdf_writer import synthetic_case, writer_authority  # noqa: E402

START = datetime(2026, 2, 28, 0, tzinfo=timezone.utc)
SEED = np.float32(.005)  # REAL EPSQ2L/2, the CPU-WRF c07 bytes (0x3ba3d70a)


def payload(tmp_path, nl, state, names=("TKE_PBL", "EL_PBL", "QKE")):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    grid = synthetic_case()[1]
    return prepare_wrfout_payload(state, grid, nl, tmp_path / "out.nc", domain="d01",
        domain_authority=writer_authority(grid, "d01"), valid_time=START, run_start=START,
        lead_hours=1., full_variable_set=True, variable_subset=names).fields


def case(bl, *, tke=SEED, el=None):
    state, _, nl = synthetic_case()
    nz, ny, nx = state.qv.shape
    state.qke = np.full((nz, ny, nx), tke, np.float32) if np.ndim(tke) == 0 else tke
    state.el_pbl = np.zeros((nz, ny, nx), np.float32) if el is None else el
    return state, dict(nl, bl_pbl_physics=bl)


def test_cold_seed_reaches_history_with_zero_top(tmp_path):
    state, nl = case(11)
    tke = payload(tmp_path, nl, state)["TKE_PBL"]
    assert tke.shape == (state.qv.shape[0] + 1, *state.qv.shape[1:]) and tke.dtype == np.float32
    assert tke[:-1].tobytes() == np.full_like(tke[:-1], SEED).tobytes()
    assert not tke[-1].any()


def test_diagnosed_tke_and_el_reach_history(tmp_path):
    rng = np.random.default_rng(11)
    shape = synthetic_case()[0].qv.shape
    tke = rng.uniform(.005, 2., shape).astype(np.float32)
    el = rng.uniform(1., 300., shape).astype(np.float32); el[0] = 0
    state, nl = case(11, tke=tke, el=el)
    fields = payload(tmp_path, nl, state)
    assert fields["TKE_PBL"][:-1].tobytes() == tke.tobytes() and not fields["TKE_PBL"][-1].any()
    assert fields["EL_PBL"][:-1].tobytes() == el.tobytes() and not fields["EL_PBL"][-1].any()


def test_cam_shared_profile_is_written_verbatim(tmp_path):
    state, nl = case(11, tke=31.)
    nz, ny, nx = state.qv.shape
    state.cam_tke_pbl = (np.arange((nz + 1) * ny * nx, dtype=np.float32).reshape(nz + 1, ny, nx) / 64)
    assert payload(tmp_path, nl, state)["TKE_PBL"].tobytes() == state.cam_tke_pbl.tobytes()


@pytest.mark.parametrize("bl", (1, 5))
def test_other_pbl_keep_their_tke_pbl_source(tmp_path, bl):
    # YSU/MYNN never receive shared TKE_PBL (pbl_driver.F); State.qke is not it.
    state, nl = case(bl)
    assert not payload(tmp_path, nl, state)["TKE_PBL"].any()


@pytest.mark.parametrize("bad", (np.nan, np.inf))
def test_nonfinite_held_tke_refused(tmp_path, bad):
    state, nl = case(11)
    state.qke = state.qke.copy(); state.qke.flat[3] = bad
    with pytest.raises(ValueError, match="TKE_PBL history nonfinite"):
        payload(tmp_path, nl, state)


def test_lead_zero_frame_carries_cold_seed_not_wrfinput_qke(tmp_path):
    """Production per-domain writer at own_step 0: the t0 diagnostics overlay
    QKE from wrfinput (absent -> 0), but WRF's t0 TKE_PBL is the seeded array."""
    from netCDF4 import Dataset
    from _production_writer import _writer
    writer, state = _writer(tmp_path, lambda *a, **k: {})
    nz, ny, nx = state.qv.shape
    state.qke = np.full((nz, ny, nx), SEED, np.float32)
    state.el_pbl = np.zeros((nz, ny, nx), np.float32)
    writer._full_variable_set = True
    writer.bundles["d01"].namelist["bl_pbl_physics"] = 11
    result = writer("d01", 0, state)
    with Dataset(result["wrfout"]) as ds:
        tke = np.asarray(ds.variables["TKE_PBL"][0])
        assert tke[:-1].tobytes() == np.full((nz, ny, nx), SEED, np.float32).tobytes()
        assert not tke[-1].any()
        assert not np.asarray(ds.variables["EL_PBL"][0]).any()


def _caller_fixture():
    spec = importlib.util.spec_from_file_location(
        'shin_hist_fixture', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    return f, f._grid(nz=8, ny=2, nx=2)


@pytest.mark.parametrize('diag', [0, 1])
def test_el_pbl_reaches_post_rk_state(diag, monkeypatch):
    """Actual operational caller + post-RK replace: diag=1 publishes the
    kernel's EL_PBL, diag=0 keeps the held field (module_bl_shinhong.F:673,1604)."""
    from gpuwrf.coupling import scan_adapters as scan
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    f, grid = _caller_fixture()
    state = f._state(grid).ensure_conditional_leaves(mp_physics=8)
    held_el = jnp.full_like(state.el_pbl, 7.)
    state = state.replace(el_pbl=held_el)
    cols = lambda value: jnp.moveaxis(value, 0, -1).reshape(4, 8)
    el = jnp.arange(state.el_pbl.size, dtype=jnp.float32).reshape(state.el_pbl.shape) + 1.

    def producer(*args, shinhong_tke_diag=1, **kwargs):
        zero = jnp.zeros((4, 8), jnp.float32)
        return dict(u=zero, v=zero, theta=zero, qv=zero, qc=zero, qi=zero, tke=args[8],
                    exch_h=zero, el_pbl=cols(el) if shinhong_tke_diag == 1 else zero)
    monkeypatch.setattr(scan, 'shinhong_columns', producer)
    monkeypatch.setattr(op, 'surface_adapter', lambda s, *a, **kw: s)
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF', '1')
    monkeypatch.setenv('GPUWRF_MICROPHYSICS_WRF_ORDER', '1')
    nl = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0),
        mp_physics=8, bl_pbl_physics=11, cu_physics=0, sf_surface_physics=0,
        sf_sfclay_physics=0, use_noahmp=False, run_physics=True, rad_rk_tendf=1)
    object.__setattr__(nl, 'shinhong_tke_diag', diag)
    forcing = op._physics_step_forcing(initial_operational_carry(state), nl, 18.,
                                      run_radiation=False, first_timestep=False)
    post = op._apply_physics_non_dry_updates(state, state, forcing.state)
    np.testing.assert_array_equal(post.el_pbl, el if diag == 1 else held_el)


def test_real_kernel_el_matches_adapter_output():
    """The adapter publishes exactly the kernel's EL_PBL (diag=1); the kernel
    itself is BYTE vs pristine WRF (tests/v040/test_shinhong_wrf_oracle.py)."""
    from gpuwrf.coupling.scan_adapters import shinhong_pbl_adapter
    f, grid = _caller_fixture()
    state = f._state(grid).ensure_conditional_leaves(mp_physics=8)
    state = state.replace(qke=jnp.full_like(state.qke, SEED))
    result, out = shinhong_pbl_adapter(state, 18., grid, shinhong_tke_diag=1, _with_sources=True)
    expect = np.moveaxis(np.asarray(out['el_pbl']).reshape(2, 2, 8), -1, 0)
    assert np.asarray(result.el_pbl).tobytes() == expect.astype(np.asarray(result.el_pbl).dtype).tobytes()
    assert expect[1:].any() and not expect[0].any()
    held = shinhong_pbl_adapter(state, 18., grid, shinhong_tke_diag=0)
    assert held.el_pbl is state.el_pbl
