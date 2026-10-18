"""History TKE_PBL = each PBL's held shared array (WRF Registry.EM_COMMON:1230, Z, rh).

v0.4.0 composites wrote TKE_PBL = 0 for MYJ/QNSE/BouLac/CAM-UW/GBM while original CPU-WRF
varies (proofs/w41_shinhongtke/PBL_TKE_HISTORY_PINS.json). Port conventions: MYJ/QNSE hold
q2 = 2*TKE in State.qke (WRF TKE_MYJ = 0.5*Q2K), BouLac/GBM/Shin-Hong hold TKE, CAM-UW holds
its interface TKE (nz+1) in carry.camuw_pbl. The unused Z top keeps the allocation zero.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shinhong_history import case, payload  # noqa: E402

FACTOR = {2: np.float32(.5), 4: np.float32(.5), 8: np.float32(1.), 12: np.float32(1.)}


@pytest.mark.parametrize("bl", sorted(FACTOR))
def test_held_qke_convention_reaches_history(tmp_path, bl):
    shape = case(bl)[0].qv.shape
    qke = np.random.default_rng(bl).uniform(1e-4, 9., shape).astype(np.float32)
    state, nl = case(bl, tke=qke)
    tke = payload(tmp_path, nl, state)["TKE_PBL"]
    assert tke.dtype == np.float32 and tke.shape == (shape[0] + 1, *shape[1:])
    assert tke[:-1].tobytes() == (FACTOR[bl] * qke).tobytes()
    assert not tke[-1].any()


@pytest.mark.parametrize("bl", (2, 4))
def test_q2_convention_halves_exactly(tmp_path, bl):
    # WRF TKE_MYJ = 0.5*Q2K (module_bl_myjpbl.F:462): MYJ cold seed EPSQ2 = .2 <-> q2 = .4.
    state, nl = case(bl, tke=np.float32(.4))
    assert np.all(payload(tmp_path, nl, state)["TKE_PBL"][:-1] == np.float32(.2))


def test_camuw_carry_tke_reaches_history_through_production_writer(tmp_path):
    from netCDF4 import Dataset
    from _production_writer import _writer
    writer, state = _writer(tmp_path, lambda *a, **k: {})
    nz, ny, nx = state.qv.shape
    state.qke = np.full((nz, ny, nx), 31., np.float32)  # not CAM-UW's TKE
    state.el_pbl = np.zeros((nz, ny, nx), np.float32)
    interface = np.random.default_rng(9).uniform(0., 15., (nz + 1, ny, nx)).astype(np.float32)
    state.camuw_pbl = SimpleNamespace(tke_pbl=interface)  # the callback reads carry.camuw_pbl
    writer._full_variable_set = True
    writer.bundles["d01"].namelist["bl_pbl_physics"] = 9
    result = writer("d01", 3, state)
    with Dataset(result["wrfout"]) as ds:
        assert np.asarray(ds.variables["TKE_PBL"][0], np.float32).tobytes() == interface.tobytes()


def test_camuw_never_takes_state_qke(tmp_path):
    state, nl = case(9, tke=np.float32(31.))
    assert not payload(tmp_path, nl, state)["TKE_PBL"].any()


@pytest.mark.parametrize("bl", (8, 12))
def test_producer_tke_reaches_held_qke(bl, monkeypatch):
    """BouLac/GBM adapters hold the kernel TKE itself in State.qke (factor 1)."""
    import importlib.util
    import jax.numpy as jnp
    from gpuwrf.coupling import scan_adapters as scan
    spec = importlib.util.spec_from_file_location('pbl_hist_fx', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(nz=8, ny=2, nx=2)
    state = f._state(grid)
    energy = jnp.arange(32, dtype=jnp.float32).reshape(4, 8) / 8 + .001

    def producer(*args, **kwargs):
        zero = jnp.zeros((4, 8), jnp.float32)
        return dict(u=zero, v=zero, theta=zero, qv=zero, qc=zero, qi=zero, tke=energy, exch_h=zero,
                    kth=zero, kethl=zero, el=zero, el_pbl=zero, ust=jnp.full((4,), .3, jnp.float32),
                    pblh=jnp.full((4,), 500., jnp.float32))
    monkeypatch.setattr(scan, 'boulac_columns' if bl == 8 else 'gbm_columns', producer)
    adapter = scan.boulac_pbl_adapter if bl == 8 else scan.gbm_pbl_adapter
    out = adapter(state, 18., grid)
    expect = np.moveaxis(np.asarray(energy).reshape(2, 2, 8), -1, 0)
    assert np.asarray(out.qke, np.float32).tobytes() == expect.tobytes()


# Original CPU-WRF lead-0 TKE_PBL bytes (v-prep composites, every mass cell; Z top 0):
# c03 MYJ / c28,c29,c89 CAM-UW 0x3e4ccccd (.2), c06,c57 BouLac 0x38d1b717 (1e-4).
WRF_LEAD0 = {2: np.frombuffer(bytes.fromhex('cdcc4c3e'), np.float32)[0],
             8: np.frombuffer(bytes.fromhex('17b7d138'), np.float32)[0],
             9: np.frombuffer(bytes.fromhex('cdcc4c3e'), np.float32)[0]}


def _initial_carry(bl, mp=8):
    import dataclasses
    import importlib.util
    import jax.numpy as jnp
    from gpuwrf.runtime import operational_mode as op
    spec = importlib.util.spec_from_file_location('pbl_seed_fx', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(nz=8, ny=2, nx=2)
    state = f._state(grid).ensure_conditional_leaves(mp_physics=mp)
    loader_qke = jnp.arange(state.qke.size, dtype=jnp.float32).reshape(state.qke.shape) / 7 + .3  # MYNN-like seed
    state = state.replace(qke=loader_qke)
    if mp == 11:
        state = state.replace(cam_tke_pbl=jnp.full_like(state.cam_tke_pbl, 73.))
    nl = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), mp_physics=mp,
        bl_pbl_physics=bl, cu_physics=0, sf_surface_physics=0, sf_sfclay_physics=0, use_noahmp=False)
    return op._initial_carry_for_run(state, nl), loader_qke


@pytest.mark.parametrize("mp", (8, 11))
@pytest.mark.parametrize("bl", (2, 8, 9))
def test_wrf_cold_seed_gives_original_lead0_tke_pbl(bl, mp):
    carry, loader_qke = _initial_carry(bl, mp)
    nz = loader_qke.shape[0]
    if bl == 9:  # history source = carry.camuw_pbl (interfaces): kts:ktf seeded, top zero
        tke = np.asarray(carry.camuw_pbl.tke_pbl, np.float32)
        assert tke.shape[0] == nz + 1 and not tke[-1].any()
        assert tke[:-1].tobytes() == np.full(tke[:-1].shape, WRF_LEAD0[9]).tobytes()
    else:
        held = np.asarray(carry.state.qke, np.float32)
        assert (FACTOR[bl] * held).tobytes() == np.full(held.shape, WRF_LEAD0[bl]).tobytes()
    if mp == 11:
        cam = np.asarray(carry.state.cam_tke_pbl, np.float32)
        assert cam[:-1].tobytes() == np.full(cam[:-1].shape, WRF_LEAD0[bl]).tobytes()
        assert np.all(cam[-1] == 73.)  # unassigned top memory retained


@pytest.mark.parametrize("bl", (1, 5))
def test_other_pbl_initial_qke_untouched(bl):
    """YSU / MYNN (release default): the loader's QKE reaches the carry unchanged."""
    carry, loader_qke = _initial_carry(bl)
    assert np.asarray(carry.state.qke).tobytes() == np.asarray(loader_qke).astype(np.asarray(carry.state.qke).dtype).tobytes()
