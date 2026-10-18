"""w41-aerosol AE08: aer_opt=3 (Thompson mp28 water/ice-friendly aerosol optics) wiring (CPU).

WRF (module_radiation_driver.F:1652-1694, :2374-2392): with aer_opt=3 + mp28 every radiation call runs gt_aod on
p_hyd / dz8w / t_phy / QV / QNWFA / QNIFA, sums TAOD5502D (history 'h'), and calc_aerosol_rrtmg_sw turns the per-layer
550 nm AOD into rural-type RH-dependent band optics for RRTMG_SWRAD (iaer=10).  The optics/flux numerics are bound
to pristine WRF in test_aer3_sw_oracle.py; these tests bind the tables, the product seams and the held history field.
"""
from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import wrf_aer_opt3 as A3

ROOT = Path(__file__).resolve().parents[3]
PRISTINE_PHYS = Path("<USER_HOME>/src/wrf_pristine/WRF/phys")
SWISS = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")


def test_generated_tables_are_the_pristine_data_statements():
    if not (PRISTINE_PHYS / "module_ra_aerosol.F").is_file():
        pytest.skip("pristine WRF source tree unavailable")
    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_aer3", ROOT / "scripts/extract_wrf_aer_opt3_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.extract(PRISTINE_PHYS) == (ROOT / "src/gpuwrf/physics/wrf_aer_opt3_tables.py").read_text()


def test_rural_tables_and_driver_parameters():
    from gpuwrf.physics import wrf_aer_opt3_tables as tab

    assert A3.TAER_TYPE == 1  # module_radiation_driver.F:1015 taer_type = 1 (rural)
    assert np.asarray(tab.RAOD_LUT).shape == np.asarray(tab.SSA_LUT).shape == np.asarray(tab.ASY_LUT).shape == (3, 14, 8)
    assert np.asarray(tab.GT_AOD_LOOKUP).shape == (8, 4, 3)
    # band 10 (0.4415-0.625 um) brackets 550 nm: the rural raod_lut there is ~1 at every RH
    assert np.allclose(np.asarray(tab.RAOD_LUT)[0, 9], 1.0, atol=0.04)


def test_lagrange_is_wrfs_window_and_reproduces_table_nodes():
    lut = np.asarray(A3._SSA)
    rh = jnp.asarray(np.asarray(A3._RHS), jnp.float32)  # exactly on the nodes
    got = np.asarray(A3._lagrange(rh, lut))
    np.testing.assert_allclose(got, lut.T, rtol=2e-6, atol=2e-7)
    # between nodes the window is ii-3 .. ii+2 (up to six points): a cubic in RH is reproduced exactly
    x = np.linspace(1.0, 98.5, 41).astype(np.float32)
    poly = lambda r: 0.5 + 1e-3 * r - 2e-5 * r**2 + 1e-7 * r**3  # noqa: E731
    cubic = np.tile(poly(np.asarray(A3._RHS, np.float64)), (14, 1))
    np.testing.assert_allclose(np.asarray(A3._lagrange(jnp.asarray(x), cubic))[:, 0], poly(x.astype(np.float64)),
                               rtol=0, atol=2e-5)


def test_gt_aod_floor_and_species_monotonic():
    p = jnp.full((3, 4), 90000.0, jnp.float32)
    t = jnp.full((3, 4), 285.0, jnp.float32)
    qv = jnp.asarray([[1e-4, 3e-3, 7e-3, 9.5e-3]] * 3, jnp.float32)  # RH low .. near saturation
    dz = jnp.full((3, 4), 200.0, jnp.float32)
    nwfa = jnp.asarray([[0.0] * 4, [1e8] * 4, [1e9] * 4], jnp.float32)
    nifa = jnp.asarray([[1e4] * 4, [1e6] * 4, [1e6] * 4], jnp.float32)
    taod = np.asarray(A3.wrf_gt_aod(p, t, qv, nwfa, nifa, dz))
    assert np.isfinite(taod).all() and (taod >= np.float32(1e-3)).all()
    assert (taod[0] == np.float32(1e-3)).all()  # clean column: WRF's per-layer floor
    assert (np.diff(taod[1:], axis=0) > 0).all() and (np.diff(taod[2]) > 0).all()  # more aerosol, more humid -> more AOD


def test_sw_state_takes_band_optics_exclusively():
    from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState

    z = jnp.ones((2, 5), jnp.float32)
    kw = dict(T=z * 280, p=z * 8e4, qv=z * 1e-3, qc=z * 0, qi=z * 0, qs=z * 0, qg=z * 0, cloud_fraction=z * 0,
              surface_albedo=jnp.ones(2), coszen=jnp.ones(2), dz=z * 100, rho=z)
    st = RRTMGSWColumnState(**kw, aerosol_bands=jnp.zeros((2, 5, 14, 3), jnp.float32))
    assert st.aerosol_bands.shape == (2, 5, 14, 3)
    assert len(jax.tree.leaves(st)) == len(jax.tree.leaves(RRTMGSWColumnState(**kw))) + 1
    with pytest.raises(ValueError):
        RRTMGSWColumnState(**kw, aerosol_bands=jnp.zeros((2, 5, 14), jnp.float32))
    with pytest.raises(ValueError):
        RRTMGSWColumnState(**kw, aerosol_bands=jnp.zeros((2, 5, 14, 3)), aerosol_ecaer=jnp.zeros((2, 5, 6)))


def test_aot_baked_table_key_names_the_aer3_tables():
    from dataclasses import replace

    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    from test_tegen_wiring import _namelist

    off = _namelist(ra_sw_physics=4, ra_lw_physics=4, mp_physics=28)
    assert baked_table_hash(None, off) != baked_table_hash(None, replace(off, aer_opt=3))


def test_history_writes_held_taod5502d_only_under_aer_opt3():
    from gpuwrf.io.producer_history import inactive_names

    lookup = lambda nl, key, default=None: nl.get(key, default) if isinstance(nl, dict) else default  # noqa: E731
    assert "TAOD5502D" in inactive_names({"mp_physics": 28, "aer_opt": 0}, lookup)
    assert "TAOD5502D" not in inactive_names({"mp_physics": 28, "aer_opt": 3}, lookup)
    src = (ROOT / "src/gpuwrf/integration/nested_pipeline.py").read_text()
    assert re.search(r"TAOD5502D=carry\.taod5502d", src)


@pytest.fixture
def carry_with_taod():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry

    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.asarray(np.arange(np.prod(shape)).reshape(shape) / 991 + 1,
                                       dtype=DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid).items()})
    taod = jnp.asarray(np.linspace(0.044, 0.9, state.theta[0].size, dtype=np.float32).reshape(state.theta.shape[1:]))
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=54, acoustic_substeps=4)
    return grid, nl, initial_operational_carry(state).replace(taod5502d=taod)


def test_restart_roundtrips_taod5502d(carry_with_taod, tmp_path):
    from gpuwrf.io.restart import read_restart, write_restart
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry

    grid, namelist, carry = carry_with_taod
    assert len(jax.tree.leaves(carry)) == len(jax.tree.leaves(carry.replace(taod5502d=None))) + 1
    write_restart(carry, namelist, grid, 3, tmp_path / "c.pkl")
    got = read_restart(tmp_path / "c.pkl")[0].taod5502d
    assert np.asarray(got).tobytes() == np.asarray(carry.taod5502d).tobytes()
    write_wrfrst_carry(carry, grid, namelist, tmp_path / "c.nc", valid_time="2026-07-25_18:00:54",
                       run_start="2026-07-25_18:00:00", step_index=1)
    got = read_wrfrst_carry(tmp_path / "c.nc", namelist=namelist)[0].taod5502d
    assert np.asarray(got).dtype == np.float32 and np.asarray(got).tobytes() == np.asarray(carry.taod5502d).tobytes()


# ---- product call path on the real Swiss case with mp28 ------------------------------------------------------

@pytest.fixture(scope="module")
def swiss_mp28(tmp_path_factory):
    if not (SWISS / "wrfinput_d01").is_file():
        pytest.skip("Swiss RD11 case unavailable")
    from test_tegen_wiring import _load

    tmp = tmp_path_factory.mktemp("aer3_swiss")
    case = tmp / "case"
    case.mkdir()
    for f in SWISS.iterdir():
        if f.name != "namelist.input" and not f.name.startswith(("wrfout_", "rsl.", "wrfrst_")):
            (case / f.name).symlink_to(f)
    text = (SWISS / "namelist.input").read_text()
    text, n = re.subn(r"^(\s*mp_physics\s*=\s*)8,", r"\g<1>28,", text, flags=re.M)
    assert n == 1
    (case / "namelist.input").write_text(text.replace("&physics\n", "&physics\n aer_opt = 3,\n", 1))
    return _load(case, tmp, ("d01",))


def test_mp28_case_binds_aer_opt3_and_seeds_a_zero_taod(swiss_mp28):
    bundles, carries, _ = swiss_mp28
    ns, carry = bundles["d01"].namelist, carries["d01"]
    assert int(ns.aer_opt) == 3 and int(ns.mp_physics) == 28 and ns.tegen_aerosol is None
    assert carry.aerod is None and carry.taod5502d is not None
    assert carry.taod5502d.dtype == jnp.float32 and carry.taod5502d.shape == carry.state.theta.shape[1:]
    assert not np.asarray(carry.taod5502d).any()  # WRF allocation zero at the t=0 history
    assert carry.state.nwfa is not None and float(np.asarray(carry.state.nwfa).min()) > 0.0


def test_coupler_feeds_wrf_aer3_band_optics_on_the_radiation_columns(swiss_mp28):
    from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs, _to_columns
    from gpuwrf.runtime import operational_mode as om

    bundles, carries, _ = swiss_mp28
    ns, carry = bundles["d01"].namelist, carries["d01"]
    kw = dict(time_utc=ns.time_utc, lead_seconds=0.0, clock_base=om._rad_clock_base(om.build_clock_base(ns)),
              radiation_static=ns.radiation_static, land_state=carry.noahmp_land)
    sw = _rrtmg_column_inputs(carry.state, ns.grid, thompson_aerosol=True, **kw)[0]
    plain = _rrtmg_column_inputs(carry.state, ns.grid, **kw)[0]
    assert plain.aerosol_bands is None and sw.aerosol_bands.shape == sw.p.shape + (14, 3)
    tau, ssa, asy, taod = A3.wrf_aer_opt3_band_optics(sw.p, sw.T, sw.qv, _to_columns(carry.state.nwfa),
                                                      _to_columns(carry.state.nifa), sw.dz)
    np.testing.assert_array_equal(np.asarray(sw.aerosol_bands), np.asarray(jnp.stack((tau, ssa, asy), -1)))
    assert float(np.asarray(taod).sum(-1).min()) >= 0.044 - 1e-6  # 44 floored layers at least
    with pytest.raises(ValueError):
        _rrtmg_column_inputs(carry.state.replace(nwfa=None), ns.grid, thompson_aerosol=True, **kw)


def test_driver_passes_aer3_and_holds_taod5502d(swiss_mp28, monkeypatch):
    from gpuwrf.coupling.physics_couplers import rrtmg_thompson_taod5502d
    from gpuwrf.runtime import operational_mode as om
    from test_tegen_wiring import _refresh, _spy

    bundles, carries, _ = swiss_mp28
    ns, carry = bundles["d01"].namelist, carries["d01"]
    calls = _spy(monkeypatch, carry.radiation_diagnostics)
    held = _refresh(carry, ns, False)
    assert not calls and held.taod5502d is carry.taod5502d
    new = _refresh(carry, ns, True)
    assert calls[-1]["thompson_aerosol"] is True and "tegen_aerosol" not in calls[-1]
    want = rrtmg_thompson_taod5502d(carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0,
                                    clock_base=om._rad_clock_base(om.build_clock_base(ns)),
                                    radiation_static=ns.radiation_static, land_state=carry.noahmp_land)
    np.testing.assert_array_equal(np.asarray(new.taod5502d), np.asarray(want))
    assert float(np.asarray(want).min()) >= 0.044 - 1e-6


def test_writer_taod5502d_is_one_extra_field_and_never_gates_the_radar_block(tmp_path, monkeypatch):
    """The held TAOD5502D (aer_opt=3) is exactly one extra history diagnostic handed to prepare_wrfout_payload:
    with TAOD and no radar the diagnostics differ from the no-TAOD call by TAOD5502D only (a TAOD hunk that
    swallowed the radar block would add REFD_MAX), and a radar-only mp=40 call without TAOD still gets the radar
    block's MSKF_REFL_10CM / REFD_MAX (cr-w41-aerosol 18:24Z)."""
    import sys
    from types import SimpleNamespace
    from gpuwrf.integration import nested_pipeline as pipeline

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from test_initial_history_wrf_t0 import _writer

    seen = []
    original = pipeline.prepare_wrfout_payload

    def spy(*args, **kwargs):
        seen.append(dict(kwargs.get("diagnostics") or {}))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "prepare_wrfout_payload", spy)

    def write(tag, *, taod=None, radar=None, mp40=False):
        case = tmp_path / tag
        case.mkdir()
        writer, state = _writer(case, lambda *a, **kw: {})
        if mp40:
            nl = writer.bundles["d01"].namelist
            writer.bundles["d01"].namelist = {**nl, "mp_physics": 40, "do_radar_ref": 1, "physics": {"mp_physics": 40}}
        held = None if taod is None else np.full(state.t_skin.shape, taod, np.float32)
        refl = None if radar is None else np.full(state.theta.shape, radar, np.float32)
        writer("d01", 200, SimpleNamespace(state=state, radar_reflectivity=refl, nwp_diagnostics=None, taod5502d=held))
        return seen[-1], held, refl

    off, _, _ = write("off")
    on, held, _ = write("on", taod=0.0442)
    assert set(on) - set(off) == {"TAOD5502D"} and not set(off) - set(on)
    assert np.asarray(on["TAOD5502D"]).tobytes() == held.tobytes()
    radar, _, refl = write("radar", radar=12.5, mp40=True)
    assert "TAOD5502D" not in radar and {"REFL_10CM", "MSKF_REFL_10CM", "REFD_MAX"} <= set(radar)
    assert np.asarray(radar["MSKF_REFL_10CM"]).tobytes() == refl.tobytes()
