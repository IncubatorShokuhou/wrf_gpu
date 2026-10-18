"""w41-aerosol AE05: aer_opt=1 wiring -- admission, released treedef, held/forced ``aerod``, restart (CPU).

WRF: ``aerosol_in`` + ``aer_time_int``/``aer_p_int`` run on domain 1 only (module_physics_init.F:2214,
module_radiation_driver.F:1827); the 4-D ``aerod`` (Registry df=(p2c)) reaches each nest by the SINT force-down
after every parent step; RRTMG_SWRAD reads it (iaer=6).  The unit chain and the SW solver are bound to pristine WRF
in test_tegen_units.py / test_tegen_sw_oracle.py; these tests bind the product seams that carry the field there.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.aerosol_options import aerosol_refusals
from gpuwrf.io.scheme_catalog import SupportStatus, classify_control
from gpuwrf.nesting.interp import build_sint_weights
from gpuwrf.nesting.nest_aerosol import force_child_aerod, force_child_carry_aerod
from gpuwrf.nesting.nest_o3 import force_child_o3rad
from gpuwrf.physics.wrf_tegen_aerosol import TegenAerosolStatic, build_tegen_aerosol_static, wrf_tegen_aerod

SWISS = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")


def _nml(aer_opt, sw=(4, 4), mp=(8, 8)):
    return {"physics": {"aer_opt": aer_opt, "ra_sw_physics": list(sw), "mp_physics": list(mp)}}


# ---- admission -------------------------------------------------------------------------------------------------

def test_admission_refuses_unported_options_and_non_rrtmg_sw():
    assert aerosol_refusals(_nml(0, (1, 1)), ("d01", "d02")) == []
    assert aerosol_refusals(_nml(1), ("d01", "d02")) == []
    (err,) = aerosol_refusals(_nml(2), ("d01",))
    assert "aer_opt=2" in err
    errs = aerosol_refusals(_nml(1, (4, 2)), ("d01", "d02"))  # d02 Goddard SW would silently drop the aerosol
    assert len(errs) == 1 and errs[0].startswith("d02")
    # aer_opt=3: WRF computes the optics only with QNWFA (mp28); the port's mp28 is single-domain
    assert aerosol_refusals(_nml(3, (4,), (28,)), ("d01",)) == []
    (err,) = aerosol_refusals(_nml(3, (4,), (8,)), ("d01",))
    assert "mp_physics=28" in err
    assert [e for e in aerosol_refusals(_nml(3, (4, 4), (28, 28)), ("d01", "d02")) if "single domains" in e]
    assert aerosol_refusals(_nml(3, (1,), (28,)), ("d01",))
    for opt in (1, 3):
        assert classify_control("aer_opt", opt).status is SupportStatus.IMPLEMENTED
    assert classify_control("aer_opt", 2).status is SupportStatus.RECOGNIZED_FAIL_CLOSED


def test_namelist_honour_report_routes_through_the_aerosol_admission():
    from gpuwrf.io.namelist_binding import PHYSICS_KNOBS, namelist_honour_report

    assert PHYSICS_KNOBS["aer_opt"] is int
    ok, _ = namelist_honour_report(_nml(1), ("d01", "d02"))
    assert not [e for e in ok if "aer_opt" in e]
    bad, _ = namelist_honour_report(_nml(2), ("d01", "d02"))
    assert [e for e in bad if "aer_opt=2" in e]


# ---- released program: the namelist treedef ------------------------------------------------------------------

def _namelist(**kw):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    grid = GridSpec.canary_3km_template()
    return OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=54, acoustic_substeps=4, **kw)


def _static(ny=3, nx=4):
    lat, lon = np.meshgrid(np.linspace(27.0, 29.0, ny, dtype=np.float32), np.linspace(-18.5, -15.0, nx, dtype=np.float32),
                           indexing="ij")
    return build_tegen_aerosol_static(lat, lon)


def test_aer_opt0_keeps_the_released_namelist_treedef():
    base = _namelist()
    leaves, tree = jax.tree.flatten(base)
    explicit = jax.tree.flatten(_namelist(aer_opt=0))
    assert explicit[1] == tree and len(explicit[0]) == len(leaves)
    assert not any(isinstance(a, tuple) and a and a[0] == "aer_opt" for a in base.tree_flatten()[1])


def test_aer_opt1_round_trips_with_a_traced_static_and_splits_the_treedef():
    static = _static()
    on = _namelist(aer_opt=1, tegen_aerosol=static)
    leaves, tree = jax.tree.flatten(on)
    assert tree != jax.tree.flatten(_namelist())[1]
    back = jax.tree.unflatten(tree, leaves)
    assert int(back.aer_opt) == 1 and isinstance(back.tegen_aerosol, TegenAerosolStatic)
    assert np.array_equal(np.asarray(back.tegen_aerosol.aerodm), np.asarray(static.aerodm))
    # the static is a traced child (one leaf per array), never baked into the aux
    assert len(leaves) == len(jax.tree.leaves(_namelist())) + 2
    # a nest: aer_opt=1 without the static (it radiates with the forced aerod)
    nest = jax.tree.unflatten(*jax.tree.flatten(_namelist(aer_opt=1))[::-1])
    assert int(nest.aer_opt) == 1 and nest.tegen_aerosol is None


def test_aot_baked_table_key_names_the_tegen_asset_only_on_aer_opt1():
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash

    off = _namelist(ra_sw_physics=4, ra_lw_physics=4)
    on = replace(off, aer_opt=1, tegen_aerosol=_static())
    assert baked_table_hash(None, off) != baked_table_hash(None, on)
    assert baked_table_hash(None, off) == baked_table_hash(None, replace(off, aer_opt=0))


# ---- nest force-down -----------------------------------------------------------------------------------------

def _weights(py=12, px=14, cy=19, cx=22):
    return build_sint_weights(parent_grid_ratio=3, i_parent_start=3, j_parent_start=2, parent_ny=py, parent_nx=px,
                              child_ny=cy, child_nx=cx)


def _parent_aerod(nz=5, py=12, px=14, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(1e-7, 4e-5, (6, nz, py, px)).astype(np.float32))


def test_force_child_aerod_is_wrf_sint_per_species_and_level():
    w, parent = _weights(), _parent_aerod()
    got = np.asarray(force_child_aerod(parent, w, parent_grid_ratio=3))
    assert got.dtype == np.float32 and got.shape == (6, 5, 19, 22) and np.isfinite(got).all()
    for s in range(6):  # each species = the o3rad force-down (bound to pristine interp_fcn SINT, RE03 G1); op-by-op
        np.testing.assert_array_equal(got[s], np.asarray(force_child_o3rad(parent[s], w, parent_grid_ratio=3)))
    # the jitted product force: XLA:CPU whole-jit fusion rounding only (E186), a few ulp of the field scale
    # (SINT's signed weights cancel, so small outputs can differ by many of their own ulps)
    jitted = np.asarray(jax.jit(lambda f: force_child_aerod(f, w, parent_grid_ratio=3))(parent))
    np.testing.assert_allclose(jitted, got, rtol=0, atol=4 * float(np.spacing(np.abs(np.asarray(parent)).max())))
    # mutant: a species-major/level-major reshape mix-up lands species on the wrong slots
    wrong = np.asarray(force_child_o3rad(parent.transpose(1, 0, 2, 3).reshape(30, 12, 14), w, parent_grid_ratio=3))
    assert np.abs(wrong.reshape(6, 5, 19, 22) - got).max() > 1e-6


def test_carry_force_maps_cases_and_passes_released_carries_through():
    @dataclass(frozen=True)
    class Carry:
        aerod: object

        def replace(self, **kw):
            return replace(self, **kw)

    w = SimpleNamespace(mass=_weights())
    parents = jnp.stack([_parent_aerod(seed=1), _parent_aerod(seed=2)])
    child0 = jnp.zeros((2, 6, 5, 19, 22), jnp.float32)
    got = force_child_carry_aerod(Carry(child0), Carry(parents), w, parent_grid_ratio=3, batched=True).aerod
    for b in range(2):
        one = force_child_carry_aerod(Carry(child0[b]), Carry(parents[b]), w, parent_grid_ratio=3).aerod
        np.testing.assert_array_max_ulp(np.asarray(got[b]), np.asarray(one), maxulp=4)
    plain = Carry(None)
    assert force_child_carry_aerod(plain, Carry(parents), w, parent_grid_ratio=3) is plain
    assert force_child_carry_aerod(Carry(child0), Carry(None), w, parent_grid_ratio=3).aerod is child0


# ---- restart ---------------------------------------------------------------------------------------------------

@pytest.fixture
def carry_with_aerod():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.runtime.operational_state import initial_operational_carry

    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.asarray(np.arange(np.prod(shape)).reshape(shape) / 991 + 1,
                                       dtype=DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid).items()})
    aerod = jnp.asarray(np.linspace(1e-9, 3e-5, 6 * state.theta.size, dtype=np.float32).reshape((6,) + state.theta.shape))
    return grid, _namelist(), initial_operational_carry(state).replace(aerod=aerod)


def test_released_carry_treedef_is_unchanged_without_aerod(carry_with_aerod):
    _grid, _nl, carry = carry_with_aerod
    plain = carry.replace(aerod=None)
    assert len(jax.tree.leaves(carry)) == len(jax.tree.leaves(plain)) + 1
    assert jax.tree.structure(plain) != jax.tree.structure(carry)
    assert "aerod" in [f for f in type(plain).__dataclass_fields__]


def test_pickle_restart_roundtrips_aerod_bitwise(carry_with_aerod, tmp_path):
    from gpuwrf.io.restart import _CARRY_FIELD_ORDER, read_restart, write_restart

    grid, namelist, carry = carry_with_aerod
    assert "aerod" in _CARRY_FIELD_ORDER
    path = tmp_path / "carry.pkl"
    write_restart(carry, namelist, grid, 7, path)
    restored = read_restart(path)[0]
    got = np.asarray(restored.aerod)
    assert got.dtype == np.float32 and got.tobytes() == np.asarray(carry.aerod).tobytes()
    assert jax.tree.structure(restored) == jax.tree.structure(carry)
    write_restart(carry.replace(aerod=None), namelist, grid, 7, path)
    assert read_restart(path)[0].aerod is None


def test_netcdf_carry_writer_roundtrips_populated_aerod(carry_with_aerod, tmp_path):
    from gpuwrf.io.wrfrst_netcdf import OPTIONAL_CARRY_FIELDS, read_wrfrst_carry, write_wrfrst_carry

    grid, namelist, carry = carry_with_aerod
    assert "aerod" in OPTIONAL_CARRY_FIELDS
    path = tmp_path / "carry.nc"
    write_wrfrst_carry(carry, grid, namelist, path, valid_time="2026-07-25_18:00:54",
                       run_start="2026-07-25_18:00:00", step_index=1)
    got = np.asarray(read_wrfrst_carry(path, namelist=namelist)[0].aerod)
    assert got.dtype == np.float32 and got.shape == carry.aerod.shape
    assert got.tobytes() == np.asarray(carry.aerod).tobytes()
    path = tmp_path / "absent.nc"
    write_wrfrst_carry(carry.replace(aerod=None), grid, namelist, path,
                       valid_time="2026-07-25_18:00:54", run_start="2026-07-25_18:00:00", step_index=1)
    assert read_wrfrst_carry(path, namelist=namelist)[0].aerod is None


# ---- product call path on real cases -------------------------------------------------------------------------

def _aer_case(src: Path, dst: Path) -> Path:
    """Symlinked copy of a WRF run dir whose namelist.input selects aer_opt = 1 (nothing else changes)."""

    dst.mkdir()
    for f in src.iterdir():
        if f.name != "namelist.input" and not f.name.startswith(("wrfout_", "rsl.", "wrfrst_")):
            (dst / f.name).symlink_to(f)
    text = (src / "namelist.input").read_text()
    head, sep, tail = text.partition("&physics")
    assert sep and "aer_opt" not in text
    (dst / "namelist.input").write_text(head + sep + "\n aer_opt = 1," + tail)
    return dst


def _load(case: Path, tmp: Path, names):
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.domain_tree import DomainTree

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
        cfg = pipeline.NestedPipelineConfig(case, tmp / "out", tmp / "proof", hours=3, max_dom=len(names))
        hierarchy, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, names)
        tree = DomainTree.from_domains(hierarchy, bundles) if len(names) > 1 else None
    return bundles, carries, tree


def _spy(monkeypatch, held_diag):
    from gpuwrf.runtime import operational_mode as om

    calls = []

    def fake(state, grid=None, **kw):
        calls.append(kw)
        return jnp.zeros(state.theta.shape, jnp.float32), held_diag

    monkeypatch.setattr(om, "rrtmg_theta_tendency", fake)
    return calls


def _refresh(carry, namelist, run_radiation):
    from gpuwrf.runtime import operational_mode as om

    return om._refresh_rrtmg_driver(carry, namelist, 0.0, run_radiation, om.build_clock_base(namelist))


def _columns(carry, namelist, static):
    from gpuwrf.coupling.physics_couplers import rrtmg_aerosol_columns
    from gpuwrf.runtime import operational_mode as om

    return rrtmg_aerosol_columns(carry.state, namelist.grid, tegen_aerosol=static, time_utc=namelist.time_utc,
                                 lead_seconds=0.0, clock_base=om._rad_clock_base(om.build_clock_base(namelist)),
                                 radiation_static=namelist.radiation_static, land_state=carry.noahmp_land)


@pytest.fixture(scope="module")
def swiss(tmp_path_factory):
    if not (SWISS / "wrfinput_d01").is_file():
        pytest.skip("Swiss RD11 case unavailable")
    tmp = tmp_path_factory.mktemp("aer_swiss")
    return _load(_aer_case(SWISS, tmp / "case"), tmp, ("d01",))


def test_swiss_single_domain_binds_the_wrf_aerosol_in_static_and_radiates_inline(swiss, monkeypatch):
    from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs
    from gpuwrf.physics.wrf_tegen_aerosol import wrf_aerosol_in
    from gpuwrf.runtime import operational_mode as om

    bundles, carries, _ = swiss
    ns, carry = bundles["d01"].namelist, carries["d01"]
    assert int(ns.aer_opt) == 1 and carry.aerod is None  # single domain: no held leaf
    rs = ns.radiation_static
    expect = wrf_aerosol_in(np.asarray(rs.xlat_deg, np.float32), np.asarray(rs.xlong_deg, np.float32))
    assert np.array_equal(np.asarray(ns.tegen_aerosol.aerodm), expect)
    calls = _spy(monkeypatch, carry.radiation_diagnostics)
    _refresh(carry, ns, True)
    assert calls[-1]["tegen_aerosol"] is ns.tegen_aerosol and "aerosol_ecaer_override" not in calls[-1]
    # the coupler's column-input path: same p_hyd/p_hyd_w and REAL julian as the SW state it feeds
    sw, *_ = _rrtmg_column_inputs(carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0,
                                  clock_base=om._rad_clock_base(om.build_clock_base(ns)),
                                  radiation_static=rs, land_state=carry.noahmp_land, tegen_aerosol=ns.tegen_aerosol)
    assert sw.aerosol_ecaer.dtype == jnp.float32 and sw.aerosol_ecaer.shape == sw.p.shape + (6,)
    tot = np.asarray(sw.aerosol_ecaer).sum(axis=(-2, -1))
    assert np.isfinite(tot).all() and 0.01 < np.median(tot) < 2.0  # column AOD of a Tegen climatology
    plain = _rrtmg_column_inputs(carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0,
                                 clock_base=om._rad_clock_base(om.build_clock_base(ns)),
                                 radiation_static=rs, land_state=carry.noahmp_land)[0]
    assert plain.aerosol_ecaer is None  # deletion-sensitive: no static, no aerosol
    # an explicit override reaches the SW state unchanged (the nested path)
    marker = jnp.full(sw.aerosol_ecaer.shape, 2.5e-6, jnp.float32)
    over = _rrtmg_column_inputs(carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0,
                                clock_base=om._rad_clock_base(om.build_clock_base(ns)), radiation_static=rs,
                                land_state=carry.noahmp_land, aerosol_ecaer_override=marker)[0]
    assert np.array_equal(np.asarray(over.aerosol_ecaer), np.asarray(marker))


def test_swiss_single_domain_sw_heating_sees_the_aerosol_and_lw_does_not(swiss):
    from gpuwrf.coupling.physics_couplers import rrtmg_radiation_diagnostics
    from gpuwrf.runtime import operational_mode as om

    bundles, carries, _ = swiss
    ns, carry = bundles["d01"].namelist, carries["d01"]
    kw = dict(time_utc=ns.time_utc, lead_seconds=11 * 3600.0, clock_base=om._rad_clock_base(om.build_clock_base(ns)),
              radiation_static=ns.radiation_static, land_state=carry.noahmp_land)
    on = rrtmg_radiation_diagnostics(carry.state, ns.grid, tegen_aerosol=ns.tegen_aerosol, **kw)
    off = rrtmg_radiation_diagnostics(carry.state, ns.grid, **kw)
    for k in ("swdown", "swnorm", "swup", "sw_toa_up"):  # daylight columns: the aerosol dims/brightens SW
        a, b = np.asarray(getattr(on, k)), np.asarray(getattr(off, k))
        assert np.isfinite(a).all() and np.abs(a - b).max() > 1.0, k
    assert np.median(np.asarray(off.swdown) - np.asarray(on.swdown)) > 0.0  # net: less sunlight at the ground
    for k in ("glw", "glw_up", "lw_toa_down", "lw_toa_up", "coszen", "surface_albedo"):  # LW has no aerosol
        assert np.array_equal(np.asarray(getattr(on, k)), np.asarray(getattr(off, k))), k


@pytest.fixture(scope="module")
def nested(tmp_path_factory):
    if not (WN3 / "wrfinput_d02").is_file():
        pytest.skip("original WN3 0227 case inputs not mounted")
    tmp = tmp_path_factory.mktemp("aer_wn3")
    return _load(_aer_case(WN3, tmp / "case"), tmp, ("d01", "d02"))


def test_nested_carries_seed_a_held_real_aerod_and_only_the_root_has_the_static(nested):
    bundles, carries, _tree = nested
    for name, carry in carries.items():
        assert int(bundles[name].namelist.aer_opt) == 1
        assert carry.aerod is not None and carry.aerod.dtype == jnp.float32
        assert carry.aerod.shape == (6,) + carry.state.theta.shape and not np.asarray(carry.aerod).any()
    assert bundles["d01"].namelist.tegen_aerosol is not None and bundles["d02"].namelist.tegen_aerosol is None


def test_nested_root_refreshes_aerod_only_at_its_radiation_calls(nested, monkeypatch):
    bundles, carries, _tree = nested
    ns, root = bundles["d01"].namelist, carries["d01"]
    calls = _spy(monkeypatch, root.radiation_diagnostics)
    held = _refresh(root, ns, False)
    assert not calls and held.aerod is root.aerod
    new = _refresh(root, ns, True)
    own = np.asarray(_columns(root, ns, ns.tegen_aerosol))
    assert np.array_equal(np.asarray(new.aerod), np.transpose(own, (3, 2, 0, 1)))
    assert np.array_equal(np.asarray(calls[-1]["aerosol_ecaer_override"]), own)
    assert own.min() >= 0.0 and own.sum(axis=(-2, -1)).min() > 0.0


def test_nested_forcedown_sets_child_aerod_and_the_child_radiates_with_it(nested, monkeypatch):
    from gpuwrf.nesting import nest_aerosol
    from gpuwrf.runtime.domain_tree import _operational_force

    bundles, carries, tree = nested
    calls = _spy(monkeypatch, carries["d01"].radiation_diagnostics)
    root = _refresh(carries["d01"], bundles["d01"].namelist, True)
    (edge,) = tree.children("d01")
    forced = _operational_force(edge, root, carries["d02"])
    ratio = int(edge.parent_grid_ratio)
    got = np.asarray(forced.aerod)
    assert np.array_equal(got, np.asarray(nest_aerosol._compiled_force(edge.weights.mass, ratio, False)(root.aerod)))
    assert got.sum(axis=(0, 1)).min() > 0.0
    # jitted product force vs the eager SINT reference: whole-jit rounding of SINT's signed weights, judged on the
    # field scale like test_force_child_aerod_is_wrf_sint_per_species_and_level (small outputs differ by many own ulps)
    eager = np.asarray(force_child_aerod(root.aerod, edge.weights.mass, parent_grid_ratio=ratio))
    scale = float(np.spacing(np.abs(np.asarray(root.aerod)).max()))
    np.testing.assert_allclose(got, eager, rtol=0, atol=4 * scale,
                               err_msg=f"max|jit-eager| {np.abs(got - eager).max():.3e}, 4*spacing(max parent) {4 * scale:.3e}")
    plain = _operational_force(edge, root.replace(aerod=None), carries["d02"])
    for name in forced.__dataclass_fields__:
        if name != "aerod":
            a, b = jax.tree.leaves(getattr(forced, name)), jax.tree.leaves(getattr(plain, name))
            assert len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)), name
    ns2 = bundles["d02"].namelist
    child = _refresh(forced, ns2, True)
    assert child.aerod is forced.aerod  # a nest never recomputes aerod
    used = np.asarray(calls[-1]["aerosol_ecaer_override"])
    assert np.array_equal(used, np.transpose(np.asarray(forced.aerod), (2, 3, 1, 0)))
    # the nest's own aerosol_in/aer_p_int (what WRF does NOT do) is a different field
    own = np.asarray(_columns(forced, ns2, build_tegen_aerosol_static(np.asarray(ns2.radiation_static.xlat_deg),
                                                                       np.asarray(ns2.radiation_static.xlong_deg))))
    assert np.abs(own - used).max() / used.max() > 1e-4


def test_nested_fused_cascade_forces_child_aerod_like_the_eager_force(nested, monkeypatch):
    from gpuwrf.runtime import domain_tree as dt

    bundles, carries, tree = nested
    ns, ns2 = bundles["d01"].namelist, bundles["d02"].namelist
    (edge,) = tree.children("d01")
    parent = carries["d01"].replace(aerod=jnp.transpose(_columns(carries["d01"], ns, ns.tegen_aerosol), (3, 2, 0, 1)))
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "0")
    monkeypatch.setattr(dt, "_advance_chunk", lambda carry, *_a, **_k: carry)
    program = dt._build_fused_cascade_program(
        parent_name="d01", parent_namelist=ns, parent_cadence=int(ns.radiation_cadence_steps),
        child_names=("d02",), child_namelists=(ns2,), child_weights=(edge.weights,),
        child_bdy_widths=(int(tree.domains["d02"].state.u_bdy.shape[2]),),
        child_ratios=(int(edge.parent_grid_ratio),), child_cadences=(int(ns2.radiation_cadence_steps),),
    )
    _parent_new, (child_new,) = program(parent, (carries["d02"],), 1, (1,))
    eager = np.asarray(dt._operational_force(edge, parent, carries["d02"]).aerod)
    got = np.asarray(child_new.aerod)
    assert got.dtype == np.float32 and np.isfinite(got).all() and got.sum(axis=(0, 1)).min() > 0.0
    np.testing.assert_array_max_ulp(got, eager, maxulp=4)


def test_swiss_tegen_aerod_is_layer_aod_from_the_table(swiss):
    """Sanity of the bound field: per-layer AOD = aerodt(p) x dp, summed close to the climatology column AOD."""

    bundles, _carries, _ = swiss
    static = bundles["d01"].namelist.tegen_aerosol
    pf = jnp.linspace(101000.0, 5000.0, 45, dtype=jnp.float32)[None, None, :] * jnp.ones((2, 2, 1), jnp.float32)
    p = 0.5 * (pf[..., :-1] + pf[..., 1:])
    sub = TegenAerosolStatic(static.aerodm[:2, :2], static.pina_hpa)
    ad = np.asarray(wrf_tegen_aerod(sub, p, pf, 57.0))
    assert ad.shape == (2, 2, 44, 6) and (ad >= 0).all() and np.isfinite(ad).all()
    assert 0.01 < ad.sum(axis=(-2, -1)).min() and ad.sum(axis=(-2, -1)).max() < 2.0
