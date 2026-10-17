"""Host seed ownership under real CPU PjRt donation; no physics/GPU claims."""
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as om
from test_v013_operational_smoke import _base_state, _grid, _namelist


def _retained_arrays(value):
    """Walk holders independently of their JAX static/dynamic registration."""
    if isinstance(value, jax.Array):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _retained_arrays(getattr(value, field.name))
    elif isinstance(value, dict):
        for child in value.values():
            yield from _retained_arrays(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _retained_arrays(child)
    else:
        for leaf in jax.tree.leaves(value):
            if leaf is not value:
                yield from _retained_arrays(leaf)


def _assert_private(carry, holders):
    arrays = list(_retained_arrays(holders))
    identities = {id(x) for x in arrays}
    pointers = {x.unsafe_buffer_pointer() for x in arrays}
    for path, leaf in jax.tree_util.tree_leaves_with_path(carry):
        if isinstance(leaf, jax.Array):
            assert id(leaf) not in identities, f"retained seed identity at {path}"
            assert leaf.unsafe_buffer_pointer() not in pointers, f"retained seed buffer at {path}"


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr(om, "_CARRY_DONATE", True)
    for name in ("GPUWRF_DYN_CARRY_FP32", "GPUWRF_DYN_RK_FP32", "GPUWRF_CARRY_REAL_ALL"):
        monkeypatch.setenv(name, "0")
    grid = _grid(nz=4, ny=2, nx=3)
    state = _base_state(grid)
    nml = replace(_namelist(grid), run_physics=False, cu_physics=0, use_noahmp=False,
                  ra_sw_physics=0, ra_lw_physics=0, force_fp64=False)
    def radiation(*args, **kwargs):
        rad = tuple(jnp.full((2, 3), x, jnp.float32) for x in (100., 300., .5))
        return (rad, None) if kwargs.get("_with_diagnostics") else rad
    monkeypatch.setattr(om, "noahmp_initial_rad", radiation)
    return state, nml


def _seeded_case(kind, scene, tmp_path):
    state, nml = scene
    def plane(value=1.):
        return jnp.full((2, 3), value, jnp.float32)
    if kind == "ruc":
        from gpuwrf.io.ruc_land_init import load_ruc_land
        from gpuwrf.coupling.ruc_surface_hook import RucRadiation
        from test_o2_ruccli import _input
        _input(tmp_path / "wrfinput_d01")
        static, land = load_ruc_land(tmp_path, "d01", num_soil_layers=6, dt=18.,
                                     p1=np.full((2, 3), 90000.))
        nml = replace(nml, sf_surface_physics=3, ruc_static=static, ruc_land=land,
                      ruc_rad=RucRadiation(plane(100.), plane(300.)))
        # Match the release REAL carry: the original failure depends on asarray
        # retaining XLAND rather than casting it to a newly allocated dtype.
        state = state.replace(xland=plane(), roughness_m=plane(.1),
                              mavail=plane(.5), soil_moisture=plane(.25))
    elif kind in ("noahclassic", "slab", "px"):
        from gpuwrf.coupling.noahclassic_surface_hook import NoahClassicLandState, NoahClassicRadiation
        from gpuwrf.coupling.slab_surface_hook import SlabLandState, SlabRadiation
        from gpuwrf.coupling.pleim_xiu_surface_hook import PleimXiuLandState, PleimXiuRadiation
        code, land_type, rad_type = {
            "noahclassic": (2, NoahClassicLandState, NoahClassicRadiation),
            "slab": (1, SlabLandState, SlabRadiation),
            "px": (7, PleimXiuLandState, PleimXiuRadiation),
        }[kind]
        # Actual carry classes; initialization forwards these seeds verbatim.
        land = land_type(*(jnp.full((2, 3, 5), 280., jnp.float32) if key == "tslb"
                           else jnp.full((2, 3, 4), .25, jnp.float32) if key in ("stc", "smc", "sh2o")
                           else plane(i + 1.) for i, key in enumerate(land_type._fields)))
        rad = rad_type(*(plane(i + 1.) for i in range(len(rad_type._fields))))
        nml = replace(nml, sf_surface_physics=code, **{kind + "_land": land, kind + "_rad": rad})
    elif kind == "noahmp":
        from gpuwrf.contracts.noahmp_state import NoahMPLandState
        land = NoahMPLandState(**{key: (jnp.zeros((2, 3), jnp.int32) if key == "isnow"
            else jnp.full((4, 2, 3), .25, jnp.float32) if key in ("tslb", "smois", "sh2o")
            else jnp.full((3, 2, 3), 0., jnp.float32) if key in ("tsno", "snice", "snliq")
            else jnp.full((7, 2, 3), 0., jnp.float32) if key == "zsnso"
            else plane(i + 1.)) for i, key in enumerate(NoahMPLandState.__slots__)})
        nml = replace(nml, sf_surface_physics=4, use_noahmp=True, noahmp_land=land)
    elif kind == "lake":
        from test_s1_lake_wiring import bundle_for
        nml = replace(nml, sf_lake_physics=1, lake_bundle=bundle_for((2, 3)))
    elif kind in ("grid_fdda", "spectral_fdda"):
        from gpuwrf.assimilation.fdda_grid import GridFddaInputs, GridFddaOptions
        from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs, SpectralFddaOptions
        data = {name + "_" + epoch: jnp.full((1, *shape), i + 1., jnp.float64)
                for i, (name, shape) in enumerate((("u", (4, 2, 4)), ("v", (4, 3, 3)),
                                                   ("t", (4, 2, 3)), ("q", (4, 2, 3))))
                for epoch in ("old", "new")}
        if kind == "spectral_fdda":
            data.update(ph_old=jnp.ones((1, 5, 2, 3), jnp.float64),
                        ph_new=jnp.ones((1, 5, 2, 3), jnp.float64))
            inputs = SpectralFddaInputs(**data, options=SpectralFddaOptions(interval_m=1))
        else:
            inputs = GridFddaInputs(**data, options=GridFddaOptions(interval_m=1))
        nml = replace(nml, data_assimilation=inputs)
    else:
        raise AssertionError(kind)
    return om._initial_carry_for_run(state, nml), nml


@pytest.mark.parametrize("kind", ["ruc", "noahclassic", "slab", "px", "noahmp", "lake",
                                  "grid_fdda", "spectral_fdda"])
def test_all_initialized_seeds_survive_donation(kind, scene, tmp_path):
    carry, nml = _seeded_case(kind, scene, tmp_path)
    _assert_private(carry, nml)
    retained = list(_retained_arrays(nml))
    original = [np.asarray(x).copy() for x in retained]
    # Match _advance_chunk: its existing guard owns in-carry duplicates.
    carry = om.unalias_donated_carry(carry, by_buffer=True)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(carry)]
    # Real donating executable, including every carry leaf and unmodified output.
    donate = jax.jit(lambda c: jax.tree.map(lambda x: x, c), donate_argnums=(0,))
    out = jax.block_until_ready(donate(carry))
    for leaf, expected in zip(jax.tree.leaves(out), before, strict=True):
        assert np.asarray(leaf).dtype == expected.dtype
        assert np.asarray(leaf).tobytes() == expected.tobytes()
    for leaf, expected in zip(retained, original, strict=True):
        assert not leaf.is_deleted()
        assert np.asarray(leaf).tobytes() == expected.tobytes()


def test_cpu_analogue_deletes_unprotected_static_seed():
    seed = jnp.arange(42 * 42, dtype=jnp.float32).reshape(42, 42)
    alias = jnp.asarray(seed, seed.dtype)
    assert alias is seed
    expected = np.asarray(seed).copy()
    out = jax.jit(lambda x: x, donate_argnums=(0,))(alias)
    out.block_until_ready()
    assert seed.is_deleted(), "CPU test must actually exercise donation"
    with pytest.raises(RuntimeError, match="deleted"):
        np.asarray(seed)
    assert np.asarray(out).tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ["urban_slucm", "tslist"])
def test_post_initialized_carries_do_not_alias_retained_holders(kind, scene):
    """These two seeds are added after _initial_carry_for_run by the pipeline."""
    from pathlib import Path
    state, nml = scene
    carry = om._initial_carry_for_run(state, nml)
    if kind == "urban_slucm":
        from gpuwrf.coupling.urban_slucm_hook import UrbanBundle, initial_urban_carry
        from gpuwrf.physics.urban_slucm import parse_urbparm
        table = parse_urbparm(Path(__file__).resolve().parents[1]
                             / "proofs/v040/urban_slucm/savepoints/URBPARM.TBL.default")
        bundle = UrbanBundle(table=table, isurban=13, natural=14)
        nml = replace(nml, sf_urban_physics=1, urban_bundle=bundle)
        urban = initial_urban_carry(bundle, ivgtyp=jnp.full((2, 3), 13, jnp.int32),
            tsk=state.t_skin, tslb=jnp.stack([state.t_skin-k for k in range(4)]),
            frc_urb2d=jnp.zeros((2, 3), jnp.float32))
        carry = carry.replace(urban_slucm=urban)
    else:
        from gpuwrf.timeseries.sampling import seed_buffer
        locations = [SimpleNamespace(i=1, j=1), SimpleNamespace(i=2, j=2)]
        # prepare_time_series supplies host booleans and station coordinates.
        buffer = seed_buffer(locations, 4, 2, land=[True, False])
        carry = carry.replace(timeseries=jax.device_put(buffer, state.theta.device))
    _assert_private(carry, nml)
    retained = list(_retained_arrays(nml))
    original = [np.asarray(x).copy() for x in retained]
    carry = om.unalias_donated_carry(carry, by_buffer=True)
    out = jax.block_until_ready(jax.jit(lambda c: c, donate_argnums=(0,))(carry))
    assert getattr(out, "urban_slucm" if kind == "urban_slucm" else "timeseries") is not None
    for leaf, expected in zip(retained, original, strict=True):
        assert not leaf.is_deleted()
        assert np.asarray(leaf).tobytes() == expected.tobytes()


def test_distinct_committed_alias_is_protected(monkeypatch):
    monkeypatch.setattr(om, "_CARRY_DONATE", True)
    seed = jnp.arange(12., dtype=jnp.float32)
    alias = jax.device_put(seed, jax.devices("cpu")[0])
    assert alias is not seed and alias.unsafe_buffer_pointer() == seed.unsafe_buffer_pointer()
    # A nested unregistered dataclass also exercises opaque static holders.
    from dataclasses import make_dataclass
    Holder = make_dataclass("Holder", [("value", object)])
    held = Holder({"nested": [Holder(seed)]})
    out = om._protect_initial_carry_seeds({"x": alias}, held)
    _assert_private(out, held)
    jax.jit(lambda x: x, donate_argnums=(0,))(out)["x"].block_until_ready()
    assert not seed.is_deleted() and not alias.is_deleted()


def test_no_donation_and_tracing_do_not_copy(scene, monkeypatch):
    seed = jnp.arange(6., dtype=jnp.float32)
    holder = SimpleNamespace(seed=seed)
    monkeypatch.setattr(om, "_CARRY_DONATE", False)
    assert om._protect_initial_carry_seeds(seed, holder) is seed
    monkeypatch.setattr(om, "_CARRY_DONATE", True)
    traced = jax.make_jaxpr(lambda x: om._protect_initial_carry_seeds(x, holder))(seed)
    plain = jax.make_jaxpr(lambda x: x)(seed)
    assert str(traced) == str(plain)


def test_private_seed_does_not_copy_internal_duplicates(monkeypatch):
    from dataclasses import make_dataclass
    monkeypatch.setattr(om, "_CARRY_DONATE", True)
    Holder = make_dataclass("Holder", [("value", object)])
    seed = jnp.arange(6., dtype=jnp.float32)
    carry = (seed, seed)
    # No external sharing: leave the normal advance's existing de-duplication
    # work where it was, avoiding extra initialization copies on release paths.
    assert om._protect_initial_carry_seeds(carry, Holder(jnp.zeros(4))) is carry
