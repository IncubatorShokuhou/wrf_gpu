"""WRF REAL snow options, real loader bindings, cache keys and restart continuity.

The numeric reference is compiled verbatim pristine WRF (small tracked JSON),
not a second copy of the JAX formula. Constructed columns exercise every phase
threshold, rain/snow/graupel/hail, fresh/aged/no snow and wet-bulb humidity.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.noahmp_options import (DEFAULTS, NoahMPOptions, option_refusals,
                                      read_bats_parameters, selected_options)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "proofs/u2_land/snow_oracle.json"


@pytest.fixture
def real(monkeypatch):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


def oracle():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("snf", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("compiled", [False, True])
def test_every_snow_partition_against_verbatim_wrf(real, snf, compiled):
    from gpuwrf.physics.noahmp.precip_heat import _atm_partition
    fx = oracle()
    rows = [case for case in fx["cases"] if case["inputs"][0] == snf]
    fields = fx["variables_in"]
    force = SimpleNamespace(**{key: jnp.asarray([case["inputs"][fields.index(key)] for case in rows], jnp.float32)
                               for key in fields[1:]})
    fn = lambda values: _atm_partition(SimpleNamespace(**values), snf)
    operands=vars(force)
    actual = np.stack([np.asarray(x) for x in (jax.jit(fn)(operands) if compiled else fn(operands))], -1)
    expected = np.asarray([case["output"][:6] for case in rows])
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, **fx["tolerance"])
    assert actual.dtype == np.float32


def test_bats_albedo_and_age_against_verbatim_wrf(real):
    from gpuwrf.physics.noahmp.energy_radiation import snow_age, snowalb_bats, TwoStreamParams
    fx = oracle()
    rows = fx["cases"][:9]
    params = fx["parameters"]
    # No unused radiation arrays are read by SNOW_AGE.
    p = TwoStreamParams(*([None] * 7), **{key: jnp.float32(params[key.upper()])
                          for key in ("tau0", "grain_growth", "extra_growth", "dirt_soot", "swemx")})
    values = np.asarray([row["inputs"] for row in rows], np.float32)
    def evaluate(values):
        age, fage = snow_age(values[:,15], values[:,10], values[:,11], values[:,12], values[:,13], p)
        direct, diffuse = snowalb_bats(jnp.asarray(values[:,9]), fage, read_bats_parameters())
        return jnp.stack((age, fage, *direct, *diffuse), -1)
    expected = np.asarray([row["output"][6:] for row in rows])
    for result in (evaluate(values), jax.jit(evaluate)(jnp.asarray(values))):
        actual = np.asarray(result)
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, **fx["tolerance"])
        assert actual.dtype == np.float32


def test_all_default_keys_leave_static_selection_absent(monkeypatch):
    monkeypatch.setattr("gpuwrf.io.noahmp_options.read_bats_parameters",
                        lambda: pytest.fail("default must not read optional BATS data"))
    assert selected_options({}) is None
    assert selected_options({"noah_mp": DEFAULTS}) is None


@pytest.mark.parametrize("code", [0, 6, 1.5, float("nan"), float("inf")])
def test_invalid_snow_codes_fail_closed(code):
    errors = option_refusals({"noah_mp": {"opt_snf": code}})
    assert errors and "opt_snf" in errors[0]


@pytest.mark.parametrize("surface", [2, 4])
def test_mosaic_refusal_explains_wrf_scope(surface):
    errors = option_refusals({"physics": {"sf_surface_physics": surface, "sf_surface_mosaic": 1}})
    assert any("WRF ignores it with Noah-MP" in e and "Noah-classic" in e for e in errors)


def test_nondefault_options_require_native_noahmp():
    assert option_refusals({"noah_mp": {"opt_snf": 4}}, replay=True)
    assert option_refusals({"noah_mp": {"opt_snf": 4}, "physics": {"sf_surface_physics": 0}})
    assert not option_refusals({"noah_mp": {"opt_snf": 4}, "physics": {"sf_surface_physics": 4}})


def test_binding_and_static_key_survive_pytree_roundtrip():
    from test_noahmp_checkpoint_v2 import _namelist
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.namelist_binding import bind_namelist_options
    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    nl = _namelist(GridSpec.canary_3km_template())
    # Existing dynamics defaults are independently tested by the shared suite.
    base = bind_namelist_options(nl, {}, "d01")
    same = bind_namelist_options(nl, {"noah_mp": DEFAULTS}, "d01")
    active = bind_namelist_options(nl, {"noah_mp": {"opt_alb": 1, "opt_snf": 4}}, "d01")
    assert base.noahmp_options is None and same.noahmp_options is None
    assert static_config_hash(base) == static_config_hash(same)
    assert static_config_hash(base) != static_config_hash(active)
    leaves, tree = jax.tree.flatten(active)
    restored = jax.tree.unflatten(tree, leaves)
    assert restored.noahmp_options == active.noahmp_options
    assert static_config_hash(restored) == static_config_hash(active)


def test_both_real_pipeline_loaders_bind_snow_options(monkeypatch, tmp_path):
    import test_namelist_binding as binding
    from gpuwrf.io.namelist_binding import bind_namelist_options as real_bind
    selections = []
    def bind(nl, data, domain, **kw):
        data = dict(data, noah_mp={"opt_alb": 1, "opt_snf": 4})
        result = real_bind(nl, data, domain, **kw)
        selections.append(result.noahmp_options)
        return result
    monkeypatch.setattr(binding.nested_pipeline, "bind_namelist_options", bind)
    monkeypatch.setattr(binding.daily_pipeline, "bind_namelist_options", bind)
    nested = tmp_path / "nested"
    daily = tmp_path / "daily"
    nested.mkdir(); daily.mkdir()
    bundles = binding._load_nested(monkeypatch, nested, {})
    for bundle in bundles.values():
        assert bundle.namelist.noahmp_options.opt_snf == 4
        assert bundle.namelist.noahmp_options.opt_alb == 1
    # The replay loader binds the request, then refuses the unsupported route.
    with pytest.raises(ValueError,match="Nondefault Noah-MP choices require the native"):
        binding._build_daily_case(monkeypatch, daily, "")
    assert len(selections) == 4


def test_noahmp_snow_selector_checkpoint_roundtrip(tmp_path):
    from test_noahmp_checkpoint_v2 import _state, _namelist, _land
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint_with_land_state
    grid = GridSpec.canary_3km_template()
    nl = replace(_namelist(grid), noahmp_options=NoahMPOptions(1, 4, read_bats_parameters()))
    land = _land()
    path = tmp_path / "snow.pkl"
    write_checkpoint(_state(grid), nl, grid, 1, path, land_state=land)
    _, restored, _, _, loaded = read_checkpoint_with_land_state(path)
    assert restored.noahmp_options == nl.noahmp_options
    for name in land.__slots__:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(land, name))
