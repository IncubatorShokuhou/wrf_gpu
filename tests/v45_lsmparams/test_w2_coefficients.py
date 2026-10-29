"""Private clock folds: public refusal, exact REAL pins, live derived operands."""
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

from gpuwrf.surface_params import controls as sc
from gpuwrf.surface_params.lsmparams import clock_controls


def context(values):
    # Algebraic pins for host wiring only; consumer test below uses real PROD.
    soil = lambda v: np.full((4, 1, 2), v, np.float32)
    ep = SimpleNamespace(smcmax=soil(1.), smcwlt=soil(0.), bexp=soil(3.), psisat=soil(1.))
    return SimpleNamespace(surface_controls=values, noahmp_energy_params=ep)


def test_off_and_legacy_clock_maps_keep_identity():
    assert clock_controls(context(None)) is None
    values = {"eg": np.ones((1, 2), np.float32)}
    assert clock_controls(context(values)) is values


@pytest.mark.parametrize("name", ("smcmax", "bexp", "psisat"))
def test_private_clock_binding_keeps_public_refusal(name):
    with pytest.raises(ValueError, match="not available"):
        clock_controls(context({name: np.ones((4, 1, 2), np.float32)}))


def test_derived_operands_have_exact_real_pins_and_changed_values(monkeypatch):
    # This test's local research policy never changes shipped admission.
    monkeypatch.setattr(sc, "UNAVAILABLE_SURFACE_CONTROLS", frozenset())
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    values = {name: getattr(context(None).noahmp_energy_params, name)
              for name in ("smcmax", "bexp", "psisat")}
    before = clock_controls(context(values))
    changed = {**values, "smcmax": values["smcmax"] * np.float32(.5)}
    after = clock_controls(context(changed))
    expected = {"__w2_inv_smcmax": np.float32(1.),
                "__w2_thkdry": np.float32(64.7) / np.float32(2700.),
                "__w2_inv_d_rsurf": np.float32(1.) / np.float32(2.2e-5)}
    for name, value in expected.items():
        shape = (1, 2) if name == "__w2_inv_d_rsurf" else (4, 1, 2)
        assert before[name].dtype == np.float32 and before[name].shape == shape
        np.testing.assert_array_equal(np.asarray(before[name]).view(np.uint32),
                                      np.full(shape, value).view(np.uint32))
    assert set(before) == set(after) == set(values) | set(expected)
    for name in expected:
        assert np.asarray(before[name]).tobytes() != np.asarray(after[name]).tobytes()
    # Concrete cache evidence for derived operands, independent of their values.
    fn = jax.jit(lambda operands: operands["__w2_inv_smcmax"] * np.float32(2.))
    x, y = np.asarray(fn(before)), np.asarray(fn(after))
    assert not np.array_equal(x, y) and fn._cache_size() == 1


def test_private_coefficients_alone_cannot_bypass_public_schema():
    with pytest.raises(ValueError, match="public soil controls"):
        sc.apply_surface_controls(object(), object(), object(), {"__w2_inv_smcmax": np.ones((4, 1, 2), np.float32)})


def namelist(values):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    return OperationalNamelist(grid=None,tendencies=None,metrics=None,dt_s=18.,
                               surface_controls=values,noahmp_energy_params=context(values).noahmp_energy_params)


@pytest.mark.parametrize("lane_arrays", (False, True))
def test_actual_clock_builder_delivers_derived_operands(monkeypatch, lane_arrays):
    from gpuwrf.runtime.operational_mode import build_clock_base
    monkeypatch.setattr(sc,"UNAVAILABLE_SURFACE_CONTROLS",frozenset())
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    values={"smcmax": np.ones((4,1,2),np.float32)}
    arrays=tuple(np.full((1,2),v,np.float32) for v in (280.,.5,.8)) if lane_arrays else None
    cb=build_clock_base(namelist(values),noahmp_static_arrays=arrays)
    assert "__w2_inv_smcmax" in cb.surface_controls, "build_clock_base dropped the coefficient seam"
    np.testing.assert_array_equal(np.asarray(cb.surface_controls["__w2_inv_smcmax"]),
                                  np.ones((4,1,2),np.float32))
    if lane_arrays:
        np.testing.assert_array_equal(np.asarray(cb.noahmp_tbot),arrays[0])


@pytest.mark.parametrize("batch_size", (1,2))
def test_actual_pipeline_batch_clock_callers(monkeypatch,batch_size):
    from gpuwrf.integration.nested_pipeline import _clock_bases_for_batch_domain
    monkeypatch.setattr(sc,"UNAVAILABLE_SURFACE_CONTROLS",frozenset())
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    members=tuple(namelist({"smcmax":np.full((4,1,2),v,np.float32)}) for v in (1.,.5))
    tree=SimpleNamespace(domains={"d01":SimpleNamespace(namelist=members[0])})
    cb=_clock_bases_for_batch_domain(tree=tree,name="d01",batch_namelists={"d01":members},batch_size=batch_size)
    assert "__w2_inv_smcmax" in cb.surface_controls
    expected=(np.ones((4,1,2),np.float32) if batch_size==1 else
              np.stack([np.full((4,1,2),np.float32(v)) for v in (1.,2.)]))
    np.testing.assert_array_equal(np.asarray(cb.surface_controls["__w2_inv_smcmax"]),expected)


def test_implicit_jitted_clock_refuses_baked_control_values(monkeypatch):
    from gpuwrf.runtime.operational_mode import build_clock_base
    monkeypatch.setattr(sc,"UNAVAILABLE_SURFACE_CONTROLS",frozenset())
    member=namelist({"psisat":np.ones((4,1,2),np.float32)})
    with pytest.raises(ValueError,match="explicit clock base"):
        jax.jit(lambda:build_clock_base(member))()


def test_real_prod_soil_operands_keep_default_bytes_and_live_changes(monkeypatch):
    import json
    import jax.numpy as jnp
    from gpuwrf.physics.noahmp.energy import EnergyParams, _tdfcnd_soil
    from gpuwrf.surface_params.lsmparams import COEFFICIENT_KEYS
    from gpuwrf.runtime.operational_mode import build_clock_base
    from gpuwrf.physics.noahmp.precision import native_real_enabled
    if not native_real_enabled():
        pytest.skip("shipped REAL path is tested in the clean release-default suite")
    monkeypatch.setattr(sc,"UNAVAILABLE_SURFACE_CONTROLS",frozenset())
    data=json.loads((Path(__file__).parent/'fixtures/prod_soil_operands.json').read_text())
    assert data['coords']==[[0,101],[2,73],[28,91]]
    ep=EnergyParams(**{key:jnp.asarray(data['parameters'][key],jnp.float32)
        if key in data['parameters'] else None for key in EnergyParams._fields})
    smc,sh2o=(jnp.asarray(data[key],jnp.float32) for key in ('smois','sh2o'))
    nl=replace(namelist({'smcmax':ep.smcmax}),noahmp_energy_params=ep)
    def program(smc,sh2o,clock):
        values=clock.surface_controls
        coefficients={target:values[key] for key,target in COEFFICIENT_KEYS.items() if key in values}
        return _tdfcnd_soil(smc,sh2o,ep._replace(smcmax=values['smcmax'],w2_coefficients=coefficients))
    fn=jax.jit(program)
    a=jax.jit(lambda smc,sh:_tdfcnd_soil(smc,sh,ep))(smc,sh2o)
    b=fn(smc,sh2o,build_clock_base(nl))
    x,y=(jax.device_get(jax.tree.leaves(v)) for v in (a,b))
    assert all(np.isfinite(v).all() for v in x+y)
    assert all(np.asarray(u).tobytes()==np.asarray(v).tobytes() for u,v in zip(x,y,strict=True)), \
        'real PROD soil default bytes changed: coefficient producer/consumer seam lost'
    changed=replace(nl,surface_controls={'smcmax':ep.smcmax*jnp.float32(1.05)})
    c=fn(smc,sh2o,build_clock_base(changed))
    assert fn._cache_size()==1
    assert any(np.asarray(u).tobytes()!=np.asarray(v).tobytes() for u,v in zip(jax.tree.leaves(b),jax.tree.leaves(c),strict=True))
