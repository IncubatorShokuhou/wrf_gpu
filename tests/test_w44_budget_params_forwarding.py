"""Budget opt-in MYNN call forwards the runtime MYNN parameters (cr-w44-params 02:42Z requirement).

With budget history ON, _physics_step_forcing calls mynn_sources_with_history instead of the
production adapter; it must receive exactly the same **_mynn_parameter_kwargs(namelist) operands
(sqfac / mynn_coefficients), traced, so the production projection and the diagnostic twin see them.
Deletion-sensitive: the mutant removes the kwargs line selected by its CALL CONTEXT, not by index.
The same holds for warm-start cycling (cr-w44-budget A7): the budget call must forward cycling=True.
"""
from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf import runtime_parameters as RP
from gpuwrf.coupling import mynn_budget_adapter as BA
from gpuwrf.runtime import operational_mode as O

ROOT = Path(__file__).resolve().parents[1]
FORWARD = "**_mynn_parameter_kwargs(namelist),"
BUDGET_CALL = "mynn,mynn_history=mynn_sources_with_history("


class Captured(Exception):
    def __init__(self, value):
        self.value = value


def _helper():
    spec = importlib.util.spec_from_file_location("budget_params_inputs", ROOT / "tests/test_v013_operational_smoke.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_traces(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


def _two_values(call, first, second):
    traces = []
    def probe(value):
        traces.append(1)
        try:
            call(value)
        except Captured as result:
            return result.value
        raise AssertionError("budget MYNN caller was not reached")
    fn = jax.jit(probe)
    for value in (first, second):
        got = np.asarray(fn(jnp.float32(value)))
        np.testing.assert_array_equal(got, np.float32(value))
    assert traces == [1], "changing the parameter value retraced the budget caller"
    fn.clear_cache()


def _budget_call(monkeypatch, name):
    helper = _helper()
    grid = helper._grid(nz=12, ny=3, nx=3)
    state = helper._base_state(grid)
    nl = helper._namelist(grid, dt_s=18., mp_physics=0, cu_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5,
                          sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0, rad_rk_tendf=1)
    carry = O.initial_operational_carry(state)
    monkeypatch.setattr(O, "_phys_tend_rk_wrf_enabled", lambda: True)

    def production(*args, **kwargs):
        raise AssertionError("budget history ON must not take the production-only MYNN call")

    def budget(*args, budget_config, **kwargs):
        assert budget_config["tke_budget"] == 1 and budget_config["bl_mynn_output"] == 1
        value = kwargs.get("sqfac") if name == "sqfac" else kwargs.get("mynn_coefficients", {}).get(name)
        assert isinstance(value, jax.core.Tracer), f"budget MYNN {name} not forwarded as a traced operand"
        raise Captured(value)

    monkeypatch.setattr(O, "mynn_adapter_with_source_leaves", production)
    monkeypatch.setattr(BA, "mynn_sources_with_history", budget)

    def call(value):
        selected = replace(nl, tke_budget=1, bl_mynn_output=1, mynn_parameters={name: value})
        O._physics_step_forcing(carry, selected, 0., run_radiation=False, first_timestep=True)
    return call


@pytest.mark.parametrize("name", tuple(RP.MYNN))
def test_budget_mynn_caller_forwards_each_parameter(monkeypatch, name):
    default = RP.MYNN[name][0]
    _two_values(_budget_call(monkeypatch, name), default, default * 1.1)


def test_forwarder_occurrences_are_the_two_production_calls_plus_the_budget_call():
    source = inspect.getsource(O._physics_step_forcing)
    assert source.count(FORWARD) == 3
    start = source.index(BUDGET_CALL)
    assert FORWARD in source[start:source.index("\n                )", start)]


def _delete_budget_forwarder(monkeypatch):
    original = O._physics_step_forcing
    source = inspect.getsource(original)
    start = source.index(BUDGET_CALL)
    end = source.index("\n                )", start)
    call = source[start:end]
    assert call.count(FORWARD) == 1, "budget call context changed"
    mutated = source[:start] + call.replace(FORWARD, "") + source[end:]
    namespace = {}
    exec(compile("from __future__ import annotations\n" + mutated, "<budget-params-forwarder-mutant>", "exec"),
         original.__globals__, namespace)
    monkeypatch.setattr(O, "_physics_step_forcing", namespace["_physics_step_forcing"])


CYCLING = '**({"cycling": True} if namelist.cycling else {}),'


def _cycling_call(monkeypatch):
    helper = _helper()
    grid = helper._grid(nz=12, ny=3, nx=3)
    state = helper._base_state(grid)
    nl = helper._namelist(grid, dt_s=18., mp_physics=0, cu_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5,
                          sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0, rad_rk_tendf=1)
    carry = O.initial_operational_carry(state)
    monkeypatch.setattr(O, "_phys_tend_rk_wrf_enabled", lambda: True)
    monkeypatch.setattr(O, "mynn_adapter_with_source_leaves",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("production-only call taken")))
    seen = {}

    def budget(*args, budget_config, **kwargs):
        seen.update(kwargs)
        raise Captured(None)

    monkeypatch.setattr(BA, "mynn_sources_with_history", budget)
    with pytest.raises(Captured):
        O._physics_step_forcing(carry, replace(nl, tke_budget=1, bl_mynn_output=1, cycling=True), 0.,
                                run_radiation=False, first_timestep=True)
    return seen


def test_budget_mynn_caller_forwards_cycling(monkeypatch):
    assert _cycling_call(monkeypatch).get("cycling") is True


def test_budget_cycling_forwarder_deletion_is_detected(monkeypatch):
    original = O._physics_step_forcing
    source = inspect.getsource(original)
    start = source.index(BUDGET_CALL)
    end = source.index("\n                )", start)
    call = source[start:end]
    assert call.count(CYCLING) == 1, "budget call context changed"
    namespace = {}
    exec(compile("from __future__ import annotations\n" + source[:start] + call.replace(CYCLING, "") + source[end:],
                 "<budget-cycling-forwarder-mutant>", "exec"), original.__globals__, namespace)
    monkeypatch.setattr(O, "_physics_step_forcing", namespace["_physics_step_forcing"])
    assert _cycling_call(monkeypatch).get("cycling") is not True


@pytest.mark.parametrize("name", ["alp1", "sqfac"])
def test_budget_forwarder_deletion_is_detected(monkeypatch, name):
    _delete_budget_forwarder(monkeypatch)
    with pytest.raises(AssertionError, match='not forwarded as a traced operand'):
        _two_values(_budget_call(monkeypatch, name), RP.MYNN[name][0], RP.MYNN[name][0] * 1.1)
