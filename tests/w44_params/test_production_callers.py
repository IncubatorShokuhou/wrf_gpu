"""Deletion-sensitive forwarding at production callsites (not fidelity claims)."""
import __future__
import ast
from dataclasses import replace
import importlib.util
import inspect
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf import runtime_parameters as RP
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.runtime import operational_mode as O

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Captured(Exception):
    def __init__(self, value):
        self.value = value


def capture(value, name):
    assert isinstance(value, jax.core.Tracer), f"{name} not forwarded as a traced operand"
    raise Captured(value)


def two_values(call, first, second):
    traces = []

    def probe(value):
        traces.append(1)
        try:
            call(value)
        except Captured as result:
            return result.value
        raise AssertionError("production consumer was not reached")

    fn = jax.jit(probe)
    for value in (first, second):
        got = np.asarray(fn(jnp.float32(value)))
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got, np.float32(value))
    assert traces == [1], "changing the operand value retraced the production caller"
    fn.clear_cache()


@pytest.fixture(autouse=True)
def clean_traces(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def real_root():
    helper = load("params_real_prod_inputs", "tests/v025/b_core/prod_inputs.py")
    hierarchy, bundles, _, _, _, carries = helper.prod_domains(root_only=True)
    from gpuwrf.runtime.domain_tree import DomainTree
    nl = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains["d01"].namelist
    return carries["d01"], replace(nl, run_physics=False)


@pytest.mark.parametrize("name,default,changed", [("dampcoef", .2, .4), ("zdamp", 5000., 6000.),
                                                    ("dampmag", 3.6, 7.2)])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_acoustic_scan_forwards_native_damping(monkeypatch, real_root, name, default, changed):
    from gpuwrf.kernels import dyn_acoustic_fp32 as N
    carry, nl = real_root
    monkeypatch.setenv("GPUWRF_DYN_FP32", "1")

    def coefficients(state, cfg, **kwargs):
        assert cfg.damping_traced, "native damping was specialized as static"
        capture(getattr(cfg, name), "native acoustic " + name)

    monkeypatch.setattr(N, "calc_coef_fp32", coefficients)
    two_values(lambda value: O._physics_boundary_step(carry,
        replace(nl, diffusion_parameters=({"dampcoef": jnp.float32(.2), "dampmag_rk": jnp.stack([value,value])}
                                        if name == "dampmag" else {name: value})), jnp.int32(0),
        run_radiation=False), default, changed)


@pytest.mark.parametrize("name,changed", [("w_alpha", .6), ("w_beta", 2.), ("w_crit_cfl", .5)])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_acoustic_stage_forwards_each_vertical_cfl_control(monkeypatch, real_root, name, changed):
    carry, nl = real_root
    monkeypatch.setattr(O, "_W_DAMP_STAGE", True)
    key = {"w_alpha": "w_alpha", "w_beta": "w_damp_on", "w_crit_cfl": "w_crit_cfl"}[name]

    def damping(rw, **kwargs):
        capture(kwargs.get(key), "stage vertical-CFL " + name)

    monkeypatch.setattr(O, "_w_damp_rw_tend_wrf", damping)
    two_values(lambda value: O._physics_boundary_step(carry,
        replace(nl, w_damping=1, diffusion_parameters={name: value}), jnp.int32(0),
        run_radiation=False), RP.W_DEFAULTS[name], changed)


@pytest.mark.parametrize("site", ["periodic", "nested_scalars", "root_theta", "root_uvw", "root_scalars"])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_each_sixth_order_production_site_forwards_factor(monkeypatch, real_root, site):
    from gpuwrf.dynamics import explicit_diffusion as ED
    carry, nl = real_root
    # Binding admits diff_6th_factor only on the native path (runtime_parameters: _NATIVE_DIFFUSION_FP32); that flag is
    # import-time (E148), so the CPU-legacy suite (GPUWRF_FAST_DEFAULTS=0) pins it like the release defaults set it.
    monkeypatch.setattr(ED, "_NATIVE_DIFFUSION_FP32", True)
    target = ("sixth_order_diffusion_tendency" if site == "periodic" else
              "wrf_sixth_order_uvw_tendf" if site == "root_uvw" else "wrf_sixth_order_scalar_tendf")
    original = getattr(O, target)

    def diffusion(*args, **kwargs):
        # The root scalar helper runs after the root theta call, with RK1 dt/3.
        if site == "root_scalars" and kwargs["dt"] != float(nl.dt_s) / nl.rk_order:
            return original(*args, **kwargs)
        capture(kwargs.get("diff_6th_factor"), "sixth-order " + site)

    monkeypatch.setattr(O, target, diffusion)

    def call(value):
        selected = replace(nl, diff_6th_opt=2, diffusion_parameters={"diff_6th_factor": value})
        if site == "periodic":
            # This internal legacy consumer remains covered even though public
            # parameter admission correctly refuses periodic sixth-order tuning.
            O._augment_large_step_tendencies(carry.state, nl.tendencies,
                replace(selected, run_boundary=False, use_flux_advection=False), base_state=carry.base_state)
        elif site == "nested_scalars":
            O._nested_scalar_sixth_order_tendencies(carry.state, selected)
        else:
            O._physics_boundary_step(carry, selected, jnp.int32(0), run_radiation=False)

    two_values(call, .12, .24)


@pytest.mark.parametrize("source_leaves", [False, True])
@pytest.mark.parametrize("name", tuple(RP.MYNN))
def test_operational_mynn_caller_forwards_each_parameter(monkeypatch, source_leaves, name):
    helper = load("params_operational_inputs", "tests/test_v013_operational_smoke.py")
    grid = helper._grid(nz=12, ny=3, nx=3)
    state = helper._base_state(grid)
    nl = helper._namelist(grid, dt_s=18., mp_physics=0, cu_physics=0,
                         bl_pbl_physics=5, sf_sfclay_physics=5,
                         sf_surface_physics=0, use_noahmp=False,
                         ra_lw_physics=0, ra_sw_physics=0, rad_rk_tendf=int(source_leaves))
    carry = O.initial_operational_carry(state)
    monkeypatch.setattr(O, "_phys_tend_rk_wrf_enabled", lambda: source_leaves)

    def adapter(*args, **kwargs):
        value = kwargs.get("sqfac") if name == "sqfac" else kwargs.get("mynn_coefficients", {}).get(name)
        capture(value, "operational MYNN " + name)

    entry = "mynn_adapter_with_source_leaves" if source_leaves else "mynn_adapter"
    monkeypatch.setattr(O, entry, adapter)

    def call(value):
        selected = replace(nl, mynn_parameters={name: value})
        O._physics_step_forcing(carry, selected, 0., run_radiation=False, first_timestep=True)

    default = RP.MYNN[name][0]
    two_values(call, default, default * 1.1)


@pytest.mark.parametrize("entry", ["mynn_adapter", "mynn_adapter_with_source_leaves"])
def test_mynn_adapter_forwards_alp1_before_first_call(monkeypatch, entry):
    def initializer(*args, **kwargs):
        capture(kwargs.get("alp1"), entry + " first-call ALP1")

    monkeypatch.setattr(C, "_mynn_state_with_first_call_qke", initializer)
    # The actual adapter reaches its first-call initializer before accessing
    # fields; stop there to isolate the forwarding from the later physics solve.
    two_values(lambda value: getattr(C, entry)(object(), 18., None, first_timestep=True,
                mynn_coefficients={"alp1": value}), .23, .253)


def test_native_mynn_advance_forwards_sqfac(monkeypatch):
    from gpuwrf.kernels import phys_mynn_columns as N
    helper = load("params_predict_inputs", "tests/v025/b_phys/test_mynn_predict_saw_floor.py")
    state, flux, _, _ = helper.operands(helper.DATA["rows"][0])

    def predictor(*args, **kwargs):
        capture(kwargs.get("sqfac"), "native predictor SQFAC")

    monkeypatch.setattr(P, "_mym_predict_qke", predictor)
    two_values(lambda value: N._advance_native(state, 18., False,
                surface=flux, edmf=False, dx=3000., sqfac=value), 3., 6.)


@pytest.mark.parametrize("name,changed", [("w_alpha", .6), ("w_beta", 2.), ("w_crit_cfl", .5)])
def test_each_vertical_cfl_control_changes_the_actual_kernel(name, changed):
    from gpuwrf.dynamics.core.advance_w import w_damp_rw_tend_wrf
    helper = load("params_vertical_cfl_inputs", "tests/v025/b_core/test_w_damp_stage.py")
    inputs = helper._synthetic(42)
    rw = jnp.asarray(inputs.pop("rw_tend"))
    fields = {key: jnp.asarray(value) for key, value in inputs.items()}
    key = {"w_alpha": "w_alpha", "w_beta": "w_damp_on", "w_crit_cfl": "w_crit_cfl"}[name]
    traces = []

    def run(value):
        traces.append(1)
        nl = type("Selection", (), {"diffusion_parameters": {name: value}})()
        kwargs = RP.w_damping_kwargs(nl)
        assert key in kwargs, f"{name} missing from production kernel arguments"
        return w_damp_rw_tend_wrf(rw, **fields, dt=18., **kwargs)

    fn = jax.jit(run)
    before = np.asarray(fn(jnp.float32(RP.W_DEFAULTS[name])))
    after = np.asarray(fn(jnp.float32(changed)))
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert np.any(before != after), f"{name} ignored by production kernel"
    assert traces == [1]


def mutate_function(monkeypatch, module, name, before, after, occurrence=0):
    """Change one live production function without editing the measured tree."""
    original = getattr(module, name)
    source = inspect.getsource(original)
    parts = source.split(before)
    assert len(parts) > occurrence + 1, "mutation site missing"
    source = before.join(parts[:occurrence + 1]) + after + before.join(parts[occurrence + 1:])
    namespace = {}
    # Keep the actual module globals so the positive test's consumer spies
    # remain visible to the mutated function; never freeze a copied namespace.
    exec(compile("from __future__ import annotations\n" + source,
                 "<params-production-caller-mutant>", "exec"), original.__globals__, namespace)
    monkeypatch.setattr(module, name, namespace[name])


@pytest.mark.parametrize("name,default,changed", [("dampcoef", .2, .4), ("zdamp", 5000., 6000.),
                                                    ("dampmag", 3.6, 7.2)])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_acoustic_default_hardcode_is_detected(monkeypatch, real_root, name, default, changed):
    if name == "dampmag":
        mutate_function(monkeypatch, O, "_acoustic_scan",
                        'namelist.diffusion_parameters["dampmag_rk"][int(stage.rk_step != 1)]', 'None')
    else:
        mutate_function(monkeypatch, O, "_acoustic_scan",
                        f'_diffusion_value(namelist, "{name}")', f'float(namelist.{name})')
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_acoustic_scan_forwards_native_damping(monkeypatch, real_root, name, default, changed)


@pytest.mark.usefixtures("research_diffusion_policy")
def test_vertical_cfl_caller_deletion_is_detected(monkeypatch, real_root):
    mutate_function(monkeypatch, O, "_acoustic_core_state_from_prep",
                    "**_w_damping_kwargs(namelist),", "")
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_acoustic_stage_forwards_each_vertical_cfl_control(monkeypatch, real_root, "w_alpha", .6)


@pytest.mark.parametrize("site,function,occurrence", [
    ("periodic", "_augment_large_step_tendencies", 0),
    ("nested_scalars", "_nested_scalar_sixth_order_tendencies", 0),
    ("root_theta", "_rk_scan_step", 0),
    ("root_uvw", "_rk_scan_step", 1),
    ("root_scalars", "_rk_scan_step", 2),
])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_individual_sixth_order_callsite_deletion_is_detected(monkeypatch, real_root, site, function, occurrence):
    mutate_function(monkeypatch, O, function, '_diffusion_value(namelist, "diff_6th_factor")',
                    "float(namelist.diff_6th_factor)", occurrence)
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_each_sixth_order_production_site_forwards_factor(monkeypatch, real_root, site)


def _callee(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def drop_call_kwargs(monkeypatch, module, function, callee, helper):
    """Remove ``**helper(...)`` from the ONE call of ``callee`` inside ``function`` (selected by AST call context,
    so other lanes' extra call sites -- e.g. an opt-in MYNN-history path -- cannot shift the mutated site)."""
    original = getattr(module, function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    sites = [(call, kw) for call in ast.walk(tree) if isinstance(call, ast.Call) and _callee(call.func) == callee
             for kw in call.keywords
             if kw.arg is None and isinstance(kw.value, ast.Call) and _callee(kw.value.func) == helper]
    assert len(sites) == 1, f"expected exactly one {helper} forward in the {callee} call of {function}: {len(sites)}"
    call, kw = sites[0]
    call.keywords.remove(kw)
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), "<params-production-caller-mutant>", "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), original.__globals__, namespace)
    monkeypatch.setattr(module, function, namespace[function])


@pytest.mark.parametrize("source_leaves", [False, True])
def test_individual_operational_mynn_forwarder_deletion_is_detected(monkeypatch, source_leaves):
    entry = "mynn_adapter_with_source_leaves" if source_leaves else "mynn_adapter"
    drop_call_kwargs(monkeypatch, O, "_physics_step_forcing", entry, "_mynn_parameter_kwargs")
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_operational_mynn_caller_forwards_each_parameter(monkeypatch, source_leaves, "alp1")


def test_mynn_forwarder_mutant_selects_by_call_context_not_occurrence():
    """Pin the robustness itself: an extra forwarder in ANOTHER call (budget's history path) must not matter."""
    source = textwrap.dedent(inspect.getsource(O._physics_step_forcing))
    tree = ast.parse(source)
    per_callee = {}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call):
            for kw in call.keywords:
                if kw.arg is None and isinstance(kw.value, ast.Call) and _callee(kw.value.func) == "_mynn_parameter_kwargs":
                    per_callee[_callee(call.func)] = per_callee.get(_callee(call.func), 0) + 1
    assert per_callee.get("mynn_adapter_with_source_leaves") == 1 and per_callee.get("mynn_adapter") == 1, per_callee
    extra = source.replace("def _physics_step_forcing(", "def _probe_extra(", 1) + (
        "\n\ndef _other(namelist):\n    return mynn_sources_with_history(**_mynn_parameter_kwargs(namelist),)\n")
    probe = ast.parse(extra)
    counted = [call for call in ast.walk(probe) if isinstance(call, ast.Call)
               and _callee(call.func) in ("mynn_adapter", "mynn_adapter_with_source_leaves")
               and any(kw.arg is None and isinstance(kw.value, ast.Call)
                       and _callee(kw.value.func) == "_mynn_parameter_kwargs" for kw in call.keywords)]
    assert len(counted) == 2  # the third forwarder sits in a different call and is never selected


@pytest.mark.parametrize("entry", ["mynn_adapter", "mynn_adapter_with_source_leaves"])
def test_individual_first_call_alp1_deletion_is_detected(monkeypatch, entry):
    mutate_function(monkeypatch, C, entry,
        '**({} if mynn_coefficients is None or "alp1" not in mynn_coefficients else {"alp1": mynn_coefficients["alp1"]})', "")
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_mynn_adapter_forwards_alp1_before_first_call(monkeypatch, entry)


def test_native_sqfac_forwarder_deletion_is_detected(monkeypatch):
    from gpuwrf.kernels import phys_mynn_columns as N
    mutate_function(monkeypatch, N, "_advance_native", "**P._sqfac_kwargs(sqfac),", "")
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_native_mynn_advance_forwards_sqfac(monkeypatch)


def test_vertical_cfl_strength_hardcode_is_detected(monkeypatch):
    from gpuwrf.dynamics.core import advance_w as W
    mutate_function(monkeypatch, W, "w_damp_rw_tend_wrf",
        "(np.asarray(W_ALPHA, dt_) if w_alpha is None else jnp.asarray(w_alpha, dt_))", "np.asarray(W_ALPHA, dt_)")
    with pytest.raises(AssertionError, match="w_alpha ignored by production kernel"):
        test_each_vertical_cfl_control_changes_the_actual_kernel("w_alpha", .6)


@pytest.mark.usefixtures("research_diffusion_policy")
def test_vertical_cfl_threshold_mapping_deletion_is_detected(monkeypatch, real_root):
    mutate_function(monkeypatch, RP, "w_damping_kwargs", '"w_beta": "w_damp_on"', '"w_beta": "w_beta_unused"')
    # operational_mode imports the helper by name (``w_damping_kwargs as _w_damping_kwargs``): route the mutant there.
    monkeypatch.setattr(O, "_w_damping_kwargs", RP.w_damping_kwargs)
    with pytest.raises(AssertionError, match="not forwarded as a traced operand"):
        test_acoustic_stage_forwards_each_vertical_cfl_control(monkeypatch, real_root, "w_beta", 2.)
