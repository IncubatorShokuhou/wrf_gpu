"""Runtime values must be live operands with fixed program keys and strict admission."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf import runtime_parameters as rp
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.runtime.operational_mode import OperationalNamelist

ROOT = Path(__file__).resolve().parents[2]


def base():
    grid = GridSpec.canary_3km_template()
    grid = dataclasses.replace(grid, bc=dataclasses.replace(grid.bc, source="wrf"))
    return OperationalNamelist(grid, None, None, 18., run_boundary=True, diff_6th_opt=2,
                               w_damping=1, damp_opt=3, dampcoef=.2, zdamp=5000.)


@pytest.fixture
def native(monkeypatch):
    from gpuwrf.dynamics import explicit_diffusion
    from gpuwrf.runtime import operational_mode
    monkeypatch.setenv("GPUWRF_DYN_FP32", "1")
    monkeypatch.setattr(explicit_diffusion, "_NATIVE_DIFFUSION_FP32", True)
    monkeypatch.setattr(operational_mode, "_W_DAMP_STAGE", True)


@pytest.mark.usefixtures("research_diffusion_policy")
def test_family_roundtrip_keys_and_digest(native):
    from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash
    original = base()
    a = rp.bind_diffusion_parameters(original, {"diff_6th_factor": .12, "dampcoef": .2})
    b = rp.bind_diffusion_parameters(original, {"diff_6th_factor": .24, "dampcoef": .4})
    assert jax.tree.structure(a) == jax.tree.structure(b) != jax.tree.structure(original)
    assert static_config_hash(a) == static_config_hash(b) != static_config_hash(original)
    assert carry_aval_hash((a,), {}) == carry_aval_hash((b,), {})
    other_names = rp.bind_diffusion_parameters(original, {"zdamp": 5000., "dampcoef": .2})
    assert static_config_hash(a) != static_config_hash(other_names)
    assert len(jax.tree.leaves(a)) == len(jax.tree.leaves(original)) + 3  # two controls + bound RK strengths
    back = jax.tree.unflatten(jax.tree.structure(a), jax.tree.leaves(a))
    assert rp.restart_parameter_digests(back) == rp.restart_parameter_digests(a)
    assert rp.restart_parameter_digests(a) != rp.restart_parameter_digests(b)
    assert rp.restart_parameter_digests(original) == {}
    assert original.tree_flatten()[1] == dataclasses.replace(original, diffusion_parameters=None).tree_flatten()[1]


@pytest.mark.usefixtures("research_diffusion_policy")
def test_two_sixth_order_values_one_compiled_program(native):
    from gpuwrf.kernels.dyn_diff_fp32 import sixth_order_fp32
    shape = (4, 9, 11)
    field = jnp.asarray(np.random.default_rng(19).normal(size=shape), jnp.float32)
    mu, maps = jnp.full(shape[1:], 90000., jnp.float32), jnp.ones(shape[1:], jnp.float32)
    c1, c2 = jnp.linspace(.9, .1, shape[0], dtype=jnp.float32), jnp.zeros(shape[0], jnp.float32)
    traces = []

    def evaluate(namelist, field):
        traces.append(1)
        return sixth_order_fp32(field, mu, c1, c2, maps, maps, name="m", dt=18.,
                                factor=rp.diffusion_value(namelist, "diff_6th_factor"), interpret=True)

    fn = jax.jit(evaluate)
    original = base()
    a = rp.bind_diffusion_parameters(original, {"diff_6th_factor": .12})
    b = rp.bind_diffusion_parameters(original, {"diff_6th_factor": .24})
    first, second = np.asarray(fn(a, field)), np.asarray(fn(b, field))
    assert traces == [1]
    assert np.isfinite(first).all() and np.isfinite(second).all() and np.any(first != 0)
    np.testing.assert_array_equal(second, np.float32(2) * first)
    static = np.asarray(jax.jit(lambda q: evaluate(original, q))(field))
    np.testing.assert_array_equal(first.view(np.uint32), static.view(np.uint32))


def _load_test_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.usefixtures("research_diffusion_policy")
def test_changed_w_crit_cfl_pristine_wrf_and_single_compile(native, tmp_path):
    from gpuwrf.dynamics.core.advance_w import w_damp_rw_tend_wrf
    helper = _load_test_helper("wdamp_inputs", ROOT / "tests/v025/b_core/test_w_damp_stage.py")
    oracle = _load_test_helper("wdamp_reference", ROOT / "tests/v025/b_core/w_damp_oracle.py")
    library = oracle.build(tmp_path)
    inputs = helper._synthetic(42)
    rw, kw = jnp.asarray(inputs["rw_tend"]), {k: jnp.asarray(v) for k, v in inputs.items() if k != "rw_tend"}
    traces = []

    def evaluate(namelist, rw, fields):
        traces.append(1)
        return w_damp_rw_tend_wrf(rw, **fields, dt=18., **rp.w_damping_kwargs(namelist))

    fn = jax.jit(evaluate)
    outputs = []
    for crit in (1., .5):
        nl = rp.bind_diffusion_parameters(base(), {"w_crit_cfl": crit})
        got = np.asarray(fn(nl, rw, kw))
        ref, cfl = oracle.evaluate(library, **inputs, dt=18., w_crit_cfl=crit)
        assert cfl > 2 and np.isfinite(got).all() and np.isfinite(ref).all()
        literal = np.asarray(w_damp_rw_tend_wrf(rw, **kw, dt=18., **rp.w_damping_kwargs(nl)))
        np.testing.assert_array_equal(literal.view(np.uint32), ref.view(np.uint32))
        # PB02: compiled STATIC and runtime-operand programs have identical
        # values, including the existing XLA-vs-Fortran reassociation residual.
        # Byte identity is a regression check; the literal comparison above is
        # the WRF check. Neither claims compiled bitwise equality with Fortran.
        static = np.asarray(jax.jit(lambda r, f: w_damp_rw_tend_wrf(
            r, **f, dt=18., w_crit_cfl=crit))(rw, kw))
        np.testing.assert_array_equal(got.view(np.uint32), static.view(np.uint32))
        outputs.append(got)
    assert traces == [1] and not np.array_equal(*outputs)


def test_changed_sixth_factor_pristine_wrf_on_real_swiss_fields(native, tmp_path):
    """Original Fortran operator, real input crop, existing b-diff bounds unchanged."""
    from netCDF4 import Dataset
    from gpuwrf.kernels.dyn_diff_fp32 import sixth_order_fp32
    oracle = _load_test_helper("diffusion_reference", ROOT / "tests/v025/b_diff/pristine_diffusion.py")
    with Dataset("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfinput_d01") as ds:
        q = np.asarray(ds["T"][0, :8, :16, :16], np.float32)
        mu = np.asarray(ds["MU"][0, :16, :16] + ds["MUB"][0, :16, :16], np.float32)
        c1, c2 = (np.asarray(ds[name][0, :8], np.float32) for name in ("C1H", "C2H"))
        mx, my = (np.asarray(ds[name][0, :16, :16], np.float32) for name in ("MAPFAC_MX", "MAPFAC_MY"))
    # The scalar branch consumes only mass-point maps. Other oracle map arrays
    # are required caller arguments but are not read in name='m'.
    maps = {name: mx if name.endswith("x") else my for name in oracle.MAPS}
    lib = oracle.build(tmp_path)
    values = tuple(map(jnp.asarray, (q, mu, c1, c2, mx, my)))
    compiled = jax.jit(lambda xs, factor: sixth_order_fp32(
        *xs, name="m", dt=18., factor=factor, interpret=True))
    bounds = json.loads((ROOT / "tests/v025/b_diff/gate_manifest.json").read_text())["operator_bounds"]
    assert bounds["rms"] == "2e-5 * max(1, reference RMS)"
    assert bounds["max_abs"] == "2e-4 * max(1, reference max abs)"
    for factor in (.12, .24):
        ref = oracle.run(lib, q, mu, c1, c2, maps, name="m", dt=18., factor=factor, monotonic=True)
        got = np.asarray(compiled(values, jnp.float32(factor)))
        assert np.isfinite(ref).all() and np.isfinite(got).all() and np.any(ref != 0)
        err = got.astype(np.float64) - ref.astype(np.float64)
        assert np.sqrt(np.mean(err**2)) <= 2e-5 * max(1., np.sqrt(np.mean(ref.astype(np.float64)**2)))
        assert np.max(np.abs(err)) <= 2e-4 * max(1., np.max(np.abs(ref)))


@pytest.mark.parametrize("name,value", [("khdif", 100), ("kvdif", 20), ("typo", 1),
    ("diff_6th_factor", -1), ("diff_6th_factor", 2), ("zdamp", 0), ("dampcoef", -1),
    ("dampcoef", np.nan), ("dampcoef", np.inf), ("dampcoef", -np.inf),
    ("dampcoef", [1]), ("dampcoef", True), ("dampcoef", "0.2")])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_bad_values_refused(native, name, value):
    with pytest.raises(ValueError):
        rp.bind_diffusion_parameters(base(), {name: value})


@pytest.mark.parametrize("changes,values", [({"damp_opt": 0}, {"dampcoef": .2}),
    ({"w_damping": 0}, {"w_alpha": .3}), ({"diff_6th_opt": 0}, {"diff_6th_factor": .12}),
    ({"run_boundary": False}, {"diff_6th_factor": .12}), ({"km_opt": 2}, {"diff_6th_factor": .12}),
    ({"tracer_opt": 1}, {"diff_6th_factor": .12})])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_inactive_consumers_refused(native, changes, values):
    with pytest.raises(ValueError):
        rp.bind_diffusion_parameters(dataclasses.replace(base(), **changes), values)


def test_file_schema_and_domain_binding(native, tmp_path):
    path = tmp_path / "parameters.json"
    domains = {"d01": {"diffusion_parameters": {"diff_6th_factor": .24}}}
    path.write_text(json.dumps({"schema": rp.SCHEMA, "domains": domains}))
    original = base()
    with pytest.raises(ValueError, match="failed the GPU qualification"):
        rp.load_parameters(path)
    for domain in ("d01", "d02"):
        with pytest.raises(ValueError, match="failed the GPU qualification"):
            rp.bind_domain_parameters(original, path, domain)
    path.write_text(json.dumps({"schema": rp.SCHEMA, "domains": {"d01": {"typo": {"x": 1}}}}))
    with pytest.raises(ValueError):
        rp.load_parameters(path)
