"""The supported km_opt=4 consumer reads runtime c_s without specializing it."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf import runtime_parameters as rp
from gpuwrf.dynamics.explicit_diffusion import horizontal_deformation_2d, smag2d_horizontal_km

ROOT = Path(__file__).resolve().parents[2]


def test_changed_c_s_pristine_wrf_and_compiled_default(tmp_path):
    from netCDF4 import Dataset
    spec = importlib.util.spec_from_file_location("smag_reference", ROOT / "proofs/w44_params/smag_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    with Dataset("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfinput_d01") as ds:
        u = jnp.asarray(ds["U"][0, :8, :16, :17], jnp.float32)
        v = jnp.asarray(ds["V"][0, :8, :17, :16], jnp.float32)
        mx, my = (jnp.asarray(ds[name][0, :16, :16], jnp.float32) for name in ("MAPFAC_MX", "MAPFAC_MY"))
        dx, dy = float(ds.DX), float(ds.DY)
    deform = horizontal_deformation_2d(u, v, dx_m=dx, dy_m=dy)
    library = oracle.build(tmp_path)
    traces = []

    def compute(fields, mx, my, c_s):
        traces.append(1)
        return smag2d_horizontal_km(*fields, dx_m=dx, dy_m=dy, c_s=c_s, msftx=mx, msfty=my)

    fn = jax.jit(compute)
    outputs = []
    for cs in (.25, .5):
        got = tuple(map(np.asarray, fn(deform, mx, my, jnp.float32(cs))))
        ref = oracle.evaluate(library, *map(np.asarray, deform), np.asarray(mx), np.asarray(my), dx=dx, dy=dy, c_s=cs)
        for actual, truth in zip(got, ref, strict=True):
            assert np.isfinite(actual).all() and np.isfinite(truth).all()
            # smag2d_km owns the interior ring on specified/nested domains.
            actual, truth = actual[:, 1:-1, 1:-1], truth[:, 1:-1, 1:-1]
            assert np.any(truth > 0)
            err = actual.astype(float) - truth.astype(float)
            assert np.sqrt(np.mean(err**2)) <= 2e-5 * max(1., np.sqrt(np.mean(truth.astype(float)**2)))
            assert np.max(np.abs(err)) <= 2e-4 * max(1., np.max(np.abs(truth)))
        outputs.append(got)
    assert traces == [1]
    assert any(not np.array_equal(a, b) for a, b in zip(*outputs))
    static = jax.jit(lambda fields, mx, my: smag2d_horizontal_km(
        *fields, dx_m=dx, dy_m=dy, c_s=.25, msftx=mx, msfty=my))(deform, mx, my)
    for a, b in zip(outputs[0], static, strict=True):
        np.testing.assert_array_equal(a.view(np.uint32), np.asarray(b).view(np.uint32))


@pytest.mark.usefixtures("research_diffusion_policy")
def test_c_s_admission_and_key(monkeypatch):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash
    monkeypatch.setenv("GPUWRF_DYN_FP32", "1")
    base = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., diff_opt=1, km_opt=4)
    a, b = (rp.bind_diffusion_parameters(base, {"c_s": value}) for value in (.25, .5))
    assert static_config_hash(a) == static_config_hash(b) != static_config_hash(base)
    assert carry_aval_hash((None, a), {}) == carry_aval_hash((None, b), {})
    assert rp.restart_parameter_digests(a) != rp.restart_parameter_digests(b)
    for km in (1, 2, 3, 5):
        with pytest.raises(ValueError, match="km_opt=4"):
            rp.bind_diffusion_parameters(replace(base, km_opt=km), {"c_s": .25})
    with pytest.raises(ValueError, match="c_k.*deferred"):
        rp.bind_diffusion_parameters(base, {"c_k": .15})


@pytest.mark.parametrize("specified", [False, True])
@pytest.mark.usefixtures("research_diffusion_policy")
def test_actual_diffusion_caller_forwards_c_s(monkeypatch, specified):
    from gpuwrf.runtime import operational_mode as om
    spec = importlib.util.spec_from_file_location("smag_call_fixture", ROOT / "tests/dynamics/test_diffopt1_smagorinsky_integration.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    grid = fixture._build_grid(ny=12, nx=16, nz=6, dx=4000.)
    if specified:
        grid = replace(grid, bc=replace(grid.bc, source="wrf"))
    state = fixture._build_state(grid)
    base = fixture._namelist(grid, diff_opt=1, km_opt=4)
    calls = []
    original = om.smag2d_horizontal_km

    def check(*args, **kwargs):
        calls.append(isinstance(kwargs["c_s"], jax.core.Tracer))
        return original(*args, **kwargs)

    monkeypatch.setattr(om, "smag2d_horizontal_km", check)
    # Direct field construction isolates forwarding from admission's native-backend
    # policy; the existing caller fixture deliberately runs the retained REAL64 path.
    a = replace(base, diffusion_parameters={"c_s": jnp.float32(.25)})
    b = replace(base, diffusion_parameters={"c_s": jnp.float32(.5)})
    fn = jax.jit(lambda state, nl: om._diffopt1_dry_forward_tendencies(state, nl, return_scalar_xkhh=True)[-1])
    first, second = np.asarray(fn(state, a)), np.asarray(fn(state, b))
    assert calls == ([True, True] if specified else [True])
    assert np.isfinite(first).all() and np.isfinite(second).all() and np.any(first > 0)
    np.testing.assert_array_equal(second, 4 * first)
