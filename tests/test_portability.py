"""Backend selection and CUDA-free execution; simulated vendors are not hardware qualification."""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.backends import pallas as portable
from gpuwrf.backends import platform


@pytest.mark.parametrize("pin", ["cpu", "rocm", "tpu", "metal", "oneapi"])
def test_bootstrap_non_cuda_never_queries_nvidia(pin, monkeypatch):
    import subprocess
    from gpuwrf.runtime import compile_cache, xla_autotune, gpu_allocator, gpu_preflight

    monkeypatch.setenv("JAX_PLATFORMS", pin)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("vendor tool invoked"))
    assert platform.target_backend() == pin
    assert compile_cache._backend_tag() == pin
    assert not xla_autotune._gpu_is_target()[0]
    env = {"JAX_PLATFORMS": pin}
    assert not gpu_allocator.configure_cli_pool(SimpleNamespace(), env)
    assert env == {"JAX_PLATFORMS": pin}
    status = gpu_preflight.run_nested_gpu_preflight(environ=env)
    assert status["skipped"] and status["backend"] == pin


@pytest.mark.parametrize("version,kind,expected", [
    ("CUDA 13.2", "NVIDIA RTX", "cuda"),
    ("ROCm 7.0", "AMD MI300", "rocm"),
    ("unknown", "other", "gpu"),
])
def test_generic_gpu_is_not_a_cuda_capability(version, kind, expected):
    dev = SimpleNamespace(platform="gpu", client=SimpleNamespace(platform_version=version), device_kind=kind)
    assert platform.device_backend(dev) == expected


def test_gpu_alias_with_amd_visibility_is_not_cuda(monkeypatch):
    monkeypatch.setattr(platform.os.path, "exists", lambda _: False)
    assert platform.target_backend({"JAX_PLATFORMS": "gpu", "ROCR_VISIBLE_DEVICES": "0"}) == "rocm"
    assert platform.target_backend({"JAX_PLATFORMS": "cuda", "ROCR_VISIBLE_DEVICES": "0"}) == "cuda"


def test_cuda_pallas_is_original_callable(monkeypatch):
    from jax.experimental import pallas

    monkeypatch.setattr(platform, "runtime_backend", lambda: "cuda")
    assert portable.pallas_call is pallas.pallas_call
    assert portable.program_id is pallas.program_id


@pytest.mark.parametrize("backend", ["cpu", "rocm", "tpu", "metal", "gpu"])
def test_non_cuda_pallas_runs_as_jax(backend, monkeypatch):
    monkeypatch.setattr(platform, "runtime_backend", lambda: backend)

    def kernel(x, out):
        out[:] = x[:] * jnp.float32(3) + jnp.float32(2)

    x = jnp.arange(8, dtype=jnp.float32)
    fn = portable.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                              interpret=False, compiler_params={"triton": {}})
    actual = jax.jit(fn)(x)
    np.testing.assert_array_equal(actual, np.arange(8, dtype=np.float32) * 3 + 2)
    hlo = jax.jit(fn).lower(x).as_text()
    assert "triton_kernel_call" not in hlo and "inline_asm" not in hlo


def test_real_rounding_helper_needs_no_ptx_on_cpu():
    from gpuwrf.kernels.dyn_rk_fp32 import _rn

    x = jnp.asarray([1, 3, 5], jnp.float32)
    actual = jax.jit(lambda a: _rn("div", a, jnp.float32(2), interpret=False))(x)
    np.testing.assert_array_equal(actual, np.array([.5, 1.5, 2.5], np.float32))


def test_state_constructor_uses_selected_cpu_backend():
    from gpuwrf.contracts.state import _gpu_device

    assert _gpu_device().platform == "cpu"


def test_all_product_pallas_imports_use_backend_layer():
    root = Path(__file__).resolve().parents[1] / "src/gpuwrf"
    modules = []
    for directory in ("kernels", "nesting", "coupling"):
        for path in (root / directory).glob("*.py"):
            source = path.read_text()
            if "pallas_call(" in source:
                modules.append(path.stem)
                assert "from jax.experimental import pallas as pl" not in source, path
                assert "from gpuwrf.backends import pallas as pl" in source, path
    assert len(modules) >= 30


def test_hydrostatic_kernel_matches_independent_numpy_recurrence():
    from gpuwrf.kernels.phys_hydrostatic_real import hydrostatic_profiles

    rng = np.random.default_rng(20261013)
    nz, ny, nx = 7, 3, 5  # Includes a masked tile tail.
    mut = rng.uniform(70000, 100000, (ny, nx)).astype(np.float32)
    moist = [rng.uniform(0, .005, (nz, ny, nx)).astype(np.float32) for _ in range(6)]
    c1 = np.linspace(.1, 1, nz, dtype=np.float32)
    c2 = np.linspace(50, 100, nz, dtype=np.float32)
    dnw = np.full(nz, -1 / nz, np.float32)
    faces = np.empty((nz + 1, ny, nx), np.float32)
    faces[-1] = 5000
    for k in reversed(range(nz)):
        total = moist[0][k].copy()
        for q in moist[1:]:
            total = total + q[k]
        mass = c1[k] * mut + c2[k]
        faces[k] = faces[k + 1] - ((np.float32(1) + total) * mass) * dnw[k]
    reference = np.float32(.5) * (faces[:-1] + faces[1:])
    fn = jax.jit(hydrostatic_profiles)
    actual, actual_faces = fn(mut, moist, c1, c2, dnw, np.float32(5000))
    for a, b in ((actual, reference), (actual_faces, faces)):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=4 * np.finfo(np.float32).eps, atol=0)
