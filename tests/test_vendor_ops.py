"""vendor_ops: vendor selection and the non-CUDA literal-RN spellings (lane o5-port, CPU only).

The ROCm checks lower Pallas bodies to Triton IR for platform 'rocm' and compile one
tiny kernel with stock Triton for gfx942 on the CPU. That is code-generation evidence
only, never AMD hardware qualification (no AMD device executes anything here).
"""
import re
import warnings

import jax
import jax.extend  # noqa: F401
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as jpl
from jax._src.pallas.triton import lowering as tl

from gpuwrf.backends import pallas as facade
from gpuwrf.backends import platform, vendor_ops


@pytest.fixture(autouse=True)
def _fresh_traces():
    """Pallas kernel traces are cached per function object (E140/E161)."""
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture
def rocm_census(monkeypatch):
    """Trace Pallas bodies as an admitted ROCm tuple would (CPU census hook)."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    vendor_ops._override_vendor("rocm")
    yield
    vendor_ops._override_vendor(None)


def _ttirs(fn, *args, platform_name="rocm"):
    """Lower every pallas_call (recursively) of fn(*args) to Triton IR text."""
    texts = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                p = eqn.params
                res = tl.lower_jaxpr_to_triton_module(p["jaxpr"], p["grid_mapping"], platform_name, None)
                texts.append(res.module.operation.get_asm(enable_debug_info=False))
            for sub in jax.extend.core.jaxprs_in_params(eqn.params):
                walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    assert texts, "no pallas_call traced"
    return texts


def test_cpu_default_is_interpreted():
    assert vendor_ops.vendor() == "interp"
    assert not facade.triton_native()
    assert facade.pallas_call is facade._jax_call
    assert not vendor_ops.platform_triton_native()


def test_cuda_selection_unchanged(monkeypatch):
    monkeypatch.setattr(platform, "runtime_backend", lambda: "cuda")
    assert vendor_ops.vendor() == "cuda"
    assert facade.pallas_call is jpl.pallas_call
    assert vendor_ops.platform_triton_native()


def test_unqualified_rocm_runs_interpreted_and_warns(monkeypatch):
    monkeypatch.setattr(platform, "runtime_backend", lambda: "rocm")
    monkeypatch.delenv("GPUWRF_ROCM_TRITON_NATIVE", raising=False)
    monkeypatch.setattr(vendor_ops, "_warned", [False])
    with pytest.warns(RuntimeWarning, match="INTERPRETED"):
        assert vendor_ops.vendor() == "interp"
    assert facade.pallas_call is facade._jax_call
    assert not vendor_ops.platform_triton_native()


def test_admitted_rocm_gets_native_callable(monkeypatch):
    monkeypatch.setattr(platform, "runtime_backend", lambda: "rocm")
    monkeypatch.setenv("GPUWRF_ROCM_TRITON_NATIVE", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert vendor_ops.vendor() == "rocm"
    assert facade.pallas_call is not facade._jax_call
    assert vendor_ops.platform_triton_native()


def test_rocm_route_table_selects_per_kernel(monkeypatch):
    calls = []
    native = lambda *a, **k: calls.append(("native", vendor_ops.vendor())) or (lambda *x: "N")  # noqa: E731
    interp = lambda *a, **k: (lambda *x: calls.append(("interp", vendor_ops.vendor())) or "I")  # noqa: E731
    monkeypatch.setenv("GPUWRF_ROCM_INTERPRET_KERNELS", " slow_kernel ,other")
    vendor_ops._override_vendor("rocm")
    try:
        call = vendor_ops.rocm_pallas_call(native, interp)
        assert call(lambda r: None, name="fast_kernel")() == "N"
        assert call(lambda r: None, name="slow_kernel")() == "I"

        def other(ref):
            pass
        assert call(other)() == "I"  # function name when no pallas name is given
        assert vendor_ops.vendor() == "rocm"  # fallback scope ends with the call
    finally:
        vendor_ops._override_vendor(None)
    assert calls == [("native", "rocm"), ("interp", "interp"), ("interp", "interp")]
    assert vendor_ops.interpreted_kernels({}) == vendor_ops.ROCM_INTERPRET_KERNELS


def _body_primitives(fn, *args):
    eqns = [e for e in jax.make_jaxpr(fn)(*args).jaxpr.eqns if e.primitive.name == "pallas_call"]
    assert len(eqns) == 1
    return eqns[0].params, {e.primitive.name for e in eqns[0].params["jaxpr"].eqns}


def test_rocm_routed_kernel_is_interpreted_with_plain_arithmetic(rocm_census, monkeypatch):
    params, prims = _body_primitives(_literal_chain, SPEC32, SPEC32, SPEC32)
    assert not params["interpret"] and "elementwise_inline_asm_p" in prims
    jax.clear_caches()
    monkeypatch.setenv("GPUWRF_ROCM_INTERPRET_KERNELS", "_literal_chain_kernel")
    params, prims = _body_primitives(_literal_chain, SPEC32, SPEC32, SPEC32)
    assert params["interpret"] and "elementwise_inline_asm_p" not in prims and "mul" in prims


def _literal_chain_kernel(a_ref, b_ref, c_ref, o_ref):
    from gpuwrf.kernels.dyn_rk_fp32 import _rn

    a, b, c = a_ref[...], b_ref[...], c_ref[...]
    o_ref[...] = _rn("div", _rn("add", _rn("mul", a, b, interpret=False), c, interpret=False), b, interpret=False)


def _literal_chain(a, b, c):
    return facade.pallas_call(_literal_chain_kernel, out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
                              grid=(1,))(a, b, c)


SPEC32 = jax.ShapeDtypeStruct((128,), jnp.float32)


def test_rocm_literal_rn_is_amdgcn_asm(rocm_census):
    (text,) = _ttirs(_literal_chain, SPEC32, SPEC32, SPEC32)
    assert "v_mul_f32 $0, $1, $2" in text and "v_add_f32 $0, $1, $2" in text
    # Deletion-sensitive: a plain-op fallback would emit contractible arith ops.
    assert "arith.mulf" not in text and "arith.addf" not in text
    assert "arith.divf" in text  # IEEE division (Owner D3), no asm
    assert ".rn.f32" not in text


def test_cuda_literal_rn_still_ptx(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(facade, "cuda_native", lambda: True)
    (text,) = _ttirs(_literal_chain, SPEC32, SPEC32, SPEC32, platform_name="cuda")
    for op in ("mul", "add", "div"):
        assert f"{op}.rn.f32 $0, $1, $2;" in text
    assert "v_mul_f32" not in text


def test_amd_rn_f64_spellings(rocm_census):
    def kernel(a_ref, b_ref, o_ref):
        a, b = a_ref[...], b_ref[...]
        o_ref[...] = vendor_ops.amd_rn("mul", vendor_ops.amd_rn("sub", a, b), a)

    spec = jax.ShapeDtypeStruct((128,), jnp.float64)
    with jax.enable_x64(True):
        (text,) = _ttirs(lambda a, b: facade.pallas_call(kernel, out_shape=spec, grid=(1,))(a, b), spec, spec)
    assert "v_add_f64 $0, $1, -$2" in text and "v_mul_f64 $0, $1, $2" in text
    assert "arith.subf" not in text and "arith.mulf" not in text


def test_amd_rn_rejects_mixed_dtypes():
    with pytest.raises(TypeError):
        vendor_ops.amd_rn("add", jnp.zeros(4, jnp.float32), jnp.zeros(4, jnp.int32))


def test_rocm_ring_relax_and_scalar_couple_lower(rocm_census):
    """W5P01: these two release kernels did not lower off CUDA (f64 reduce_precision)."""
    from gpuwrf.coupling import boundary_apply
    from gpuwrf.kernels import dyn_ring_relax_fp32 as ring

    nz, ny, nx, w = 3, 9, 10, 5
    f32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)  # noqa: E731
    leaf = f32(4, w, nz, max(ny, nx))
    texts = _ttirs(lambda fld, v, r, fx, gx, mu, c1, c2: ring.ring_relax_tendency(
        fld, v, r, fx, gx, spec=1, relax=4, mass=(mu, c1, c2)),
        f32(nz, ny, nx), leaf, leaf, f32(w), f32(w), f32(ny, nx), f32(nz), f32(nz))
    texts += _ttirs(boundary_apply._couple_scalar_real4_arrays.__wrapped__,
                    f32(nz, ny, nx), f32(ny, nx), f32(nz), f32(nz))
    for text in texts:
        assert "v_mul_f32 $0, $1, $2" in text and "v_add_f32 $0, $1, $2" in text
        assert not re.search(r"\bf64\b", text), "reduce_precision prescreen leaked into the ROCm body"


def _rn_census(text):
    count = lambda pattern: len(re.findall(pattern, text))  # noqa: E731
    return {"mul": (count(r"v_mul_f32 \$0, \$1, \$2"), count(r"mul\.rn\.f32")),
            "add": (count(r"v_add_f32 \$0, \$1, \$2"), count(r"add\.rn\.f32")),
            "sub": (count(r"v_sub_f32 \$0, \$1, \$2"), count(r"sub\.rn\.f32")),
            "plain": (count(r"arith\.mulf"), count(r"arith\.addf"), count(r"arith\.subf")),
            "div": (count(r"arith\.divf"), count(r"div\.rn\.f32"))}


def test_rocm_sint_sides_keeps_every_literal_rn(monkeypatch):
    """Critic V11: the nest SINT kernel's non-CUDA rn must stay one AMDGCN instruction per CUDA .rn site."""
    from gpuwrf.nesting.interp import build_sint_weights
    from gpuwrf.nesting.sint_kernel import sint_sides

    weights = build_sint_weights(parent_grid_ratio=3, i_parent_start=5, j_parent_start=7, parent_ny=30,
                                 parent_nx=31, child_ny=28, child_nx=31)
    field = jnp.zeros((4, 30, 31), jnp.float32)
    fn = lambda: sint_sides(field, weights, parent_grid_ratio=3, width=5, side_len=32)  # noqa: E731
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(facade, "cuda_native", lambda: True)
    (cuda,) = _ttirs(fn, platform_name="cuda")
    monkeypatch.setattr(facade, "cuda_native", lambda: False)
    jax.clear_caches()
    vendor_ops._override_vendor("rocm")
    try:
        (rocm,) = _ttirs(fn)
    finally:
        vendor_ops._override_vendor(None)
    c, r = _rn_census(cuda), _rn_census(rocm)
    for op in ("mul", "add", "sub"):
        assert c[op][0] == 0 and c[op][1] > 0, (op, c)
        assert r[op] == (c[op][1], 0), (op, r, c)  # same RN sites, AMDGCN spelling, no PTX
    assert r["plain"] == c["plain"] and r["plain"][:2] == (0, 0), (r, c)  # no contractible fallback
    assert r["div"] == (c["div"][1], 0) and c["div"][0] == 0  # IEEE '/' where CUDA uses div.rn
    assert not re.search(r"\bf64\b", rocm)


@pytest.mark.parametrize("vendor, selected", (("rocm", True), ("interp", False), ("cuda", True)))
def test_mcica_kiss_kernel_follows_triton_native_guard(monkeypatch, vendor, selected):
    """Critic V12: an admitted ROCm tuple keeps the McICA KISS lane kernel; the interpreter never takes it."""
    from gpuwrf.kernels import rad_mcica

    calls = []
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "1")
    monkeypatch.setattr(rad_mcica, "kiss_random_lanes", lambda *a, **k: calls.append(k) or "lanes")
    vendor_ops._override_vendor(vendor)
    try:
        out = rad_mcica._random_values(jnp.full((2, 5), 5.0e4, jnp.float32), first=0, ng=3, legacy_fp64=False)
    finally:
        vendor_ops._override_vendor(None)
    assert (out == "lanes") is selected and len(calls) == int(selected), (vendor, calls)


def test_rocm_amdgcn_has_no_fma_for_literal_chain(rocm_census, tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton_cache"))  # never the shared ~/.triton
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from gpuwrf.kernels.dyn_rk_fp32 import _rn

    if "amd" not in backends:
        pytest.skip("stock Triton without the AMD backend")

    def literal(a_ref, b_ref, c_ref, o_ref):
        o_ref[...] = _rn("add", _rn("mul", a_ref[...], b_ref[...], interpret=False), c_ref[...], interpret=False)

    def plain(a_ref, b_ref, c_ref, o_ref):  # control: the contractible spelling
        o_ref[...] = a_ref[...] * b_ref[...] + c_ref[...]

    counts = {}
    for name, body in (("literal", literal), ("plain", plain)):
        (text,) = _ttirs(lambda a, b, c, body=body: facade.pallas_call(
            body, out_shape=SPEC32, grid=(1,))(a, b, c), SPEC32, SPEC32, SPEC32)
        path = tmp_path / f"{name}.ttir"
        path.write_text(text)
        asm = triton.compile(str(path), target=GPUTarget("hip", "gfx942", 64),
                             options={"num_warps": 4, "num_stages": 1}).asm["amdgcn"]
        ins = [line.split()[0] for line in asm.splitlines() if re.match(r"\s+v_", line)]
        counts[name] = {"fma": sum(bool(re.match(r"v_(pk_)?fma[ckma]*_f32", i)) for i in ins),
                        "mul": sum(i.startswith(("v_mul_f32", "v_pk_mul_f32")) for i in ins),
                        "add": sum(i.startswith(("v_add_f32", "v_pk_add_f32")) for i in ins)}
    assert counts["literal"]["fma"] == 0 and counts["literal"]["mul"] >= 1 and counts["literal"]["add"] >= 1, counts
    assert counts["plain"]["fma"] >= 1, counts  # the control proves the check can fail


def test_portable_rn_plain_on_interpreter():
    x = jnp.asarray([1, 3, 5], jnp.float32)
    for op, ref in (("add", x + 2), ("sub", x - 2), ("mul", x * 2), ("div", x / 2)):
        np.testing.assert_array_equal(vendor_ops.portable_rn(op, x, jnp.float32(2), interpret=True), ref)


class _FakeCuda:
    platform = "gpu"
    device_kind = "NVIDIA GeForce GTX 1080 Ti"

    def __init__(self, cc):
        self.compute_capability = cc
        self.client = type("Client", (), {"platform_version": "cuda 12090"})()


@pytest.fixture
def kernel_call_route(monkeypatch):
    """Isolate ensure_kernel_call_route: restore JAX's own Triton lowering rule afterwards."""
    from jax._src.pallas import core as pallas_core
    from jax._src.pallas.triton import core as triton_core
    from jax._src.pallas.triton import pallas_call_registration as registration

    original = (pallas_core.get_lowering_rule(triton_core.CompilerParams, "gpu"), registration.pallas_call_lowering)
    monkeypatch.setattr(vendor_ops, "_kernel_call_route", {"installed": False, "checked": False})
    yield lambda: pallas_core.get_lowering_rule(triton_core.CompilerParams, "gpu")
    pallas_core.register_lowering_rule(triton_core.CompilerParams, original[0], "gpu")
    registration.pallas_call_lowering = original[1]


def test_kernel_call_route_off_by_default_without_device_query(kernel_call_route, monkeypatch):
    monkeypatch.delenv("GPUWRF_PRE_AMPERE_KERNEL_CALL", raising=False)
    monkeypatch.setattr(jax, "devices", lambda *a: pytest.fail("device queried while the route is not requested"))
    before = kernel_call_route()
    assert vendor_ops.ensure_kernel_call_route() is False
    assert kernel_call_route() is before


@pytest.mark.parametrize("cc,installed", [("6.1", True), ("7.5", True), ("8.0", False), ("8.6", False)])
def test_kernel_call_route_only_below_ampere(kernel_call_route, monkeypatch, cc, installed):
    monkeypatch.setenv("GPUWRF_PRE_AMPERE_KERNEL_CALL", "1")
    monkeypatch.setattr(jax, "devices", lambda *a: [_FakeCuda(cc)])
    before = kernel_call_route()
    if installed:
        with pytest.warns(RuntimeWarning, match="stock Triton"):
            assert vendor_ops.ensure_kernel_call_route() is True
        assert kernel_call_route() is not before and kernel_call_route().__name__ == "kernel_call_rule"
    else:
        assert vendor_ops.ensure_kernel_call_route() is False
        assert kernel_call_route() is before


def test_kernel_call_route_ignores_cpu(kernel_call_route, monkeypatch):
    monkeypatch.setenv("GPUWRF_PRE_AMPERE_KERNEL_CALL", "1")
    before = kernel_call_route()
    assert vendor_ops.ensure_kernel_call_route() is False  # real CPU device
    assert kernel_call_route() is before
