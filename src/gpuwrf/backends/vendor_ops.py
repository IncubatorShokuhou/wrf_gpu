"""Vendor-specific Pallas-Triton constructs behind one interface (v0.5.0 port groundwork).

Kernel bodies stay vendor-neutral ``jnp``. This module owns the few spellings
that differ between GPU vendors and selects which code shape a body takes:

``vendor()``         ``"cuda"`` | ``"rocm"`` | ``"interp"``: who executes a Pallas body.
``triton_native()``  native Triton code shape (``vendor() != "interp"``).
``platform_triton_native()``  the same decision for backend-dark guards that consult
                     the actual PJRT client rather than the local facade.
``amd_rn(op, a, b)`` WRF-literal binary32/64 ``+ - *`` as AMDGCN instructions with one
                     IEEE round-to-nearest each, never contracted into an FMA (the
                     analogue of the PTX ``{op}.rn.f32`` spelling). Division is the plain
                     IEEE ``/`` (the AMDGCN ``v_div_scale/fmas/fixup`` sequence; Owner D3).
``portable_rn(op, a, b, interpret=)`` the non-CUDA branch of the existing RN helpers.

CUDA is unchanged by construction. CUDA call sites keep their own PTX emitters, so
their Python frames and source lines stay identical and the CUDA program (canonical
jaxpr, raw HLO, Triton IR with locations) is byte-identical (FINDINGS E109/E134).
This module only supplies the non-CUDA branches.

ROCm native Triton is an unqualified capability tuple today: it runs interpreted
(the S1-port correctness route) with a one-time warning unless
``GPUWRF_ROCM_TRITON_NATIVE=1`` explicitly admits the experimental native route.
The variable is read at trace time, so it enters the AOT trace-env hash.
"""
from __future__ import annotations

import os
import threading
import warnings

import jax
import jax.numpy as jnp

from . import platform as _platform

_TRUE = ("1", "true", "on", "yes")
_override: str | None = None  # census/test hook only; see _override_vendor
_warned = [False]
_local = threading.local()  # per-kernel interpreter fallback while one body is traced

# Route table: Pallas kernel names that stay interpreted on the native ROCm route
# (a kernel that fails a qualified tuple's gate). Empty until hardware evidence.
ROCM_INTERPRET_KERNELS: frozenset[str] = frozenset()

# AMDGCN spellings: f32 VOP2 forms; f64 has no v_sub_f64 (negated source modifier).
_AMD_ASM = {
    ("add", "f32"): "v_add_f32 $0, $1, $2",
    ("sub", "f32"): "v_sub_f32 $0, $1, $2",
    ("mul", "f32"): "v_mul_f32 $0, $1, $2",
    ("add", "f64"): "v_add_f64 $0, $1, $2",
    ("sub", "f64"): "v_add_f64 $0, $1, -$2",
    ("mul", "f64"): "v_mul_f64 $0, $1, $2",
}


def rocm_native_admitted(environ=None) -> bool:
    """Explicit opt-in for the unqualified ROCm native Triton route."""
    env = os.environ if environ is None else environ
    return env.get("GPUWRF_ROCM_TRITON_NATIVE", "0").strip().lower() in _TRUE


def _rocm_native() -> bool:
    if _platform.runtime_backend() != "rocm":
        return False
    if rocm_native_admitted():
        return True
    if not _warned[0]:
        _warned[0] = True
        warnings.warn("gpuwrf: ROCm device detected; Pallas kernels run INTERPRETED (no qualified "
                      "ROCm capability tuple). GPUWRF_ROCM_TRITON_NATIVE=1 admits the experimental "
                      "native Triton route.", RuntimeWarning, stacklevel=3)
    return False


def interpreted_kernels(environ=None) -> frozenset[str]:
    """Route table plus ``GPUWRF_ROCM_INTERPRET_KERNELS`` (comma-separated kernel names)."""
    env = os.environ if environ is None else environ
    extra = {name.strip() for name in env.get("GPUWRF_ROCM_INTERPRET_KERNELS", "").split(",") if name.strip()}
    return ROCM_INTERPRET_KERNELS | extra


def kernel_name(args, kwargs):
    """``pallas_call`` name, else the kernel function's name (through functools.partial)."""
    if kwargs.get("name"):
        return kwargs["name"]
    kernel = args[0] if args else kwargs.get("kernel")
    return getattr(getattr(kernel, "func", kernel), "__name__", None)


def vendor() -> str:
    """Code shape of a Pallas body: the local facade decides CUDA (release probes patch it)."""
    from . import pallas as facade

    if facade.cuda_native():
        return "cuda"
    if getattr(_local, "interp", 0):
        return "interp"  # this body is being traced for the interpreter fallback
    if _override is not None:
        return _override
    return "rocm" if _rocm_native() else "interp"


def triton_native() -> bool:
    return vendor() != "interp"


def rocm_pallas_call(native_call, interpreted_call):
    """Facade ``pallas_call`` for an admitted ROCm tuple: native unless the kernel is routed to the interpreter.

    A routed kernel's body is traced with ``vendor() == "interp"`` (plain arithmetic, no AMDGCN asm),
    which the interpreter can lower; every other kernel keeps the native callable.
    """
    def call(*args, **kwargs):
        if kernel_name(args, kwargs) not in interpreted_kernels():
            return native_call(*args, **kwargs)
        inner = interpreted_call(*args, **kwargs)

        def run(*operands, **kw):
            _local.interp = getattr(_local, "interp", 0) + 1
            try:
                return inner(*operands, **kw)
            finally:
                _local.interp -= 1
        return run
    return call


def platform_triton_native() -> bool:
    """Backend-dark guards (actual PJRT client): CUDA, or admitted ROCm."""
    if _override is not None:
        return _override != "interp"
    return _platform.cuda_native() or _rocm_native()


def amd_rn(op, a, b):
    """One AMDGCN instruction per op: exact IEEE RN, no contraction (W5P02 codegen probe)."""
    a, b = jnp.broadcast_arrays(a, b)
    dtype = jnp.dtype(a.dtype)
    if b.dtype != dtype or dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"amd_rn needs matching f32/f64 operands, got {a.dtype}/{b.dtype}")
    if op == "div":
        return a / b  # IEEE-correct AMDGCN division sequence (Owner decision D3)
    from jax.experimental.pallas import triton as pt

    width = "f32" if dtype == jnp.dtype(jnp.float32) else "f64"
    return pt.elementwise_inline_asm(
        _AMD_ASM[(op, width)], args=[a, b], constraints="=v,v,v", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, dtype)])[0]


def portable_rn(op, a, b, *, interpret):
    """Non-CUDA branch of the literal-RN helpers: AMDGCN on admitted ROCm, else plain ops."""
    if not interpret and vendor() == "rocm":
        return amd_rn(op, a, b)
    return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
            "div": lambda: a / b}[op]()


def pre_ampere_route_requested(environ=None) -> bool:
    """Opt-in: native Pallas below compute capability 8.0 through the stock-Triton kernel-call route."""
    env = os.environ if environ is None else environ
    return env.get("GPUWRF_PRE_AMPERE_KERNEL_CALL", "0").strip().lower() in _TRUE


_kernel_call_route = {"installed": False, "checked": False}


def _cuda_cc(device) -> float | None:
    if getattr(device, "platform", None) != "gpu" or _platform.device_backend(device) != "cuda":
        return None
    try:
        return float(getattr(device, "compute_capability"))
    except (TypeError, ValueError):
        return None


def ensure_kernel_call_route() -> bool:
    """Lower every Triton ``pallas_call`` via stock Triton + jaxlib ``triton_kernel_call`` on CUDA cc < 8.0.

    XLA's in-graph Triton refuses devices below compute capability 8.0; the kernel-call route
    (``gpuwrf.kernels._triton_kernel_call``, JAX's own non-default launch branch) compiles the same
    TTIR with the installed stock Triton for the actual device. Only with
    ``GPUWRF_PRE_AMPERE_KERNEL_CALL=1`` AND a CUDA device below cc 8.0; otherwise nothing is touched
    (no device query while the variable is unset). Unqualified tuple: experimental.
    """
    if _kernel_call_route["installed"]:
        return True
    if not pre_ampere_route_requested() or _kernel_call_route["checked"]:
        return False
    _kernel_call_route["checked"] = True
    cc = _cuda_cc(jax.devices()[0])
    if cc is None or cc >= 8.0:
        return False
    from jax._src.pallas import core as pallas_core
    from jax._src.pallas.triton import core as triton_core
    from jax._src.pallas.triton import pallas_call_registration as registration

    def kernel_call_rule(ctx, *nodes, **params):
        from gpuwrf.kernels._triton_kernel_call import _lowering

        if params.get("compiler_params") is None:
            params["compiler_params"] = triton_core.CompilerParams()
        return _lowering(ctx, *nodes, **params)

    pallas_core.register_lowering_rule(triton_core.CompilerParams, kernel_call_rule, "gpu")
    registration.pallas_call_lowering = kernel_call_rule  # pallas_calls without compiler_params
    _kernel_call_route["installed"] = True
    warnings.warn(f"gpuwrf: CUDA compute capability {cc} < 8.0: every Pallas kernel is compiled by stock Triton "
                  "and launched through triton_kernel_call (GPUWRF_PRE_AMPERE_KERNEL_CALL=1, experimental, "
                  "unqualified tuple).", RuntimeWarning, stacklevel=3)
    return True


def _override_vendor(name: str | None, native: bool = True) -> None:
    """CPU census/test hook: pretend the kernels run for ``name`` (never product code)."""
    global _override
    if name not in (None, "cuda", "rocm", "interp"):
        raise ValueError(name)
    _override = None if name is None else (name if native else "interp")
