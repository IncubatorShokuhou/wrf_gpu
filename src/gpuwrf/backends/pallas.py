"""Local Pallas dispatch: original native callable or pure-JAX interpretation.

This facade changes no JAX globals. On CUDA ``pallas_call`` is the original
function object, preserving call stacks/source locations and native IR.
On other backends Pallas's interpreter lowers the same body to JAX operations;
these are portable execution, not an independent scientific reference.
An explicitly admitted ROCm tuple (``vendor_ops``) also gets the original callable.
"""
from jax.experimental import pallas as _pallas

from .platform import cuda_native
from .vendor_ops import amd_rn, platform_triton_native, portable_rn  # noqa: F401  (kernel-facing)


def _jax_call(*args, **kwargs):
    kwargs["interpret"] = True
    kwargs.pop("compiler_params", None)
    return _pallas.pallas_call(*args, **kwargs)


def vendor():
    from . import vendor_ops

    return vendor_ops.vendor()


def triton_native():
    return vendor() != "interp"


def __getattr__(name):
    if name == "pallas_call":
        selected = vendor()
        if selected == "cuda":
            from .vendor_ops import ensure_kernel_call_route

            ensure_kernel_call_route()  # no-op unless GPUWRF_PRE_AMPERE_KERNEL_CALL=1 on cc < 8.0
            return _pallas.pallas_call
        if selected == "rocm":
            from .vendor_ops import rocm_pallas_call

            return rocm_pallas_call(_pallas.pallas_call, _jax_call)
        return _jax_call
    return getattr(_pallas, name)
