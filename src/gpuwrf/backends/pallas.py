"""Local Pallas dispatch: original CUDA callable or pure-JAX interpretation.

This facade changes no JAX globals. On CUDA ``pallas_call`` is the original
function object, preserving call stacks/source locations and native IR.
On other backends Pallas's interpreter lowers the same body to JAX operations;
these are portable execution, not an independent scientific reference.
"""
from jax.experimental import pallas as _pallas

from .platform import cuda_native


def _jax_call(*args, **kwargs):
    kwargs["interpret"] = True
    kwargs.pop("compiler_params", None)
    return _pallas.pallas_call(*args, **kwargs)


def __getattr__(name):
    if name == "pallas_call":
        return _pallas.pallas_call if cuda_native() else _jax_call
    return getattr(_pallas, name)
