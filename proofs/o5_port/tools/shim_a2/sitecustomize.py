"""o5-port box-only shim (EXPERIMENT, not product code): every Pallas lowering on CUDA through the A2 route.

With ``O5_A2_ROUTE=1`` every Triton ``pallas_call`` is lowered to the unchanged TTIR, compiled by the installed
STOCK Triton for the visible device's compute capability, and launched by jaxlib ``triton_kernel_call``
(gpuwrf.kernels._triton_kernel_call, the CAM-MG route; JAX's own non-default launch branch). XLA therefore
never sees an in-graph ``__gpu$xla.gpu.triton`` call, which it refuses below compute capability 8.0.
Kernel bodies, dispatch (facade native: the CUDA code shape incl. PTX ``.rn`` ops) and the rest of the
program are the product's. Separate JAX cache so no A2 executable mixes with normal ones.
No JAX import at interpreter start; the rule is installed right after JAX registers its own.
"""
import os
import sys

if os.environ.get("O5_A2_ROUTE") == "1":
    import importlib.abc
    import importlib.util

    _cache = os.path.join(os.path.expanduser("~"), "wrf_gpu2_cache", "jax_o5_a2")
    os.environ["GPUWRF_JAX_CACHE_DIR"] = _cache
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _cache
    os.environ["GPUWRF_XLA_AUTOTUNE_CACHE_DIR"] = _cache + "_autotune"
    _TARGET = "jax._src.pallas.triton.pallas_call_registration"
    _state = {"lowered": 0}

    def _install(registration) -> None:
        from jax._src.pallas import core as pallas_core
        from jax._src.pallas.triton import core as triton_core

        def a2_rule(ctx, *nodes, **params):
            from gpuwrf.kernels._triton_kernel_call import _lowering

            if params.get("compiler_params") is None:
                params["compiler_params"] = triton_core.CompilerParams()
            _state["lowered"] += 1
            return _lowering(ctx, *nodes, **params)

        pallas_core.register_lowering_rule(triton_core.CompilerParams, a2_rule, "gpu")
        registration.pallas_call_lowering = a2_rule  # pallas_calls without compiler_params
        sys.stderr.write("[o5-port A2 shim] every Pallas kernel -> stock Triton + triton_kernel_call\n")

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            exec_module = spec.loader.exec_module

            def exec_and_install(module):
                exec_module(module)
                _install(module)

            spec.loader.exec_module = exec_and_install
            return spec

    sys.meta_path.insert(0, _Hook())
