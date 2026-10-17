"""wz1080 box-only Pallas portability shim (EXPERIMENT; not product code).

XLA refuses Triton below compute capability 8.0, so on the GTX 1080 Ti (sm_61) every Pallas kernel of the
release defaults fails to compile. With ``WZ_PALLAS_PORTABLE=1`` this module (on PYTHONPATH via the box's
activate.sh) wraps ``jax.experimental.pallas.pallas_call`` right after it is first imported: on a GPU with
cc < 8.0 every call runs in Pallas *interpret* mode (the kernel body lowered to ordinary XLA ops), which is
what ``gpuwrf.backends.pallas`` already does on non-CUDA backends. Same equations, NOT bitwise to Triton,
slow; for debugging release-default paths only. It also moves the JAX cache to a separate directory so no
portable executable is ever mixed with normal ones. ``WZ_PALLAS_PORTABLE_FORCE=1`` forces interpret on any
device (local test of the hook). No JAX import happens here at interpreter start.
"""
import os
import sys

if os.environ.get("WZ_PALLAS_PORTABLE") == "1":
    import importlib.abc
    import importlib.util

    _cache = os.path.join(os.path.expanduser("~"), "wrf_gpu2_cache", "jax_portable")
    os.environ["GPUWRF_JAX_CACHE_DIR"] = _cache
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _cache
    _state = {"old_gpu": None, "calls": 0}

    def _interpret_needed() -> bool:
        if os.environ.get("WZ_PALLAS_PORTABLE_FORCE") == "1":
            return True
        if _state["old_gpu"] is None:
            import jax

            dev = jax.devices()[0]
            cc = getattr(dev, "compute_capability", None)
            try:
                _state["old_gpu"] = dev.platform == "gpu" and cc is not None and float(cc) < 8.0
            except (TypeError, ValueError):
                _state["old_gpu"] = False
        return bool(_state["old_gpu"])

    def _patch_pallas(pallas_module) -> None:
        original = pallas_module.pallas_call

        def pallas_call(*args, **kwargs):
            if _interpret_needed():
                kwargs["interpret"] = True
                kwargs.pop("compiler_params", None)
                _state["calls"] += 1
            return original(*args, **kwargs)

        pallas_call.__wrapped__ = original
        pallas_call.__doc__ = original.__doc__
        pallas_module.pallas_call = pallas_call
        sys.stderr.write("[wz1080 shim] Pallas portable mode armed (interpret on cc<8.0 GPUs)\n")

    def _patch_platform(platform_module) -> None:
        # gpuwrf's own portable route (used on non-CUDA GPUs): kernels take their pure-JAX arithmetic branch instead
        # of Triton inline PTX, the facade interprets, tridiagonal solves use the JAX Thomas scan.
        original = platform_module.cuda_native

        def cuda_native() -> bool:
            return False if _interpret_needed() else original()

        cuda_native.__wrapped__ = original
        platform_module.cuda_native = cuda_native

    _TARGETS = {"jax.experimental.pallas": _patch_pallas, "gpuwrf.backends.platform": _patch_platform}

    class _PostImport(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            patch = _TARGETS.get(fullname)
            if patch is None or fullname in _done:
                return None
            _done.add(fullname)
            sys.meta_path.remove(self)  # avoid recursion while resolving the real spec
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if len(_done) < len(_TARGETS):
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                patch(module)

            spec.loader.exec_module = exec_module
            return spec

    _done: set = set()
    for _name, _patch in _TARGETS.items():
        if _name in sys.modules:
            _done.add(_name)
            _patch(sys.modules[_name])
    if len(_done) < len(_TARGETS):
        sys.meta_path.insert(0, _PostImport())
