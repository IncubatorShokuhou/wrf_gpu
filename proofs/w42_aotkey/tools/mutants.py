import sys
from pathlib import Path
p = Path("src/gpuwrf/runtime/aot_key_cli.py")
s = Path(sys.argv[2]).read_text()
m = sys.argv[1]
if m == "M1_no_redirect":   # command no longer enters the private cache
    a = '        with private_compile_caches(scratch / "cache"):\n'
    b = '        if True:\n'
elif m == "M2_env_only":    # env redirected, JAX config dir left at the bundle
    a = '    jax.config.update("jax_compilation_cache_dir", str(jax_dir))\n    jax_cache.reset_cache()'
    b = '    jax_cache.reset_cache()'
elif m == "M3_autotune":    # also redirect the keyed autotune dir
    a = '"TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda"}'
    b = '"TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "GPUWRF_XLA_AUTOTUNE_CACHE_DIR": "autotune"}'
elif m == "M4_no_libm":     # GPUWRF_JAX_CACHE_DIR not redirected
    a = '_PRIVATE_CACHE_ENV = {"JAX_COMPILATION_CACHE_DIR": "jax", "GPUWRF_JAX_CACHE_DIR": "jax",'
    b = '_PRIVATE_CACHE_ENV = {"JAX_COMPILATION_CACHE_DIR": "jax",'
assert s.count(a) == 1, m
p.write_text(s.replace(a, b))
