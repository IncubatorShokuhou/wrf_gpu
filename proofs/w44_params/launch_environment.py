"""Record the exact product child environment before CPU-only smoke substitution."""
import os
import sys

FIELDS = ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'CUDA_VISIBLE_DEVICES',
          'GPUWRF_FAST_DEFAULTS', 'GPUWRF_JAX_CACHE', 'GPUWRF_AOT_SLIM_MODULE',
          'GPUWRF_CENSUS', 'GPUWRF_CENSUS_SLIM', 'GPUWRF_RRTMG_MP_RE',
          'XLA_FLAGS', 'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_PREALLOCATE',
          'XLA_CLIENT_MEM_FRACTION', 'OMP_NUM_THREADS', 'JAX_COMPILATION_CACHE_DIR')


def select_device(cpu_smoke):
    product = {name: os.environ.get(name) for name in FIELDS}
    assert product['JAX_PLATFORMS'] == 'cuda', product
    assert product['JAX_PLATFORM_NAME'] is None, product
    assert product['GPUWRF_FAST_DEFAULTS'] == '1' and product['GPUWRF_AOT_SLIM_MODULE'] == '0'
    if cpu_smoke:
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1'
        os.environ['JAX_PLATFORMS'] = 'cpu'  # Sole device-layer substitution, before any JAX import.
    else:
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    return {'product_child_env': product, 'child_argv': list(sys.argv),
            'device_override': {'JAX_PLATFORMS': 'cpu'} if cpu_smoke else {}}
