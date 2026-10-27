"""Last JAX-dark boundary: attest CUDA env, or substitute CPU for launch smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--entry', required=True)
args, rest = p.parse_known_args()
assert os.environ.get('JAX_PLATFORMS') == 'cuda', 'product child must request cuda, never gpu/rocm'
assert not os.environ.get('JAX_PLATFORM_NAME'), 'legacy backend override must be cleared'
assert 'jax' not in sys.modules
smoke = os.environ.get('TOPO_LAUNCH_SMOKE') == '1'
if smoke:
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1', 'CPU smoke must not hold official GPU lock'
else:
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1', 'official GPU lock required'
out = Path(rest[rest.index('--out') + 1])
out.mkdir(parents=True, exist_ok=True)
product_env = {k: v for k, v in os.environ.items() if k.startswith(('GPUWRF_', 'JAX_', 'XLA_', 'TRITON_', 'CUDA_')) or k in ('PYTHONPATH', 'OMP_NUM_THREADS', 'TMPDIR')}
record = dict(entry=args.entry, argv=rest, product_env=product_env,
    product_env_sha256=hashlib.sha256(json.dumps(product_env, sort_keys=True).encode()).hexdigest(),
    CPU_launch_smoke=smoke, backend_substitution={'JAX_PLATFORMS': ['cuda', 'cpu']} if smoke else None)
(out / 'DEVICE_ENV.json').write_text(json.dumps(record, indent=2) + '\n')
if smoke:
    os.environ['JAX_PLATFORMS'] = 'cpu'
os.execv(sys.executable, [sys.executable, args.entry, *rest])
