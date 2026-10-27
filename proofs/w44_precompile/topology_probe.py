"""CPU-only API probe; never a model fidelity or performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not Path('/tmp/wrf_gpu2_quiet').exists()
started = time.monotonic()
import jax
import jax.numpy as jnp
from jax.experimental import topologies
from jax._src import compiler
from gpuwrf.runtime.aot_executable import _gpu_blob_format, target_fingerprint

assert all(d.platform == 'cpu' for d in jax.devices())
target = args.target.read_bytes()
topology = topologies.get_topology_desc(platform='cuda', target_config=target.decode(), topology='1x1x1')
dev = topology.devices[0]
spec = jax.ShapeDtypeStruct((16,), jnp.float32, sharding=jax.sharding.SingleDeviceSharding(dev))
lowered = jax.jit(lambda v: jnp.sin(v) + 1).trace(spec).lower(lowering_platforms=('cuda',))
compiler._get_cross_compile_backend = lambda _: None
exe = lowered.compile(compiler_options={'xla_gpu_autotune_level': 0,
    'xla_gpu_experimental_aot_compiled_thunks': True}).runtime_executable()
blob = exe.serialize()
args.out.mkdir(parents=True, exist_ok=False)
(args.out / 'toy.xlaexec').write_bytes(blob)
rec = dict(physical_backend='cpu', gpu_executed=False, target_sha256=hashlib.sha256(target).hexdigest(),
           fingerprint=target_fingerprint(dev), executable_format=_gpu_blob_format(blob),
           bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), wall_s=time.monotonic()-started,
           scope='CompileOnly API feasibility; toy is not model evidence')
(args.out / 'receipt.json').write_text(json.dumps(rec, indent=2)+'\n')
print(json.dumps(rec), flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)  # E146 CompileOnly interpreter-finalization crash
