"""Bounded real-range REAL4 RN check and CompileOnly CUDA codegen probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', required=True, type=Path)
p.add_argument('--target', type=Path)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
from gpuwrf.physics.wrf_cam_ozone import _real4_div_rn

assert jax.devices()[0].platform == 'cpu'
F = np.float32
# Every integer second over 72h and one-ULP neighbours, all REAL month-day
# representable values over [0, 366], plus a dense signed longitude cohort.
seconds = np.arange(259201, dtype=F)
days = np.arange(F(1).view('I'), F(366).view('I') + 1, 31, dtype=np.uint32).view(F)
longitude = np.linspace(-180, 180, 500001, dtype=F)
groups = [(seconds, 60), (seconds / F(60), 1440),
          (F(2 * np.pi) * days, 365), (longitude, 15),
          (np.linspace(-30, 31, 300001, dtype=F), 60)]
for delta in (28, 29, 30, 31):
    groups.append((np.linspace(0, delta, 300001, dtype=F), delta))
rows = []
for x, divisor in groups:
    x = np.concatenate([np.nextafter(x, F(-np.inf)), x, np.nextafter(x, F(np.inf))])
    # Both operands traced: NumPy true division is the independent RN reference.
    got = np.asarray(jax.jit(_real4_div_rn)(x, F(divisor)))
    expected = np.asarray(x / F(divisor), F)
    np.testing.assert_array_equal(got.view('I'), expected.view('I'))
    rows.append(dict(divisor=divisor, values=x.size, max_ULP=0))
rec = dict(platform='cpu', GPU_executed=False, ranges=rows,
           total_values=sum(r['values'] for r in rows), exact=True)
if a.target:
    from jax.experimental import topologies
    from jax._src import compiler
    compiler._get_cross_compile_backend = lambda _: None
    topology = topologies.get_topology_desc(platform='cuda', target_config=a.target.read_text(), topology='1x1x1')
    sharding = jax.sharding.SingleDeviceSharding(topology.devices[0])
    s = jax.ShapeDtypeStruct((1024,), jnp.float32, sharding=sharding)
    lowered = jax.jit(_real4_div_rn).trace(s, s).lower(lowering_platforms=('cuda',))
    raw = lowered.compiler_ir('hlo').as_serialized_hlo_module_proto()
    rec['raw_full_constant_proto_sha256'] = hashlib.sha256(raw).hexdigest()
    compiled = lowered.compile(compiler_options={'xla_gpu_autotune_level': 0})
    module = compiled.runtime_executable().hlo_modules()[0]
    text = module.to_string()
    (a.out / 'quotient.optimized.hlo').write_text(text)
    rec['optimized_full_constant_proto_sha256'] = hashlib.sha256(module.as_serialized_hlo_module_proto()).hexdigest()
    assert 'f64[1024]{0} divide(' in text, text
    assert 'f32[1024]{0} convert(' in text, text
    assert 'custom-call(' not in text
    rec['cuda_optimized_f64_divide'] = True
(a.out / 'quotient.json').write_text(json.dumps(rec, indent=2) + '\n')
print(json.dumps(rec, indent=2))
