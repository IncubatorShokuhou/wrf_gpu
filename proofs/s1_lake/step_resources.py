"""Measure the changed two-column Lake executable on CPU, without a speed claim."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'scripts/v025'))
import cpu_guard  # noqa: E402,F401
import jax
from jax import numpy as jnp
import numpy as np
from gpuwrf.physics.lake_initialization import LakeColumnState
from gpuwrf.physics.lake_step import LakeAtmosphere, lake_step

assert jax.devices()[0].platform == 'cpu'
fixture = repo / 'proofs/s1_lake/fixtures'
cases = json.loads((fixture / 'manifest.json').read_text())['cases']
cases = [next(c for c in cases if c['name'] == name) for name in ('open_shallow', 'snow_19')]
with np.load(fixture / 'columns.npz', allow_pickle=False) as oracle:
    columns = LakeColumnState(*(jnp.asarray(np.stack([
        oracle[f"{c['name']}/step0/{field}"].reshape(-1) for c in cases
    ]), jnp.float32) for field in LakeColumnState._fields))
    columns = columns._replace(**{name: value[:, 0] for name, value in columns._asdict().items()
                                  if value.shape[-1] == 1})
atmosphere = LakeAtmosphere(*(jnp.asarray([c[key] for c in cases], jnp.float32)
    for key in ('ta', 'pbot', 'psfc', 'dzair', 'qv', 'u', 'v', 'lw', 'emiss',
                'rain', 'dt', 'sw', 'alb', 'lat')))
start = time.perf_counter()
executable = jax.jit(lake_step).lower(columns, atmosphere).compile()
compile_s = time.perf_counter() - start
result = jax.block_until_ready(executable(columns, atmosphere))
samples = []
for _ in range(8):
    start = time.perf_counter()
    jax.block_until_ready(executable(columns, atmosphere))
    samples.append(time.perf_counter() - start)
memory = executable.memory_analysis()
leaves = [np.asarray(x) for x in jax.tree.leaves(result)]
report = dict(backend='cpu', n_columns=2, cases=[c['name'] for c in cases], compile_s=compile_s,
    warm_median_s=statistics.median(samples), samples_s=samples,
    compiled_memory={key: getattr(memory, key) for key in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
        'alias_size_in_bytes', 'generated_code_size_in_bytes')},
    materialized_output_bytes=sum(x.nbytes for x in leaves),
    payload_dtypes=sorted({str(np.asarray(x).dtype) for x in result.columns}),
    nonfinite=sum(int((~np.isfinite(x)).sum()) for x in leaves),
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((repo / 'src/gpuwrf/physics').glob('lake_*.py'))},
    scope='Changed compact Lake executable only; CPU memory/readiness/wall diagnostics, no GPU/S1 claim')
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
assert report['nonfinite'] == 0 and report['payload_dtypes'] == ['float32']
