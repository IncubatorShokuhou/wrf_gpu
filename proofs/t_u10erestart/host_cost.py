"""Measure the changed host admission only; no model/GPU performance claim."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import statistics
import sys
import time
import tracemalloc

assert os.environ['JAX_PLATFORMS'] == 'cpu'
root, output = map(lambda value: Path(value).resolve(), sys.argv[1:])
sys.path.insert(0, str(root/'src'))
import numpy as np
from gpuwrf.io import restart_schema as schema

assert Path(schema.__file__).resolve() == root/'src/gpuwrf/io/restart_schema.py'
results = []
for shape in ((42, 42), (117, 267)):
    zero = np.zeros(shape, np.float32)
    history = SimpleNamespace(t2=zero, u10e=zero+4.5, v10e=zero-2.5)
    for surface in (5, 2):
        nml = {'sf_sfclay_physics': surface}
        for _ in range(100):
            schema.validate_history_winds(history, nml, shape=shape)
        samples = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            schema.validate_history_winds(history, nml, shape=shape)
            samples.append(time.perf_counter_ns()-start)
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        schema.validate_history_winds(history, nml, shape=shape)
        additional = tracemalloc.get_traced_memory()[1]-before
        tracemalloc.stop()
        results.append({'shape': shape, 'sf_sfclay_physics': surface,
                        'samples': len(samples), 'median_us': statistics.median(samples)/1000,
                        'p95_us': sorted(samples)[949]/1000,
                        'additional_tracked_peak_bytes': additional})
report = {'platform': 'cpu', 'gpu_executed': False, 'cache': 'NONE',
          'source': str(root), 'schema_sha256': hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest(),
          'scope': 'host validity checks outside timestep; concurrent shared host, not whole-forecast/GPU timing',
          'memory_scope': 'additional tracemalloc-tracked Python/NumPy peak; inputs allocated before measurement',
          'results': results}
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
