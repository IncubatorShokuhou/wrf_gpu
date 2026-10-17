"""Bounded CPU cost for offline operations, not model/GPU forecast performance."""
from __future__ import annotations

import json
import inspect
import resource
import sys
import time

import numpy as np

from gpuwrf.ens.statistics import crps, products

random = np.random.default_rng(901)
values = random.normal(280, 5, size=(12, 117, 267))
observations = random.normal(280, 5, size=(117, 267))
operations = {"products": lambda: products(values, thresholds=[273.15, 283.15, 303.15]),
              "crps_empirical": lambda: crps(values, observations),
              "crps_fair": lambda: crps(values, observations, fair=True)}
receipt = {"schema": "u1-ens.cpu-cost.v1", "work": "12members x117x267 surfacegrid; all float64 offline", "repeats": 7,
           "input_bytes": values.nbytes+observations.nbytes, "qualification": "CPU cost only; no GPU/model claim", "timings": {}}
for name, operation in operations.items():
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter()-start)
    outputs = result.values() if isinstance(result, dict) else [result]
    assert all(np.isfinite(value).all() for value in outputs)
    receipt["timings"][name] = {"median_s": float(np.median(samples)), "samples_s": samples}
receipt["process_maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
from gpuwrf import cli
candidate = inspect.getsource(cli.build_parser)
addition = '    from gpuwrf.ens.cli import register_parser as register_ens_parser\n\n    register_ens_parser(subparsers)\n'
assert candidate.count(addition) == 1
baseline = candidate.replace(addition, '')
namespace = dict(vars(cli))
exec(compile(baseline, cli.__file__, 'exec'), namespace)
for name, operation in (("parser_without_ens", namespace['build_parser']), ("parser_with_ens", cli.build_parser)):
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter()-start)
    receipt['timings'][name] = dict(median_s=float(np.median(samples)), repeats=100)
bridge = sys.modules.get("jax._src.xla_bridge")
assert not getattr(bridge, "_backends", {})
receipt["jax_backends_initialized"] = False
print(json.dumps(receipt, indent=2, allow_nan=False))
