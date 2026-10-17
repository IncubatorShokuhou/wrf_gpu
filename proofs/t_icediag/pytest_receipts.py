"""Durable per-node results under hard RAM caps; no assertion modifications."""
import gc
import json
import os
from pathlib import Path
import resource


def pytest_collection_modifyitems(config, items):
    path = os.environ.get('ICEDIAG_COLLECTION')
    if path:
        Path(path).write_text(json.dumps([item.nodeid for item in items], indent=2) + '\n')
    skip = os.environ.get('ICEDIAG_COMPLETED')
    if skip:
        completed = set(json.loads(Path(skip).read_text()))
        items[:] = [item for item in items if item.nodeid not in completed]


def pytest_runtest_logreport(report):
    path = os.environ.get('ICEDIAG_EVENTS')
    if path:
        row = dict(node=report.nodeid, phase=report.when, outcome=report.outcome,
                   duration_s=report.duration, maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if report.failed:
            row['failure'] = str(report.longrepr)
        with open(path, 'a') as stream:
            stream.write(json.dumps(row) + '\n')
    if report.when == 'teardown' and os.environ.get('ICEDIAG_CLEAR_JAX') == '1':
        import jax
        jax.clear_caches()
        gc.collect()
