"""Record completed cases and release test compilation caches between cases."""
import ctypes
import gc
import json
import os
from pathlib import Path
import sys

OUT = Path(os.environ['RESTART_MEMORY_RECEIPT'])


def record(**values):
    status = Path('/proc/self/status').read_text().splitlines()
    values['memory'] = [line for line in status if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:'))]
    with OUT.open('a') as stream:
        stream.write(json.dumps(values)+'\n')


def pytest_collection_finish(session):
    record(event='collection', nodes=[item.nodeid for item in session.items])


def pytest_runtest_logreport(report):
    record(event='report', nodeid=report.nodeid, when=report.when,
           outcome=report.outcome, duration=report.duration)
    if report.when == 'teardown':
        if 'jax' in sys.modules:
            sys.modules['jax'].clear_caches()
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
        record(event='released', nodeid=report.nodeid)
