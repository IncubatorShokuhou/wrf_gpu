"""Frozen D6 arithmetic with bounded read-only Dataset reuse per domain.

Same I/O wrapper as t-p3fix TP33: avoid reopening every 375-variable header
for each field/frame. No model execution or tolerance changes.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import netCDF4

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('case', type=Path)
p.add_argument('--domains', nargs='+', default=['d01'])
p.add_argument('--hours', type=int, default=24)
a = p.parse_args()
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
scorer_path = Path(__file__).with_name('score.py')
spec = importlib.util.spec_from_file_location('frozen_d6', scorer_path)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
original = scorer.Dataset
netCDF4.set_chunk_cache(256 << 10, 521, .75)
for domain in a.domains:
    start = time.monotonic()
    with ExitStack() as stack:
        handles = {}
        class Held:
            def __init__(self, ds): self.ds = ds
            def __enter__(self): return self.ds
            def __exit__(self, *exc): return False
        def held(path, mode='r', *args, **kwargs):
            assert mode == 'r' and not args and not kwargs
            key = str(path)
            if key not in handles:
                handles[key] = stack.enter_context(original(key, 'r'))
            return Held(handles[key])
        scorer.Dataset = held
        result = scorer.score(a.source, a.case, domain, a.hours)
        opened = len(handles)
    scorer.Dataset = original
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), domain=domain,
        scope='Frozen D6 with read-only handles reused; no model/tolerance change',
        scorer_sha256=hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
        manifest_sha256=result['manifest_sha256'], verdict=result['verdict'],
        paired_frames=result['paired_frames'], bounded_misses=len(result['failures']),
        nonfinite=len(result['nonfinite']), wall_s=time.monotonic()-start,
        handles=opened, all_handles_closed=True, cache='256KiB NetCDF per-variable cache; per-domain handles closed',
        GPU_executed=False)
    (a.case / f'SCORING_IO_{domain}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
