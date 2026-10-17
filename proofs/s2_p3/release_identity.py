"""Accepted native-program Swiss probe; only the lane scratch path changes."""
from pathlib import Path
import sys
import json
import hashlib

repo, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path[:0] = [str(repo/'src'), str(repo/'proofs/o1_nlbind')]
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
original_load = pipeline._load_domains

def load_on_cpu(*args, **kwargs):
    facade.cuda_native = lambda: False
    try:
        return original_load(*args, **kwargs)
    finally:
        facade.cuda_native = lambda: True

pipeline._load_domains = load_on_cpu
probe = repo/'proofs/o1_nlbind/probe.py'
text = probe.read_text()
original = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert text.count(original) == 1
text = text.replace(original, f'LANE = Path({str(output.parent)!r})')
sys.argv = [str(probe), str(repo), str(output), 'release']
exec(compile(text, str(probe), 'exec'), {'__name__':'__main__', '__file__':str(probe)})
result = json.loads(output.read_text())
result['port_probe'] = {'platform':'cpu', 'native_facade_at_trace':True, 'gpu_executed':False,
                        'primary_probe_sha256':hashlib.sha256((repo/'proofs/s1_port/release_identity.py').read_bytes()).hexdigest()}
output.write_text(json.dumps(result,indent=2)+'\n')
