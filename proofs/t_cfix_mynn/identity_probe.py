"""Lane-local copy of proofs/s1_port/release_identity.py: ONLY the temp-case LANE dir differs
(s1-port -> t-cfix-mynn) and, when F20_DUMP is set, the canonical jaxpr text (the exact string the
probe hashes) is also written to that path for a main-vs-candidate diff. Hashing is untouched.
usage: python identity_probe.py SOURCE_ROOT OUTPUT_JSON [CASE]
"""
from pathlib import Path
import hashlib
import json
import os
import sys

root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if len(sys.argv) > 3:
    os.environ["O1_CASE"] = sys.argv[3]
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
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
probe = root / "proofs/o1_nlbind/probe.py"
source = probe.read_text()
anchor = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert source.count(anchor) == 1
source = source.replace(anchor, 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/t-cfix-mynn/identity")')
dump = os.environ.get("F20_DUMP")
if dump:
    hook = '        rec["sha"] = hashlib.sha256(text.encode()).hexdigest()\n'
    assert source.count(hook) == 1
    source = source.replace(hook, hook + f'        Path({dump!r}).write_text(text)\n')
sys.argv = [str(probe), str(root), str(output), "release"]
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
result = json.loads(output.read_text())
print(json.dumps({k: v.get("sha") for k, v in result["arms"].items()}))
