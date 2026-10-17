"""Reuse the established release jaxpr probe with portable dispatch held native at trace time.

CPU-only: initialization interprets Pallas, then tracing returns the original
Pallas callable just as the predecessor probe does. No GPU execution claim.
usage: python release_identity.py SOURCE_ROOT OUTPUT_JSON [CASE]
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
        # Only this local facade changes. Backend/vendor guards and the retained
        # CPU Thomas solver keep their real CPU policy, matching the old probe.
        facade.cuda_native = lambda: True


pipeline._load_domains = load_on_cpu
probe = root / "proofs/o1_nlbind/probe.py"
source = probe.read_text()
source = source.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                        'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-port")')
sys.argv = [str(probe), str(root), str(output), "release"]
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
result = json.loads(output.read_text())
result["port_probe"] = {"upstream_probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                        "platform": "cpu", "native_facade_at_trace": True,
                        "gpu_executed": False}
output.write_text(json.dumps(result, indent=2) + "\n")
