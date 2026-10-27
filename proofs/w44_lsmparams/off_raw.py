"""Run the established real-case trace and lower full native CUDA HLO, CPU only.

Use one stable alias for both frozen trees to keep raw source metadata exact.
No native compilation or GPU initialization occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--alias", required=True)
p.add_argument("--out", required=True)
p.add_argument("--case", required=True)
a = p.parse_args()
root = Path(a.alias)  # Deliberately never resolve this import/metadata alias.
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
import os
os.environ["O1_CASE"] = a.case
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
source = source.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                        'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams")')
needle = '        if STEPS:\n'
assert source.count(needle) == 1
source = source.replace(needle, '''        original_backend = jax.default_backend
        jax.default_backend = lambda: "gpu"  # Native CUDA lowering only (E156).
        try:
            native_trace = om._advance_chunk_fori.trace(
                carry, ns, jax.numpy.asarray(1, jax.numpy.int32), cb,
                n_steps=1, cadence=int(ns.radiation_cadence_steps))
            lowered = native_trace.lower(lowering_platforms=("cuda",))
            hlo = lowered.compiler_ir("hlo").as_hlo_text()
        finally:
            jax.default_backend = original_backend
        Path(OUT + ".hlo").write_text(hlo)
        rec["raw_hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
        rec["raw_hlo_bytes"] = len(hlo.encode())
        rec["native_custom_calls"] = hlo.count("custom-call(")
        rec["gpu_executed"] = False
        if STEPS:
''')
sys.argv = [str(probe), str(root), a.out, "release"]
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
result = json.loads(Path(a.out).read_text())
assert not result["arms"]["release"].get("error"), result
assert result["arms"]["release"]["native_custom_calls"] > 0, result
