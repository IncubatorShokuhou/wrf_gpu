"""Full 324-field CPU JIT proof on the declared non-FMA target, not native ISA.

Launch in a FRESH process with XLA_FLAGS including --xla_cpu_max_isa=SSE4_2.
Per-jit compiler_options cannot change an already-created CPU target. This
receipt does not qualify GPU arithmetic or enable either operational PBL.
"""
import json
import os
from pathlib import Path
import runpy
import sys

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert "--xla_cpu_max_isa=SSE4_2" in os.environ.get("XLA_FLAGS", "")

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
test = runpy.run_path(str(ROOT/"tests/test_v040_keps_operators.py"))
records = json.loads(test["FIXTURE"].read_text())["records"]
count = 0
for n in (8, 12, 44):
    rows = [r for r in records if r["n"] == n]
    batch = {k: np.stack([test["inputs"](r)[k] for r in rows]) for k in test["inputs"](rows[0])}
    result = jax.jit(jax.vmap(test["evaluate"]))(batch)
    for i, row in enumerate(rows):
        test["check_fields"]({k: v[i] for k, v in result.items()}, row["output"], row["name"])
        count += len(result)
receipt = {"status": "PASS", "scope": "standalone helpers on non-FMA CPU target ONLY",
           "field_comparisons": count, "records": len(records), "float32_ulps": test["ULPS"],
           "backend": jax.default_backend(), "XLA_FLAGS": os.environ["XLA_FLAGS"],
           "default_native_jit_shear_analytic": "NOT QUALIFIED", "full_pbl": "NOT QUALIFIED"}
assert count == 324
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
