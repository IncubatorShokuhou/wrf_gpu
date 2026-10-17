"""CPU-only changed-artifact compile/memory receipt; no GPU speed claim.

Run from an immutable tree. The old arm omits the new optional records; both
arms use the same coupled forcedown implementation and identical operands.
"""
import json
from pathlib import Path
import statistics
import sys
import time

import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests")]
from test_s1_nestspec import operands, EXTRA
from gpuwrf.nesting.boundary_construction import _compiled_coupled_updates

assert jax.devices()[0].platform == "cpu"
parent, child, pm, cm, weights = operands.__wrapped__()
receipt = {"platform": "cpu", "source": str(ROOT), "cache": "disabled",
           "work": "constructed signed operator stress, parent32x28 child22x19 nz3 ratio3 bdy_width5",
           "qualification": "CPU operator cost only; GPU launches/time/peak VRAM UNKNOWN", "arms": {}}
for name, state in (("baseline", child.replace(**{f"{n}_bdy": None for n in EXTRA})), ("extra10", child)):
    t0 = time.perf_counter()
    executable = _compiled_coupled_updates.lower(
        state, parent, weights, pm, cm, bdy_width=5, parent_grid_ratio=3,
        sint_kernel=False, no_contract=False).compile()
    compile_s = time.perf_counter() - t0
    jax.block_until_ready(executable(state, parent, weights, pm, cm))
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        result = executable(state, parent, weights, pm, cm)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    memory = executable.memory_analysis()
    receipt["arms"][name] = {"compile_s": compile_s, "warm_median_s": statistics.median(times),
        "state_bytes": state.bytes(), "record_bytes": sum(x.size * x.dtype.itemsize for x in result.values()),
        "memory": {key: getattr(memory, key) for key in (
            "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
