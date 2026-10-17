"""Changed boundary-consumer cost on CPU; product imports require an immutable tree."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

SRC, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path[:0] = [str(SRC / "src"), str(SRC / "tests")]
import jax
from test_s1_nestspec import operands, EXTRA
from gpuwrf.coupling import boundary_apply as bd

assert jax.devices()[0].platform == "cpu"
assert Path(bd.__file__).resolve().is_relative_to(SRC)
_, child, _, cm, _ = operands.__wrapped__()
config = bd.BoundaryConfig(update_cadence_s=18., nested_frozen_wrf_boundary_bundle=True)
receipt = {"platform": "cpu", "source": str(SRC), "cache": "disabled",
           "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "work": "constructed signed operator stress, child22x19 nz3 bdy_width5 dt6 lead6 parentdt18",
           "qualification": "CPU component cost only; GPU timing/peak VRAM UNKNOWN", "families": {}}
for family, function in (
    ("rk1_tendency", lambda state: bd.nested_scalar_boundary_tendencies(state, 6., cm, 6., config)),
    ("spec_final", lambda state: bd.nested_scalar_spec_final(state, 6., cm, config)),
):
    program = jax.jit(function)
    receipt["families"][family] = {}
    for name, state in (("baseline", child.replace(**{f"{n}_bdy": None for n in EXTRA})), ("extra10", child)):
        t0 = time.perf_counter()
        executable = program.lower(state).compile()
        compile_s = time.perf_counter() - t0
        jax.block_until_ready(executable(state))
        times = []
        for _ in range(7):
            t0 = time.perf_counter()
            jax.block_until_ready(executable(state))
            times.append(time.perf_counter() - t0)
        memory = executable.memory_analysis()
        receipt["families"][family][name] = {
            "compile_s": compile_s, "warm_median_s": statistics.median(times),
            "memory": {key: getattr(memory, key) for key in (
                "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}}
OUT.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
