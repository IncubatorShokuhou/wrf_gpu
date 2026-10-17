"""Prove unchanged shared radiation programs without repeating numerical work.

Usage: CPU/QUIET/affinity wrapper, then shared_trace.py IMMUTABLE_TREE RECEIPT.
This is program-identity evidence only. Existing original-WRF qualification
and completed shared numerical checks remain the separate references.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
sys.path[:0] = [str(root / "src"), str(root / "tests")]
import jax
import numpy as np
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
import test_v013_operational_smoke as smoke

assert jax.devices()[0].platform == "cpu"
assert Path(om.__file__).resolve().is_relative_to(root / "src")
assert os.environ.get("GPUWRF_FAST_DEFAULTS") == "0"


def canonical(program):
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", str(program))
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


receipt = dict(source_commit=subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    mode="program identity, not numerical fidelity", backend="cpu",
    cache="disabled", affinity=sorted(os.sched_getaffinity(0)), cases={})
for sw in (2, 3):
    grid = smoke._grid()
    state = smoke._base_state(grid)
    nl = smoke._namelist(grid, mp_physics=0, bl_pbl_physics=0,
                         sf_sfclay_physics=0, cu_physics=0, ra_sw_physics=sw)
    smoke._resolve_operational_suite(nl)
    carry = initial_operational_carry(state)
    start = time.perf_counter()
    program = jax.make_jaxpr(lambda c: om._physics_step_forcing(
        c, nl, 0., run_radiation=True).carry.rthraten)(carry)
    text = canonical(program)
    constants = []
    for value in jax.tree.leaves(program.consts):
        array = np.asarray(value)
        constants.append(dict(dtype=array.dtype.str, shape=list(array.shape),
            sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest()))
    receipt["cases"][str(sw)] = dict(jaxpr_sha256=hashlib.sha256(text.encode()).hexdigest(),
        constants=constants, trace_s=time.perf_counter()-start,
        source_test="test_ra_sw_operational_runs_and_heats", nz=24, ny=4, nx=4)
    output.write_text(json.dumps(receipt, indent=2)+"\n")
    jax.clear_caches()
print(json.dumps(receipt))
