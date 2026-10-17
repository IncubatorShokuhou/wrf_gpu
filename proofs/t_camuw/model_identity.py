"""Canonical actual-CPU CAM loop + captured constant bytes for validation reuse.

Uses production loader and CPU facade without monkeypatches. This is a
regression/source-composition proof; the original CPU-WRF forecast is fidelity.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root, case, output = (Path(s).resolve() for s in sys.argv[1:4])
sys.path.insert(0, str(root / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform == "cpu"
assert Path(om.__file__).resolve().is_relative_to(root)
t0 = time.perf_counter()
cfg = NestedPipelineConfig(case, output.parent / "identity_out", output.parent / "identity_proof", hours=4, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
nl, carry = bundles["d01"].namelist, carries["d01"]
clock = om.build_clock_base(nl)
loaded = time.perf_counter()
initial_arrays = []
for value in jax.tree.leaves(carry):
    a = np.asarray(value)
    assert a.dtype.kind != "O"
    initial_arrays.append(dict(dtype=a.dtype.str, shape=list(a.shape),
                               sha256=hashlib.sha256(a.tobytes()).hexdigest()))
if "--initial-only" in sys.argv[4:]:
    output.write_text(json.dumps(dict(source=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        platform="cpu", gpu_executed=False, initial_arrays=initial_arrays,
        load_s=loaded-t0, scope="initial carry identity only; no model step or trace"), indent=2) + "\n")
    print("initial carry arrays", len(initial_arrays), "load_s", loaded-t0)
    sys.exit(0)
program = jax.make_jaxpr(lambda c: om._advance_chunk_fori(
    c, nl, jnp.int32(1), clock, n_steps=jnp.int32(1), cadence=jnp.int32(nl.radiation_cadence_steps)))(carry)
text = re.sub(r"name_and_src_info=[^\]\n]*", "", str(program))
text = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)
constants = []
for value in jax.tree.leaves(program.consts):
    a = np.asarray(value)
    constants.append(dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest()))
receipt = dict(source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    src_tree=subprocess.check_output(["git", "rev-parse", "HEAD:src/gpuwrf"], cwd=root, text=True).strip(),
    platform="cpu", gpu_executed=False, bl=nl.bl_pbl_physics, mp=nl.mp_physics,
    disable_guards=nl.disable_guards, jaxpr_sha256=hashlib.sha256(text.encode()).hexdigest(),
    constants=constants, initial_arrays=initial_arrays, chars=len(text), load_s=loaded-t0, trace_s=time.perf_counter()-loaded,
    scope="actual CPU CAM regression/source composition; no new forecast or GPU evidence")
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k:v for k,v in receipt.items() if k not in ("constants", "initial_arrays")}))
