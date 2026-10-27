"""Actual Swiss production root: two W2 values, identical trace and cheap key.

CPU initialization, native-at-trace only. Compiled execution is separately
tested by program.py and the eventual manager-window GPU root arm.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
root = str(Path(a.repo).resolve())
sys.path[:0] = [root + "/src", root + "/proofs/o1_nlbind"]
import jax
import numpy as np
import cpu_safe
cpu_safe.install()
from jax.experimental import pallas as pl
from gpuwrf.backends import pallas as facade
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import aot_cheap_key as ck, operational_mode as om
from gpuwrf.surface_params import controls as sc

assert jax.devices()[0].platform == "cpu"
assert str(Path(om.__file__).resolve()).startswith(root)
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
original_call = pl.pallas_call


def interpret(*args, **kwargs):
    kwargs["interpret"] = True
    kwargs.pop("compiler_params", None)
    return original_call(*args, **kwargs)


case = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams") as tmp:
    facade.cuda_native = lambda: False
    pl.pallas_call = interpret
    cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    try:
        _, bundles, _, _, _, carries = pipeline._load_domains(cfg, ("d01",))
    finally:
        facade.cuda_native = lambda: True
        pl.pallas_call = original_call
base, carry = bundles["d01"].namelist, carries["d01"]
om._resolve_operational_suite(base)
cadence = int(base.radiation_cadence_steps)
out = {"source": root, "platform": "cpu", "gpu_executed": False,
       "scope": "actual Swiss production native trace and actual AOT call key; no step execution", "arms": []}
for factor in (1., 1.1):
    multipliers = {name: factor for name in sc.LSM_CONTROLS if name != "rsmin"}
    multipliers["snow_emis"] = 1. if factor == 1. else .99
    values = sc.expand_controls(base, multipliers, land_state=carry.noahmp_land)
    nl = dataclasses.replace(base, surface_controls=values)
    cb = om.build_clock_base(nl)
    kwargs = {"n_steps": 1, "cadence": cadence}
    key = ck.cheap_key(om._advance_chunk_fori, (carry, nl, jax.numpy.asarray(1, jax.numpy.int32), cb), kwargs, nl)
    assert key is not None
    jp = str(jax.make_jaxpr(lambda c, n, clock: om._advance_chunk_fori(c, n, 1, clock, **kwargs))(carry, nl, cb))
    canonical = re.sub(r"name_and_src_info=[^\]\n]*", "", jp)
    canonical = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", canonical)
    out["arms"].append({"factor": factor, "cheap_key": key, "control_digest": sc.controls_digest(values),
                        "signature": sc.signature(values),
                        "canonical_sha256": hashlib.sha256(canonical.encode()).hexdigest(), "chars": len(canonical)})
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out), flush=True)
assert out["arms"][0]["control_digest"] != out["arms"][1]["control_digest"]
assert out["arms"][0]["cheap_key"] == out["arms"][1]["cheap_key"]
assert out["arms"][0]["canonical_sha256"] == out["arms"][1]["canonical_sha256"]
# Exercise the actual CLI loader seam as well as the direct root call. Its
# validation must see the initialized carry without retaining it in the aux.
control_file = Path(a.out).with_suffix(".controls.npz")
sc.save_controls(control_file, {"d01": sc.prior_controls(base, list(multipliers), land_state=carry.noahmp_land)})
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams") as tmp:
    facade.cuda_native = lambda: False
    pl.pallas_call = interpret
    cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1,
                                       surface_controls=control_file)
    try:
        _, bound, _, _, _, bound_carries = pipeline._load_domains(cfg, ("d01",))
    finally:
        facade.cuda_native = lambda: True
        pl.pallas_call = original_call
loaded = bound["d01"].namelist
assert loaded.noahmp_land is None
assert sc.controls_digest(loaded.surface_controls) == out["arms"][0]["control_digest"]
out["loader_binding"] = {"surface_controls_digest": sc.controls_digest(loaded.surface_controls),
                         "production_namelist_land_is_none": True,
                         "carry_land_present": bound_carries["d01"].noahmp_land is not None}
Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
