"""Short original CPU-WRF tracer comparison through the real native loader.

Usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=0 python coupled_probe.py SRC OUT
Requires the lane's staged original CPU-WRF three-step case. Never writes refs.
"""
import hashlib
import gzip
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import sys
import time

SRC, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(SRC / "src"))
sys.path.insert(0, str(SRC / "proofs/o1_nlbind"))
import cpu_safe
cpu_safe.install()
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.contracts import state as sc
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.tracers.runtime import names
from jax.experimental import pallas as pl

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(str(SRC))
interpret = os.environ.get("TRACER_INTERPRET", "0") == "1"
if interpret:
    # CPU has no implementation of the GPU LayoutConstraint custom call.
    # Drop only physical-layout hints in this host execution harness, retaining
    # every arithmetic kernel, input and physics flag. This is not a GPU gate.
    from jax.experimental import layout
    layout.with_layout_constraint = lambda value, *args, **kwargs: value
    original_pallas_call = pl.pallas_call
    def interpret_call(*args,**kwargs):
        kwargs["interpret"] = True
        kwargs.pop("compiler_params",None)
        return original_pallas_call(*args,**kwargs)
    pl.pallas_call = interpret_call
else:
    assert os.environ["GPUWRF_FAST_DEFAULTS"] == "0"
sc._gpu_device = lambda: jax.devices("cpu")[0]
case = Path(os.environ.get("TRACER_CPU_CASE", "<USER_HOME>/wrf_gpu2_lanes/u2-tracers/TR01_original_cpu_case"))
result = {"source": str(SRC), "case": str(case), "backend": "cpu", "cache": "disabled",
          "scope": "3 steps/54 seconds original CPU-WRF comparison; no GPU/full-forecast claim", "frames": [],
          "interpret":interpret,"cpu_layout_constraints_omitted":interpret,
          "fast_defaults":os.environ.get("GPUWRF_FAST_DEFAULTS")}
t0 = time.monotonic()
cfg = pipeline.NestedPipelineConfig(case, OUT.parent / "unused_out", OUT.parent / "unused_proofs", hours=1,max_dom=1)
_, bundles, _, _, _, carries = pipeline._load_domains(cfg,("d01",))
nl, carry = bundles["d01"].namelist, carries["d01"]
off_control = os.environ.get("TRACER_OFF_CONTROL", "0") == "1"
if off_control:
    nl = replace(nl, tracer_opt=0, tracer_config=None)
    carry = carry.replace(state=carry.state.replace(tracers=None, tracer_bdy=None, tracer_budget=None),
                          tracer_tendencies=None)
result["tracers_off_control"] = off_control
comparison = os.environ.get("TRACER_COMPARE_FORECAST")
comparison = json.loads(Path(comparison).read_text()) if comparison else None
# The production _advance_chunk wrapper performs this once at run admission.
# A direct compiled call needs the same protection for aliased initial saves.
carry = om.unalias_donated_carry(carry, by_buffer=True)
jax.block_until_ready(carry)
om._resolve_operational_suite(nl)
clock = om.build_clock_base(nl)
result["load_s"] = time.monotonic()-t0
result["selectors"] = {k: getattr(nl,k) for k in ("tracer_opt","tracer_adv_opt","tracer_pblmix","diff_opt","km_opt","diff_6th_opt","dt_s")}
requested_steps = int(os.environ.get("TRACER_MAX_STEPS", "3"))
if requested_steps not in range(4):
    raise ValueError("TRACER_MAX_STEPS must be 0..3 for the staged reference")
result["requested_steps"] = requested_steps
result["scope"] = f"{requested_steps} steps/{requested_steps * nl.dt_s:g} seconds original CPU-WRF comparison; no GPU/full-forecast claim"
initial_reference = os.environ.get("TRACER_COMPARE_INITIAL_CARRY")
if initial_reference:
    with gzip.open(initial_reference, "rb") as stream:
        original_carry, _ = pickle.load(stream)
    new_leaves = jax.tree.leaves(carry)
    old_leaves = jax.tree.leaves(original_carry)
    assert len(new_leaves) == len(old_leaves), (len(new_leaves), len(old_leaves))
    initial_records = []
    for index, (new, old) in enumerate(zip(new_leaves, old_leaves)):
        a, b = np.asarray(new), np.asarray(old)
        equal = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        initial_records.append({"leaf": index, "shape": list(a.shape), "dtype": a.dtype.str,
                                "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "byte_equal": equal})
    result["initial_carry_reuse"] = {"reference": initial_reference,
        "reference_sha256": hashlib.sha256(Path(initial_reference).read_bytes()).hexdigest(),
        "leaf_count": len(initial_records), "all_byte_equal": all(r["byte_equal"] for r in initial_records),
        "records": initial_records,
        "scope": "argument-byte regression proof; program/constants identity checked separately"}
    assert result["initial_carry_reuse"]["all_byte_equal"], "initial carry changed; numerical reuse rejected"
refs = sorted(case.glob("wrfout_d01_*"))
assert len(refs) == 4
program = None
for step in range(requested_steps + 1):
    start = time.monotonic()
    if step:
        if program is None:
            lowered = om._advance_chunk_fori.lower(carry,nl,jnp.asarray(step,jnp.int32),clock,
                n_steps=1,cadence=int(nl.radiation_cadence_steps))
            result["lowered_hlo_sha256"] = hashlib.sha256(
                lowered.compiler_ir("hlo").as_hlo_text().encode()).hexdigest()
            program = lowered.compile()
            result["compile_s"] = time.monotonic()-start
            try:
                stats=program.memory_analysis()
                result["cpu_compiled_memory_bytes"]={key:getattr(stats,key) for key in
                    ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")}
            except (AttributeError,NotImplementedError) as error:
                result["cpu_compiled_memory_bytes"]={"status":"UNKNOWN","reason":str(error)}
            OUT.write_text(json.dumps(result,indent=2)+"\n")
        carry = program(carry,nl,jnp.asarray(step,jnp.int32),clock,
                        n_steps=1,cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(carry)
    fields = (np.asarray(carry.state.tracers) if carry.state.tracers is not None
              else np.empty((0, *carry.state.qv.shape), np.float32))
    frame = {"step": step, "elapsed_s": time.monotonic()-start, "nonfinite": int((~np.isfinite(fields)).sum()), "tracers": {}}
    if os.environ.get("TRACER_SAVE_FRAMES", "0") == "1":
        # Observation is outside the compiled forecast. Preserve its exact
        # program while retaining all operands for source/transport diagnosis.
        saved = OUT.with_name(OUT.stem + f"_frame{step}.pickle.gz")
        with gzip.open(saved, "wb", compresslevel=1) as handle:
            pickle.dump((jax.device_get(carry), nl), handle, protocol=5)
        frame["saved_carry"] = str(saved)
        frame["saved_carry_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
    with Dataset(refs[step]) as ds:
        for i,name in enumerate(names(nl)):
            truth = np.asarray(ds.variables[name][0])
            delta = fields[i].astype(np.float64)-truth.astype(np.float64)
            index = tuple(int(k) for k in np.unravel_index(np.argmax(np.abs(delta)), delta.shape))
            frame["tracers"][name] = {"max_abs":float(np.max(np.abs(delta))),
                "max_index_zyx":index, "reference_at_max":float(truth[index]),
                "port_at_max":float(fields[i][index]),
                "level_max_abs":np.max(np.abs(delta), axis=(1,2)).tolist(),
                "rmse":float(np.sqrt(np.mean(delta**2))), "reference_nonfinite":int((~np.isfinite(truth)).sum()),
                "cpu_min":float(truth.min()),"port_min":float(fields[i].min()),
                "cpu_max":float(truth.max()),"port_max":float(fields[i].max())}
        frame["weather_vs_original_cpu"] = {}
        for field, reference in (("u", ("U",)), ("v", ("V",)), ("w", ("W",)),
                ("mu_total", ("MU", "MUB")), ("ph_total", ("PH", "PHB")), ("qv", ("QVAPOR",))):
            truth = np.asarray(ds.variables[reference[0]][0], np.float32)
            for part in reference[1:]:
                truth = truth + np.asarray(ds.variables[part][0], np.float32)
            value = np.asarray(getattr(carry.state, field))
            delta = value.astype(np.float64) - truth.astype(np.float64)
            frame["weather_vs_original_cpu"][field] = dict(max_abs=float(np.max(np.abs(delta))),
                rmse=float(np.sqrt(np.mean(delta**2))))
    if comparison is not None:
        with gzip.open(comparison["frames"][step]["saved_carry"], "rb") as stream:
            other, _ = pickle.load(stream)
        checked, unequal = [], []
        for field in carry.state.active_field_names():
            if field in ("tracers", "tracer_bdy", "tracer_budget"):
                continue
            a, b = np.asarray(getattr(carry.state, field)), np.asarray(getattr(other.state, field))
            checked.append(field)
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                unequal.append(field)
        frame["on_off_state_regression"] = dict(checked=len(checked), unequal=unequal,
            byte_equal=not unequal, scope="regression only, not WRF fidelity")
    leaves = [np.asarray(x) for x in jax.tree.leaves(carry)]
    frame["whole_carry_nonfinite"] = sum(int((~np.isfinite(x)).sum()) for x in leaves if np.issubdtype(x.dtype,np.floating))
    if carry.census is not None:
        frame["guards"] = int(np.asarray(carry.census.guards).sum())
    result["frames"].append(frame)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(frame),flush=True)
result["wall_s"] = time.monotonic()-t0
result["pass"] = all(f["nonfinite"]==0 and f["whole_carry_nonfinite"]==0 and
    f.get("guards",0)==0 and all(t["reference_nonfinite"]==0 and t["max_abs"]<=1e-4 for t in f["tracers"].values()) for f in result["frames"])
OUT.write_text(json.dumps(result,indent=2)+"\n")
assert result["pass"], "original CPU-WRF tracer gate failed (fixed maxabs1e-4)"
