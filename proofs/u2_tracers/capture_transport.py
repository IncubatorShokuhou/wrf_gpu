"""Proof-only per-RK tracer operands, admitted only after whole-carry BYTE gate.

An observer can change compiled arithmetic (E205). Reject every captured rate
unless the complete first-step carry equals the separately saved production
carry byte for byte. No GPU or performance claim; no in-loop host transfers.
"""
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

root, forecast, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
import cpu_safe
cpu_safe.install()
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import layout, pallas
from gpuwrf.runtime import operational_mode as om
from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
from gpuwrf.dynamics import flux_advection
from gpuwrf.tracers import runtime

assert jax.devices()[0].platform == "cpu"
layout.with_layout_constraint = lambda value, *args, **kwargs: value
original_pallas = pallas.pallas_call
def interpret(*args, **kwargs):
    kwargs["interpret"] = True
    kwargs.pop("compiler_params", None)
    return original_pallas(*args, **kwargs)
pallas.pallas_call = interpret
receipt = json.loads(forecast.read_text())
def read_frame(index):
    with gzip.open(receipt["frames"][index]["saved_carry"], "rb") as stream:
        return pickle.load(stream)
initial, nl = read_frame(0)
expected, _ = read_frame(1)
initial = om.unalias_donated_carry(jax.device_put(initial), by_buffer=True)
last_only = os.environ.get("TRACER_CAPTURE_LAST", "0") == "1"
if not last_only:
    nl = replace(nl, metrics=real_metrics(nl.metrics))
clock = om.build_clock_base(nl)
records, active = [], False
active_base = None
original_advect = flux_advection.advect_moisture_scalars
original_stage = runtime.stage_update

def advect(fields, old, vel, **kw):
    result = original_advect(fields, old, vel, **kw)
    if active:
        records.append(dict(q=fields[0], qold=(fields[0] if old is None else old[0]),
            adv=result[0], ru=vel.ru_full, rv=vel.rv_full, rom=vel.rom,
            **{key: kw[key] for key in ("mut", "mu_old", "c1", "c2", "rdzw", "fzm", "fzp", "msfty")}))
    return result

def stage(state, origin, frozen, namelist, **kw):
    global active
    selected = not last_only or int(kw["rk_step"]) == 3
    active = selected
    result = original_stage(state, origin, frozen, namelist, **kw)
    active = False
    if selected:
        records[-1].update(frozen=frozen[0], result=result.tracers[0],
            old_pert=origin.mu_perturbation, mub=active_base,
            dt_rk=jnp.float32(kw["dt_rk"]))
    return result
flux_advection.advect_moisture_scalars = advect
runtime.stage_update = stage

def one(carry, namelist, index, clock_base, cadence):
    global active_base
    active_base = carry.base_state.mub
    records.clear()
    result = om._physics_boundary_step(om._layout_pin("carry", carry), namelist, index,
        run_radiation=jnp.equal(jnp.mod(index - 1, cadence), 0), clock_base=clock_base)
    assert len(records) == (1 if last_only else 3)
    return om._layout_pin("carry", result), tuple(records)

def observed(carry, namelist, index, clock_base, n_steps, cadence):
    if last_only:
        # Match production's real_metrics cast inside the compiled call. Build
        # observer shapes directly rather than tracing a dead full physics step.
        namelist = replace(namelist, metrics=real_metrics(namelist.metrics))
        volume = carry.state.qv
        plane = carry.state.mu_total
        prototypes = {key:volume for key in ("q", "qold", "adv", "frozen", "result")}
        prototypes.update({key:plane for key in ("mut", "mu_old", "old_pert", "mub", "msfty")})
        prototypes.update(ru=carry.state.u, rv=carry.state.v, rom=carry.state.w,
            dt_rk=jnp.float32(0), **{key:getattr(namelist.metrics, name) for key,name in
                (("c1","c1h"),("c2","c2h"),("rdzw","rdnw"),("fzm","fnm"),("fzp","fnp"))})
        empty = ({key:jnp.zeros_like(value, dtype=jnp.float32) for key,value in prototypes.items()},)
    else:
        _, example = one(carry, namelist, index, clock_base, cadence)
        empty = jax.tree.map(jnp.zeros_like, example)
    return jax.lax.fori_loop(0, n_steps,
        lambda i, pair: one(pair[0], namelist, index + i, clock_base, cadence), (carry, empty))

result, captured = jax.jit(observed, donate_argnums=(0,) if om._CARRY_DONATE else ())(
    initial, nl, jnp.int32(1), clock, jnp.int32(1), jnp.int32(nl.radiation_cadence_steps))
jax.block_until_ready(result)
actual_leaves, structure = jax.tree.flatten(result)
expected_leaves, expected_structure = jax.tree.flatten(expected)
assert structure == expected_structure
unequal = [i for i, (a,b) in enumerate(zip(actual_leaves,expected_leaves,strict=True))
           if np.asarray(a).dtype != np.asarray(b).dtype or np.asarray(a).tobytes() != np.asarray(b).tobytes()]
out = dict(source=str(root), proof_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           gate="complete first-step production carry BYTE (E205)", leaves=len(actual_leaves),
           unequal_leaves=unequal, accepted=not unequal, backend="cpu", cache="NONE",
           stages=[3] if last_only else [1,2,3])
output.write_text(json.dumps(out,indent=2)+"\n")
assert not unequal, "observer changed production carry; captured rates REJECTED"
arrays = {f"rk{stage}_{key}":np.asarray(value) for stage,record in zip(out["stages"],captured,strict=True)
          for key,value in record.items()}
arrays.update({key:np.asarray(getattr(nl.metrics,key)) for key in ("msftx","msfty","msfux","msfuy","msfvx","msfvy")})
np.savez(output.with_suffix(".npz"), **arrays)
print(json.dumps(out),flush=True)
