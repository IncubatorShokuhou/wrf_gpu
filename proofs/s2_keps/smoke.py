# TEMF lane adaptation of proofs/o1_nlbind/probe.py; artifacts stay in own lane.
"""CLI-binding probe on the real Swiss single-domain case (RD11, 42x42x44, dt 18 s).

usage: JAX_PLATFORMS=cpu python probe.py <src_root> <out.json> <arm> [<arm> ...]
  arm = "release" | "key=val,key=val" (edits of &physics/&dynamics in namelist.input)
For each arm a private case dir is built (symlinks to RD11/cpu + the edited namelist.input) and loaded through the
REAL CLI loader nested_pipeline._load_domains (the native single-root path), so the arm proves the binding itself.
Recorded per arm: the bound OperationalNamelist fields, the location-stripped (E151) jaxpr sha of the production
step loop operational_mode._advance_chunk_fori (1 step), or the trace/load error.  With O1_EXEC_STEPS=N>0 the arm
also EXECUTES N production steps (native facade dispatches to its CPU fallback).
Use GPUWRF_CARRY_DONATE=0: this direct harness retains the step-entry carry for
delta reporting and does not use the public runner's donation preparation.
and records non-finite counts + census guards.  tridiagonal_solve -> pure-JAX Thomas in every arm (E105).
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

SRC, OUT, ARMS = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, str(Path(SRC) / "proofs/o1_nlbind"))
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path(os.environ.get("O1_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s2-keps")
STEPS = int(os.environ.get("O1_EXEC_STEPS", "0"))
# O1_INTERPRET_EXEC=1: keep every pallas_call in interpret mode for load, trace AND execution, so the release
# defaults (Triton kernels) can EXECUTE on CPU; the traced sha then differs from the Triton release sha by design.
INTERPRET_EXEC = os.environ.get("O1_INTERPRET_EXEC", "0") == "1"
FIELDS = ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "ra_lw_physics", "ra_sw_physics", "cu_physics",
          "sf_surface_physics", "use_noahmp", "sf_urban_physics", "diff_opt", "km_opt", "epssm", "damp_opt", "zdamp",
          "dampcoef", "w_damping", "diff_6th_opt", "diff_6th_factor", "moist_adv_opt", "scalar_adv_opt",
          "topo_shadow_length_m", "cumulus_cadence_steps", "radiation_cadence_steps")
_real_pallas_call = pl.pallas_call


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


def canon(jaxpr) -> str:
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


def array_fingerprint(tree):
    digest=hashlib.sha256()
    leaves=jax.tree.leaves(tree)
    for index,leaf in enumerate(leaves):
        value=np.asarray(leaf)
        digest.update(repr((index,value.shape,value.dtype.str)).encode())
        digest.update(value.tobytes(order='C'))
    return dict(sha256=digest.hexdigest(),leaves=len(leaves))


def edit_namelist(text: str, edits: dict) -> str:
    groups = {}
    for key, val in edits.items():
        group = "dynamics" if key in {"diff_opt", "km_opt", "epssm", "damp_opt", "zdamp", "dampcoef", "w_damping",
                                      "diff_6th_opt", "diff_6th_factor", "moist_adv_opt", "scalar_adv_opt",
                                      "khdif", "kvdif", "c_s", "c_k", "top_lid"} else "physics"
        pat = re.compile(rf"^\s*{key}\s*=.*$", re.M | re.I)
        line = f" {key} = {val},"
        if pat.search(text):
            text = pat.sub(line, text)
        else:
            groups.setdefault(group, []).append(line)
    for group, lines in groups.items():
        text = re.sub(rf"(&{group}\s*\n)", lambda m: m.group(1) + "\n".join(lines) + "\n", text, count=1, flags=re.I)
    return text


out = {"src": SRC, "om": om.__file__, "case": str(CASE), "exec_steps": STEPS, "interpret_exec": INTERPRET_EXEC,
       "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset=1>"), "arms": {}}
for arm in ARMS:
    edits = {} if arm == "release" else dict(kv.split("=", 1) for kv in arm.split(","))
    rec = {"edits": edits}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(dir=LANE) as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            for f in CASE.iterdir():
                if f.name != "namelist.input":
                    (case / f.name).symlink_to(f)
            (case / "namelist.input").write_text(edit_namelist((CASE / "namelist.input").read_text(), edits))
            cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
            if STEPS == 0 or INTERPRET_EXEC:
                pl.pallas_call = _interpret_pallas_call
            try:
                _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
            finally:
                if not INTERPRET_EXEC:
                    pl.pallas_call = _real_pallas_call
        ns, carry = bundles["d01"].namelist, carries["d01"]
        rec["bound"] = {f: (getattr(ns, f) if not isinstance(getattr(ns, f), float) else float(getattr(ns, f)))
                        for f in FIELDS}
        rec['transport_dtypes']={k:str(getattr(carry.state,k).dtype) for k in ('theta','qv','qc','qi','qke')}
        rec['boundary']={k:getattr(ns.boundary_config,k) for k in ('spec_zone','have_bcs_moist','have_bcs_scalar')}
        if os.environ.get('O1_QC_PACKET_INDEX'):
            # Proof-only fixed-point accounting on the existing native packet
            # path. This ON program is diagnostic, not production-equivalent.
            from gpuwrf.diagnostics import f2_scalar_pd_packet as packet_module
            import jax.numpy as jnp
            base_capture=packet_module.capture
            target=int(os.environ['O1_QC_PACKET_INDEX'])
            def fixed_capture(packet,smaller,context,step,index,post_rk):
                return base_capture(packet,step==3,context,step,jnp.int32(target),post_rk)
            packet_module.capture=fixed_capture
            rec['fixed_qc_packet_index']=target
        if os.environ.get('O1_SOURCE_SEAL')=='1':
            rec['input_arrays']=array_fingerprint(carry)
        rec["load_s"] = round(time.time() - t0, 1)
        om._resolve_operational_suite(ns)
        jax.clear_caches()
        cb = om.build_clock_base(ns)
        t1 = time.time()
        jx = jax.make_jaxpr(lambda c, nl=ns, cb=cb: om._advance_chunk_fori(
            c, nl, 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps)))(carry)
        text = canon(jx)
        rec["sha"] = hashlib.sha256(text.encode()).hexdigest()
        rec["chars"] = len(text)
        rec["trace_s"] = round(time.time() - t1, 1)
        if os.environ.get('O1_SOURCE_SEAL')=='1':
            rec['closed_constants']=array_fingerprint(jx.consts)
            hlo=om._advance_chunk_fori.lower(carry,ns,1,cb,n_steps=1,
                cadence=int(ns.radiation_cadence_steps)).compiler_ir('hlo').as_hlo_text()
            rec['lowered_hlo_sha256']=hashlib.sha256(hlo.encode()).hexdigest()
        out['arms'][arm]=rec
        Path(OUT).write_text(json.dumps(out,indent=1,default=str))
        print(arm,{'phase':'trace-complete','load_s':rec['load_s'],'trace_s':rec['trace_s'],'sha':rec['sha']},flush=True)
        if STEPS:
            t2 = time.time()
            c = carry
            capture_arrays=os.environ.get('O1_PD_ARRAYS')
            if capture_arrays:
                from pd_capture import make_step
                diagnostic_step=make_step(om,ns,cb)
                rec['diagnostic_scope']='actual PD input/output arrays; CPU diagnostic ON, not production-equivalent'
            for i in range(STEPS):
                if capture_arrays:
                    c,pd_arrays=diagnostic_step(c,np.int32(i+1))
                else:
                    c = om._advance_chunk_fori(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
                jax.block_until_ready(c)
            if capture_arrays:
                np.savez(capture_arrays,**{k:np.asarray(v) for k,v in pd_arrays.items()})
                rec['pd_arrays_path']=capture_arrays
                rec['pd_arrays_sha256']=hashlib.sha256(Path(capture_arrays).read_bytes()).hexdigest()
            leaves = [np.asarray(x) for x in jax.tree.leaves(c)]
            rec["nonfinite_values"] = int(sum(int((~np.isfinite(x)).sum()) for x in leaves
                                              if np.issubdtype(x.dtype, np.floating)))
            census = getattr(c, "census", None)
            if census is not None:
                rec["guard_total"] = int(np.asarray(census.guards).sum())
                from gpuwrf.diagnostics.census import GUARDS,EVENTS,F2_PHASES,F2_FIELDS,F2_EVENTS
                counts=np.asarray(census.guards)
                rec['guards']={guard:{event:int(counts[g,e]) for e,event in enumerate(EVENTS) if counts[g,e]}
                               for g,guard in enumerate(GUARDS) if counts[g].any()}
                f2=np.asarray(census.f2_counts)
                rec['f2_events']=[dict(phase=phase,field=field,event=event,count=int(f2[p,f,e]),
                    step=int(np.asarray(census.f2_first_step)[p,f,e]),
                    index=int(np.asarray(census.f2_first_index)[p,f,e]),
                    value=float(np.asarray(census.f2_first_value)[p,f,e]),
                    min_negative=float(np.asarray(census.f2_min_negative)[p,f]))
                    for p,phase in enumerate(F2_PHASES) for f,field in enumerate(F2_FIELDS)
                    for e,event in enumerate(F2_EVENTS) if f2[p,f,e]]
                if census.f2_qc_packet is not None:
                    from gpuwrf.diagnostics.f2_scalar_pd_packet import HEADERS,VALUES
                    packet=census.f2_qc_packet
                    rec['qc_packet']=dict(header=dict(zip(HEADERS,map(int,np.asarray(packet.header)),strict=True)),
                        values=dict(zip(VALUES,map(float,np.asarray(packet.values)),strict=True)),
                        clocks=np.asarray(packet.clocks).view(np.float64).tolist(),
                        scope='CPU diagnostic ON accounting; original-WRF operator audit required; not production-equivalent')
            s0, s1 = carry.state, c.state
            rec["delta"] = {f: float(np.abs(np.asarray(getattr(s1, f), np.float64)
                                            - np.asarray(getattr(s0, f), np.float64)).max())
                            for f in ("theta", "qv", "u", "w")}
            rec["exec_s"] = round(time.time() - t2, 1)
            if c.state.epsilon_pbl is not None:
                from gpuwrf.contracts.precision import EPSILON_PBL_LEAVES
                from gpuwrf.nesting.scalar_species import epsilon_pbl_scalars
                active={k:getattr(c.state,k) for k in EPSILON_PBL_LEAVES if getattr(c.state,k) is not None}
                rec["epsilon_nonzero"]={k:int(np.count_nonzero(np.asarray(v))) for k,v in active.items()}
                rec["epsilon_bytes"]=sum(x.size*x.dtype.itemsize for x in active.values())
                rec["epsilon_all_real32"]=all(x.dtype==np.float32 for x in active.values() if np.issubdtype(x.dtype,np.floating))
                rec["call_counter"]=int(c.state.epsilon_pbl_step)
                rec["scalar_delta"]={k:float(np.max(np.abs(np.asarray(getattr(c.state,k))-np.asarray(getattr(carry.state,k))))) for k in epsilon_pbl_scalars(ns.bl_pbl_physics)}

    except Exception as exc:  # noqa: BLE001 - an arm that cannot load/trace/run is a RESULT
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = round(time.time() - t0, 1)
    out["arms"][arm] = rec
    Path(OUT).write_text(json.dumps(out, indent=1, default=str))
    print(arm, {k: v for k, v in rec.items() if k not in ("traceback_tail", "bound")}, flush=True)

assert all('error' not in rec for rec in out['arms'].values()),'trace/execution failure; see receipt'
if STEPS:
    assert all(rec['nonfinite_values']==0 and rec.get('guard_total')==0 and
               rec.get('epsilon_all_real32',True) for rec in out['arms'].values()),'CPU coupling gate failed'
