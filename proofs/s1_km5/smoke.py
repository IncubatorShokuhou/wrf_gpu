"""CPU coupled km_opt=5 smoke on real Swiss d01 (42x42x44, dt18).

usage: smoke.py OUT.json [STEPS=3] [BL=5] [ARMS=km5]
Select GPUWRF_FAST_DEFAULTS=1 O1_PALLAS_INTERPRET=1 GPUWRF_CENSUS=1 for
release REAL32 carry. Pallas executes its real kernel bodies in interpret mode;
PTX REAL rounding is CPU-emulated, so this is not GPU binary qualification.
The carry is loaded with the release configuration; km5 uses its separate
zero-initialized LES TKE, the same cold-start seeding tested in the product.
"""
import dataclasses
import json
import sys
import tempfile
import re
import time
import os
import hashlib
from pathlib import Path

NATIVE_GPU = os.environ.get('K5_NATIVE_GPU') == '1'
RELEASE_CONTROL = os.environ.get('K5_RELEASE_CONTROL') == '1'
KM2_CONTROL = os.environ.get('K5_KM2_CONTROL') == '1'
assert not (RELEASE_CONTROL and KM2_CONTROL)
if NATIVE_GPU:
    if os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1':
        raise RuntimeError('K5 native diagnostic requires scripts/with_gpu_lock.sh')
    if os.environ.get('O1_PALLAS_INTERPRET') == '1':
        raise RuntimeError('K5 native diagnostic must use production GPU kernels')
    if os.environ.get('GPUWRF_FAST_DEFAULTS') != '1':
        raise RuntimeError('K5 native diagnostic requires release fast defaults')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'o1_smag3d'))
if not NATIVE_GPU:
    import cpu_safe  # noqa: E402
    cpu_safe.install()

if os.environ.get("O1_PALLAS_INTERPRET") == "1":
    # Release fast defaults (GPUWRF_FAST_DEFAULTS=1, REAL32 carry) select Triton Pallas kernels that cannot
    # execute on CPU (E156): run EVERY pallas_call in interpret mode (same kernel bodies, CPU semantics).
    from jax.experimental import pallas as _pl

    _real_pallas_call = _pl.pallas_call

    def _interpret_pallas_call(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return _real_pallas_call(*a, **k)

    _pl.pallas_call = _interpret_pallas_call
    # The release REAL kernels also pin REAL4 rounding with PTX inline asm ("<op>.rn.f32"/"<op>.<mode>.f64"),
    # which has no CPU lowering. CPU-execution shim (NOT the product instruction): exact op in f64, then
    # round to the result precision (the same emulation as boundary_apply's own CPU prescreen).
    import jax.experimental.pallas.triton as _pt

    def _cpu_inline_asm(asm, *, args, result_shape_dtypes, **_kw):
        op, mode, ptx = asm.split()[0].split(".")[:3]
        a, b = (jnp.asarray(x, jnp.float64) for x in args)
        exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
        out_dtype = result_shape_dtypes[0].dtype
        if jnp.dtype(out_dtype) == jnp.float32:
            exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
        return [exact.astype(out_dtype)]

    _pt.elementwise_inline_asm = _cpu_inline_asm
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == ('gpu' if NATIVE_GPU else 'cpu')
if not NATIVE_GPU:
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
OUT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BL = 0
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/s1-km5") as tmp:
    case=Path(tmp)/"case";case.mkdir()
    for f in CASE.iterdir():
        if f.name!="namelist.input":(case/f.name).symlink_to(f)
    text=(CASE/"namelist.input").read_text()
    edits={} if RELEASE_CONTROL else {"diff_opt":2,"km_opt":2 if KM2_CONTROL else 5,"bl_pbl_physics":0,"sf_sfclay_physics":5}
    for key,val in edits.items():
        pattern=rf"^\s*{key}\s*=.*$"
        assert re.search(pattern,text,re.M|re.I),key
        text=re.sub(pattern,f" {key} = {val},",text,flags=re.M|re.I)
    (case/"namelist.input").write_text(text)
    cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
ns, carry0 = bundles["d01"].namelist, carries["d01"]
assert carry0.census is not None, 'K5 coupled gate requires GPUWRF_CENSUS=1'
report = {"steps": STEPS, "bl_pbl_physics": int(ns.bl_pbl_physics), "dt": float(ns.dt_s),
          "platform": jax.devices()[0].platform, "native_gpu": NATIVE_GPU,
          "official_lock_held": os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1',
          "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "pallas_interpret": os.environ.get("O1_PALLAS_INTERPRET"),
          "use_flux_advection": bool(ns.use_flux_advection), "theta_dtype": str(carry0.state.theta.dtype), "u_dtype": str(carry0.state.u.dtype)}
report['disable_guards']=bool(ns.disable_guards)
report['jax_enable_x64']=bool(jax.config.x64_enabled)
report['metrics_c1h_dtype']=str(ns.metrics.c1h.dtype)
report['metrics_precision']=str(ns.metrics.precision)
report['xla_flags']=os.environ.get('XLA_FLAGS','')
report['namelist_sha256']=hashlib.sha256(text.encode()).hexdigest()
report['resolved']={n:getattr(ns,n) for n in ('diff_opt','km_opt','bl_pbl_physics',
    'mp_physics','cu_physics','sf_sfclay_physics','sf_surface_physics','dt_s',
    'radiation_cadence_steps')}
Path(OUT).write_text(json.dumps(report, indent=1))
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["km5"]
arms = {"release": ns} if RELEASE_CONTROL else {"km2": ns} if KM2_CONTROL else {"km5": ns}
if os.environ.get('K5_OFF_CONTROL')=='1':
    arms['off_same_physics']=dataclasses.replace(ns,diff_opt=0,km_opt=0)
if os.environ.get('K5_ARMS'):
    wanted=os.environ['K5_ARMS'].split(',')
    all_arms={'km5':ns,'off_same_physics':dataclasses.replace(ns,diff_opt=0,km_opt=0)}
    arms={n:all_arms[n] for n in wanted}
if RELEASE_CONTROL:
    assert (ns.diff_opt,ns.km_opt,ns.bl_pbl_physics,ns.sf_sfclay_physics)==(1,4,5,5)
elif KM2_CONTROL:
    assert (ns.diff_opt,ns.km_opt,ns.bl_pbl_physics,ns.sf_sfclay_physics)==(2,2,0,5)
else:
    assert (ns.diff_opt,ns.km_opt,ns.bl_pbl_physics,ns.sf_sfclay_physics)==(2,5,0,5)
finals = {}
for name, nl in arms.items():
    cb = om.build_clock_base(nl)
    # Own copy per arm: release defaults DONATE the carry (GPUWRF_CARRY_DONATE=1).
    carry = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry0)
    t0 = time.time()
    step_records=[]
    jax.eval_shape(lambda c:om._advance_chunk(c,nl,1,cb,n_steps=1,
        cadence=int(nl.radiation_cadence_steps)),carry)
    print('COUPLED_SHAPE_PREFLIGHT_PASS',flush=True)
    # Production entry _advance_chunk (unalias for donation + _advance_chunk_fori loop), one step per call.
    for i in range(STEPS):
        carry = om._advance_chunk(carry, nl, i + 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(carry)
        step_records.append({"step":i+1,"tke_max":None if carry.state.tke is None else float(np.asarray(carry.state.tke).max()),
                             "pblh_max":float(np.asarray(carry.state.pblh).max()),
                             "qc_min":float(np.asarray(carry.state.qc).min()),
                             "qc_max":float(np.asarray(carry.state.qc).max()),
                             "qc_negative_cells":int((np.asarray(carry.state.qc)<0).sum()),
                             "qc_over_cap_cells":int((np.asarray(carry.state.qc)>.05).sum()),
                             "guard_candidates":int(np.asarray(carry.census.guards).sum()) if carry.census is not None else None})
        from gpuwrf.diagnostics.census import f2_record, f2_negative_summary
        host = jax.device_get(carry.census)
        step_records[-1].update(f2=f2_record(host,carry.state.theta.shape),
                               negative_summary=f2_negative_summary(host,carry.state.theta.shape))
        report[name]={'partial':True,'steps_completed':i+1,'step_records':step_records}
        Path(OUT).write_text(json.dumps(report, indent=1))
        print(name,'STEP',{k:v for k,v in step_records[-1].items()
                           if k not in ('f2','negative_summary')},flush=True)
    leaves = jax.tree.leaves(carry)
    nonfinite = int(sum(int((~np.isfinite(np.asarray(x))).sum()) for x in leaves
                        if np.issubdtype(np.asarray(x).dtype, np.floating)))
    finals[name] = carry
    report[name] = {"wall_s": round(time.time() - t0, 1), "nonfinite_values": nonfinite, "leaves": len(leaves),"step_records":step_records,
                    "resolved":{n:getattr(nl,n) for n in ('diff_opt','km_opt','bl_pbl_physics','mp_physics','cu_physics','sf_sfclay_physics','sf_surface_physics','dt_s','radiation_cadence_steps')}}
    if getattr(carry, "census", None) is not None:
        from gpuwrf.diagnostics.census import EVENTS, GUARDS

        g = np.asarray(carry.census.guards)
        report[name]["guards_fired"] = {GUARDS[r]: dict(zip(EVENTS, map(int, g[r]))) for r in range(g.shape[0]) if g[r].any()}
        report[name]["guard_total"] = int(g.sum())
        from gpuwrf.diagnostics.census import f2_record,f2_negative_summary
        host=jax.device_get(carry.census)
        report[name]['f2']=f2_record(host,carry.state.theta.shape)
        report[name]['negative_summary']=f2_negative_summary(host,carry.state.theta.shape)
    print(name,{k:v for k,v in report[name].items()
                if k not in ('step_records','f2','negative_summary')},flush=True)
    Path(OUT).write_text(json.dumps(report, indent=1))

if 'km5' not in finals:
    report['diagnostic_only']=True
    assert all(report[n]['nonfinite_values']==0 for n in finals),report
    Path(OUT).write_text(json.dumps(report,indent=1))
    raise SystemExit(0)
carry=finals['km5']
assert report["km5"]["nonfinite_values"] == 0, report["km5"]["nonfinite_values"]
tke = np.asarray(carry.state.tke)
report["km5"].update(tke_min=float(tke.min()), tke_max=float(tke.max()), tke_active=int(np.count_nonzero(tke)), tke_dtype=str(tke.dtype))
assert np.isfinite(tke).all() and tke.max() > 0
Path(OUT).write_text(json.dumps(report, indent=1))

from gpuwrf.contracts.precision import SMS_CARRY_LEAVES
report["km5"]["sms_carry"]={n:{"dtype":str(getattr(carry.state,n).dtype),
    "nonzero":int(np.count_nonzero(np.asarray(getattr(carry.state,n))))} for n in SMS_CARRY_LEAVES}
assert all(v["dtype"]=="float32" for v in report["km5"]["sms_carry"].values())
assert report["km5"]["sms_carry"]["sms_wspd"]["nonzero"]>0
assert report["km5"]["sms_carry"]["sms_ustm"]["nonzero"]>0
assert any(report["km5"]["sms_carry"][n]["nonzero"]>0 for n in ('sms_br','sms_rmol'))
assert np.asarray(carry.state.pblh).max()>0
Path(OUT).write_text(json.dumps(report,indent=1))
assert report["km5"].get("guard_total", 0) == 0, report["km5"].get("guards_fired", {})
