"""o2-fdda probe (from proofs/o1_nlbind/probe.py) on the real Swiss single-domain case (RD11, 42x42x44, dt 18 s).

Adds &fdda edits and a real-structured wrffdda_d01 (proofs/o2_fdda/make_wrffdda.py, O2_WRFFDDA) in the case dir.

usage: JAX_PLATFORMS=cpu python probe.py <src_root> <out.json> <arm> [<arm> ...]
  arm = "release" | "key=val,key=val" (edits of &physics/&dynamics in namelist.input)
For each arm a private case dir is built (symlinks to RD11/cpu + the edited namelist.input) and loaded through the
REAL CLI loader nested_pipeline._load_domains (the native single-root path), so the arm proves the binding itself.
Recorded per arm: the bound OperationalNamelist fields, the location-stripped (E151) jaxpr sha of the production
step loop operational_mode._advance_chunk_fori (1 step), or the trace/load error.  With O1_EXEC_STEPS=N>0 the arm
also EXECUTES N production steps (needs GPUWRF_FAST_DEFAULTS=0: release Triton kernels cannot run on CPU, E156)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
if os.environ.get("O2_MUTANT") == "no_qv":
    # Coupled-level deletion check: drop the FDDA qv source (moist_tend(P_QV) and its conv_t_tendf_to_moist term).
    from gpuwrf.coupling import fdda_rk as _fr
    _orig_sources = _fr.fdda_step_sources

    def _no_qv(*a, **k):
        carry, (ru, rv, t, q) = _orig_sources(*a, **k)
        return carry, (ru, rv, t, q * 0)

    _fr.fdda_step_sources = _no_qv
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path(os.environ.get("O1_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o2-fdda")
WRFFDDA = Path(os.environ.get("O2_WRFFDDA", "<USER_HOME>/wrf_gpu2_lanes/o2-fdda/case/wrffdda_d01"))
FDDA_KEYS = {"grid_fdda", "gfdda_interval_m", "gfdda_end_h", "guv", "gt", "gq", "fgdt", "fgdtzero", "if_ramping",
             "dtramp_min", "if_no_pbl_nudging_uv", "if_no_pbl_nudging_t", "if_no_pbl_nudging_q", "if_zfac_uv",
             "k_zfac_uv", "if_zfac_t", "k_zfac_t", "if_zfac_q", "k_zfac_q", "io_form_gfdda", "gfdda_inname"}
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


def edit_namelist(text: str, edits: dict) -> str:
    groups = {}
    for key, val in edits.items():
        group = "fdda" if key in FDDA_KEYS else "dynamics" if key in {"diff_opt", "km_opt", "epssm", "damp_opt", "zdamp", "dampcoef", "w_damping",
                                      "diff_6th_opt", "diff_6th_factor", "moist_adv_opt", "scalar_adv_opt",
                                      "khdif", "kvdif", "c_s", "c_k", "top_lid"} else "physics"
        pat = re.compile(rf"^\s*{key}\s*=.*$", re.M | re.I)
        line = f" {key} = {val},"
        if pat.search(text):
            text = pat.sub(line, text)
        else:
            groups.setdefault(group, []).append(line)
    for group, lines in groups.items():
        if not re.search(rf"&{group}\s*\n", text, flags=re.I):
            text = text.rstrip() + f"\n&{group}\n/\n"
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
            (case / "wrffdda_d01").symlink_to(WRFFDDA)
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
        rec["load_s"] = round(time.time() - t0, 1)
        rec["data_assimilation"] = type(ns.data_assimilation).__name__
        fdda_carry = getattr(carry, "fdda", None)  # absent on pre-o2 trees (main baseline arms)
        rec["carry_fdda"] = None if fdda_carry is None else {k: str(getattr(fdda_carry, k).dtype) for k in
                                                              ("u_old", "rundgdten")}
        if fdda_carry is not None:
            rec["fdda_options"] = repr(ns.data_assimilation.options)
            rec["record_steps"] = list(ns.data_assimilation.record_steps)
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
        if STEPS:
            t2 = time.time()
            c = carry
            for i in range(STEPS):
                c = om._advance_chunk_fori(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
                jax.block_until_ready(c)
            leaves = [np.asarray(x) for x in jax.tree.leaves(c)]
            rec["nonfinite_values"] = int(sum(int((~np.isfinite(x)).sum()) for x in leaves
                                              if np.issubdtype(x.dtype, np.floating)))
            census = getattr(c, "census", None)
            if census is not None:
                rec["guard_total"] = int(np.asarray(census.guards).sum())
            s0, s1 = carry.state, c.state
            rec["delta"] = {f: float(np.abs(np.asarray(getattr(s1, f), np.float64)
                                            - np.asarray(getattr(s0, f), np.float64)).max())
                            for f in ("theta", "qv", "u", "w")}
            rec["exec_s"] = round(time.time() - t2, 1)
            np.savez(LANE / f"probe_final_{abs(hash(arm)) % 10**8}.npz", theta=np.asarray(s1.theta),
                     u=np.asarray(s1.u), v=np.asarray(s1.v), qv=np.asarray(s1.qv))
            rec["final_npz"] = str(LANE / f"probe_final_{abs(hash(arm)) % 10**8}.npz")
            if getattr(c, "fdda", None) is not None:
                rec["fdda_held_absmax"] = {k: float(np.abs(np.asarray(getattr(c.fdda, k))).max())
                                           for k in ("rundgdten", "rvndgdten", "rthndgdten", "rqvndgdten")}
    except Exception as exc:  # noqa: BLE001 - an arm that cannot load/trace/run is a RESULT
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = round(time.time() - t0, 1)
    out["arms"][arm] = rec
    Path(OUT).write_text(json.dumps(out, indent=1, default=str))
    print(arm, {k: v for k, v in rec.items() if k not in ("traceback_tail", "bound")}, flush=True)
