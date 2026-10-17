"""CPU coupled smoke of mp_physics=10/14/16/26 on the real Swiss single-domain case (RD11, 42x42x44, dt 18 s), lane s1-numbers.

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 O1_PALLAS_INTERPRET=1 python smoke.py <src_root> <out.json> <steps> mp[,mp...]
Each arm builds a private case dir (symlinks to RD11 + namelist.input with only mp_physics edited), loads it through the
REAL CLI loader nested_pipeline._load_domains, and runs N production steps (om._advance_chunk).  Recorded: carry
finiteness, census guards, the root boundary route of the extras (root_scalar_rk1_split + _root_flow_only), and per extra
species its min/max and whether the outer spec ring was rewritten by flow_dep_bdy (ring vs interior copy check).
Release mode (as o1-smag3d): REAL32 carry/kernels, every pallas_call in interpret mode, the kernels' PTX "<op>.rn.f32"
inline asm emulated exactly-rounded in f64 -> REAL4.  Same kernel bodies, CPU instruction semantics -- NOT the GPU binary.
tridiagonal_solve is a pure-JAX Thomas scan (E105 CPU deadlock; not the product solver)."""
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

SRC, OUT, STEPS, MPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(m) for m in sys.argv[4].split(",")]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(SRC)/"proofs/o1_nlbind"))
import cpu_safe  # noqa: E402

cpu_safe.install()
if os.environ.get("O1_PALLAS_INTERPRET") == "1":
    from jax.experimental import pallas as _pl

    _real_pallas_call = _pl.pallas_call

    def _interpret_pallas_call(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return _real_pallas_call(*a, **k)

    _pl.pallas_call = _interpret_pallas_call
    import jax.experimental.pallas.triton as _pt

    def _cpu_inline_asm(asm, *, args, result_shape_dtypes, **_kw):
        op = asm.split()[0].split(".")[0]
        a, b = (jnp.asarray(x, jnp.float64) for x in args)
        exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
        out_dtype = result_shape_dtypes[0].dtype
        if jnp.dtype(out_dtype) == jnp.float32:
            exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
        return [exact.astype(out_dtype)]

    _pt.elementwise_inline_asm = _cpu_inline_asm
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.coupling import boundary_apply as bd  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu" and om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path(os.environ.get("O2_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
LANE = Path(os.environ.get("O2_LANE", "<USER_HOME>/wrf_gpu2_lanes/s1-numbers"))
report = {"src": SRC, "steps": STEPS, "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset=1>"),
          "pallas_interpret": os.environ.get("O1_PALLAS_INTERPRET"), "arms": {}}
for mp in MPS:
    rec = {}
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=LANE) as tmp:
        case = Path(tmp) / "case"
        case.mkdir()
        for f in CASE.iterdir():
            if f.name != "namelist.input":
                (case / f.name).symlink_to(f)
        text = re.sub(r"^\s*mp_physics\s*=.*$", f" mp_physics = {mp},", (CASE / "namelist.input").read_text(),
                      flags=re.M | re.I)
        (case / "namelist.input").write_text(text)
        cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
        _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
    ns, carry = bundles["d01"].namelist, carries["d01"]
    assert int(ns.mp_physics) == mp
    species = (om._diffused_scalar_species(ns) if hasattr(om, "_diffused_scalar_species") else
               om._advected_scalar_species(ns) + (("Ni", "Nr") if carry.state.Ni is not None else ()))
    relaxed, flow = bd.root_scalar_rk1_split(ns.boundary_config, species, **om._root_flow_only(ns))
    extras = {10: ("Ni", "Nr", "Ns", "Ng"), 14: ("Nc", "Nr", "Nn"),
              16: ("Nc", "Nr", "Nn"), 26: ("qh", "Nc", "Nr", "Nn")}[mp]
    rec.update(mp=mp, theta_dtype=str(carry.state.theta.dtype), root_rk1_active=bool(om._root_scalar_bdy_rk1_active(ns)),
               relaxed=relaxed, flow=flow, extras=extras, load_s=round(time.time() - t0, 1),
               disable_guards=bool(ns.disable_guards), ccn_conc=getattr(ns, "ccn_conc", 1e8))
    cb = om.build_clock_base(ns)
    s0 = carry.state
    # release defaults DONATE the carry: host copies of everything compared after stepping
    init = {n: np.asarray(getattr(s0, n), np.float64).copy() for n in extras}
    base = {f: np.asarray(getattr(s0, f), np.float64).copy() for f in ("theta", "qv", "u", "w")}
    for i in range(STEPS):
        carry = om._advance_chunk(carry, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
        jax.block_until_ready(carry)
    leaves = jax.tree.leaves(carry)
    rec["nonfinite_values"] = int(sum(int((~np.isfinite(np.asarray(x))).sum()) for x in leaves
                                      if np.issubdtype(np.asarray(x).dtype, np.floating)))
    if getattr(carry, "census", None) is not None:
        from gpuwrf.diagnostics.census import EVENTS, GUARDS, f2_record, f2_negative_summary
        host_census = jax.device_get(carry.census)
        g = np.asarray(host_census.guards)
        rec["guards_fired"] = {GUARDS[r]: dict(zip(EVENTS, map(int, g[r]))) for r in range(g.shape[0]) if g[r].any()}
        rec["guard_candidates"] = int(g[:, :2].sum())
        rec["repair_candidates"] = int(g[:, 2].sum())
        rec["f2"] = f2_record(host_census, carry.state.qv.shape)
        rec["f2_negative_summary"] = f2_negative_summary(host_census, carry.state.qv.shape)
    s1 = carry.state
    rec["extras_stats"] = {}
    for n in extras:
        a = np.asarray(getattr(s1, n), np.float64)
        rec["extras_stats"][n] = {"min": float(a.min()), "max": float(a.max()), "init_max": float(init[n].max()),
                                  "changed": bool(np.any(a != init[n]))}
    rec["delta"] = {f: float(np.abs(np.asarray(getattr(s1, f), np.float64) - base[f]).max())
                    for f in ("theta", "qv", "u", "w")}
    rec["moisture_minmax"] = {f: {"min": float(np.asarray(getattr(s1, f)).min()),
                                   "max": float(np.asarray(getattr(s1, f)).max())} for f in ("qv", "qc", "qr", "qi", "qs", "qg")}
    rec["wall_s"] = round(time.time() - t0, 1)
    report["arms"][f"mp_physics={mp}"] = rec
    Path(OUT).write_text(json.dumps(report, indent=1, default=str))
    brief = {k: v for k, v in rec.items() if not k.startswith("f2")}
    brief["f2_events"] = rec.get("f2", {}).get("events", [])
    brief["negative_minima"] = [r for r in rec.get("f2_negative_summary", {}).get("records", [])
                               if r["negative_count"]]
    print(mp, brief, flush=True)
