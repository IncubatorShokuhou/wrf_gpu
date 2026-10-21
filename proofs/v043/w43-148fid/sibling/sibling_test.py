"""Fused-sibling independence test for the one-way fused cascade (#148 critic, ALISIOS priority 1).

Product builder ``domain_tree._build_fused_cascade_program`` (the same ``fused_jit`` that runs the 9-nest fused
d02 + 7 leaves) with parent = d01 and N sibling leaves, inputs from OUR cases only (no ALISIOS data):
  --case PROD --copies N : N copies of the real PROD d02 (real carry, weights, ratio, cadence) -> real PROD shapes
  --case CROP --max-dom 4: the three different real-structured crops of PROD d02 (make_crop_case.py), each its own
                           namelist/weights/shape, as distinct siblings
Arms (fixed parent carry P; base child carries K_i; m = N//2):
  A = (K_0..K_{N-1})   B = K with child m perturbed   C = K with child 0 perturbed   I = (K_0,) isolated program
  mutant = coupled builder (child k forced state += 1e-3*mean(theta of child k-1)) -> must FAIL the gate
Phases: ORD (parent step 2, child steps 4..) and RAD (parent step pcad+1, child steps ccad..ccad+ratio-1, i.e. the
radiation step (step-1) % cadence == 0 is inside). Each call = one parent step with all child substeps.
Gate: every leaf of every UNPERTURBED child (same position) and of the parent is byte-identical, B vs A and C vs A.
--lower-only: trace + lower the arm-A program (no XLA compile, no execution); reports HLO size and peak host RSS.
--reuse-check: CPU check that every arm/phase hits ONE jit cache entry (perturbation changes values only: shape, dtype,
               weak_type, sharding, committedness kept); cheap proxy jit over the exact fused_jit argument tuples, no
               fused compile. The v1 perturbation (GPU phase-2 run 1, 2026-10-21) is checked alongside as the control.
Every run counts JAX jaxpr traces and backend compiles per stamp (log.json compile_counts).

usage: python sibling_test.py --out DIR [--case DIR --max-dom K | --copies N] [--mode legacy|release]
                              [--skip-mutant] [--skip-isolated] [--require-gpu] [--lower-only] [--reuse-check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


def _env(mode: str) -> None:
    os.environ["GPUWRF_NESTED_AOT"] = "0"          # call fused_jit directly: no blob load/serialize
    os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
    if mode == "legacy":
        os.environ["GPUWRF_FAST_DEFAULTS"] = "0"   # CPU-cost path; the GPU phase uses release defaults


def _closure(fn, name):
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


def _testbed_tree(np):
    """Dynamics-only fallback (run_physics False, no radiation): flat neutral 3 km parent 48x42x8 with a warm
    bubble and 5 m/s flow, three DIFFERENT 1 km siblings (24x21 @(5,5), 18x15 @(20,8), 21x27 @(30,20))."""
    from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
    from gpuwrf.runtime.domain_tree import DomainBundle, DomainTree
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    from gpuwrf.validation import moving_nest_testbed as tb

    ratio, pdx, nz, dt_s, ztop = 3, 3000.0, 8, 6.0, 8000.0
    bubble = lambda x, y: 2.0 * np.exp(-(((x - 60e3) / 15e3) ** 2 + ((y - 50e3) / 15e3) ** 2))  # noqa: E731
    pgrid = tb.build_flat_grid(nx=48, ny=42, nz=nz, dx_m=pdx, z_top_m=ztop)
    bundles = {"d01": DomainBundle("d01", tb.build_neutral_state(pgrid, z_top_m=ztop, u0_m_s=5.0, v0_m_s=2.0,
                                                                  theta_anomaly=bubble),
                                   tb.build_domain_namelist(pgrid, dt_s=dt_s, is_child=False), grid=pgrid)}
    nests = []
    for k, (nx, ny, i0, j0) in enumerate(((24, 21, 5, 5), (18, 15, 20, 8), (21, 27, 30, 20)), start=2):
        g = tb.build_flat_grid(nx=nx, ny=ny, nz=nz, dx_m=pdx / ratio, z_top_m=ztop)
        st = tb.build_neutral_state(g, z_top_m=ztop, u0_m_s=5.0, v0_m_s=2.0, theta_anomaly=bubble,
                                    x_coords_m=tb.cell_center_coords_m(n=nx, dx_m=pdx / ratio, parent_dx_m=pdx,
                                                                       parent_start_1based=i0, ratio=ratio),
                                    y_coords_m=tb.cell_center_coords_m(n=ny, dx_m=pdx / ratio, parent_dx_m=pdx,
                                                                       parent_start_1based=j0, ratio=ratio))
        name = f"d0{k}"
        bundles[name] = DomainBundle(name, st, tb.build_domain_namelist(g, dt_s=dt_s / ratio, is_child=True,
                                                                         parent_dt_s=dt_s), grid=g)
        nests.append(DomainNest("d01", name, ratio, i0, j0, feedback=False))
    hierarchy = DomainHierarchy.from_edges(tuple(bundles), tuple(nests))
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    carries = {k: _initial_carry_for_run(b.state, b.namelist) for k, b in bundles.items()}
    return tree, carries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", type=Path, default=PROD)
    ap.add_argument("--max-dom", type=int, default=2)
    ap.add_argument("--copies", type=int, default=0, help="N copies of d02 as siblings (PROD shapes)")
    ap.add_argument("--mode", choices=("legacy", "release"), default="legacy")
    ap.add_argument("--phases", default="ORD,RAD")
    ap.add_argument("--skip-mutant", action="store_true")
    ap.add_argument("--skip-isolated", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--lower-only", action="store_true")
    ap.add_argument("--reuse-check", action="store_true", help="CPU: one jit cache entry for all arms (no compile)")
    ap.add_argument("--abort-on-recompile", action="store_true", help="stop (rc 3) if fused_jit compiles a 2nd time")
    ap.add_argument("--testbed", action="store_true", help="dynamics-only synthetic fallback (no physics/radiation)")
    a = ap.parse_args()
    _env(a.mode)
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import domain_tree as dt
    from gpuwrf.runtime.domain_tree import DomainTree

    log = {"args": {k: str(v) for k, v in vars(a).items()}, "platform": jax.devices()[0].platform,
           "device": str(jax.devices()[0]),
           "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GPUWRF_", "JAX_", "XLA_"))},
           "timings": {}}
    log["jax_devices"] = [str(d) for d in jax.devices()]
    smi = os.environ.get("SIBLING_SMI_FILE")
    if smi and Path(smi).is_file():
        log["nvidia_smi_start"] = Path(smi).read_text()
    if a.require_gpu:
        assert jax.devices()[0].platform == "gpu", f"not a GPU process: {jax.devices()[0]}"

    counts = {"trace": 0, "backend_compile": 0, "fused_trace": 0, "fused_compile": 0, "fused_compile_s": 0.0,
              "proxy_trace": 0}

    def on_event(event, duration, **kw):
        fun = str(kw.get("fun_name", ""))
        if event == "/jax/core/compile/jaxpr_trace_duration":
            counts["trace"] += 1
            counts["fused_trace"] += "fused_jit" in fun
            counts["proxy_trace"] += "reuse_proxy" in fun
        elif event == "/jax/core/compile/backend_compile_duration":
            counts["backend_compile"] += 1
            if "fused_jit" in fun:
                counts["fused_compile"] += 1
                counts["fused_compile_s"] = round(counts["fused_compile_s"] + duration, 1)

    jax.monitoring.register_event_duration_secs_listener(on_event)
    log["compile_counts"] = {}

    def stamp(k):
        log["timings"][k] = round(time.time() - t0, 1)
        log["compile_counts"][k] = dict(counts)
        log["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        (a.out / "log.json").write_text(json.dumps(log, indent=1, default=str))
        print(f"[{log['timings'][k]:8.1f}s] {k} rss {log['peak_rss_gib']} GiB", flush=True)

    if a.testbed:
        tree, carries = _testbed_tree(np)
    else:
        names = pipeline.domain_names_for(a.max_dom)
        config = pipeline.NestedPipelineConfig(a.case, a.out / "run_out", a.out / "run_proof", hours=3.0,
                                               max_dom=a.max_dom, scratch_dir=a.out / "scratch")
        hierarchy, bundles, meta, _start, _dt, carries = pipeline._load_domains(config, names)
        nest = meta.get("nesting", {"feedback": False, "spec_zone": 1, "smooth_option": 1})
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=nest["feedback"],
                                       feedback_spec_zone=nest["spec_zone"], feedback_smooth_option=nest["smooth_option"])
    edges = list(tree.children("d01"))
    if a.copies:
        edges = [edges[0]] * a.copies
    n = len(edges)
    assert n >= 2, "need >= 2 siblings"
    pnl = tree.domains["d01"].namelist
    P = carries["d01"]
    K = [carries[e.child] for e in edges]
    NL = tuple(tree.domains[e.child].namelist for e in edges)
    W = tuple(e.weights for e in edges)
    BW = tuple(int(tree.domains[e.child].state.u_bdy.shape[2]) for e in edges)
    R = tuple(int(e.parent_grid_ratio) for e in edges)
    pcad = int(pnl.radiation_cadence_steps)
    CAD = tuple(int(nl.radiation_cadence_steps) for nl in NL)
    log.update(children=[e.child for e in edges], child_shapes=[list(k.state.theta.shape) for k in K],
               parent_cadence=pcad, child_cadences=CAD, ratios=R, bdy_widths=BW)
    stamp("loaded")

    def perturb_v1(carry, rng):
        """GPU phase-2 run 1 perturbation (kept only as the --reuse-check control)."""
        leaves, tdef = jax.tree_util.tree_flatten(carry)
        out = []
        for x in leaves:
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) and getattr(x, "size", 0) > 0:
                xn = np.asarray(x)
                out.append(jnp.asarray(xn * (1 + 1e-2 * rng.standard_normal(xn.shape).astype(xn.dtype)), dtype=x.dtype))
            else:
                out.append(x)
        c = jax.tree_util.tree_unflatten(tdef, out)
        st = c.state
        add = {f: (getattr(st, f) + jnp.asarray(rng.standard_normal(getattr(st, f).shape), dtype=getattr(st, f).dtype)
                   ).astype(getattr(st, f).dtype) for f in ("theta", "u", "v")}
        return c.replace(state=st.replace(**add))

    def like(x, v):
        """NumPy values v as an array with x's dtype, sharding and committedness (x is never weak-typed here)."""
        v = np.asarray(v, dtype=x.dtype)
        return jax.device_put(v, x.sharding) if x.committed else jnp.asarray(v, dtype=x.dtype)

    def perturbable(x):
        return (isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating) and x.size > 0
                and not jax.typeof(x).weak_type)

    def perturb(carry, rng):
        """Same draws as v1, but values only: weak-typed leaves stay unperturbed and every new leaf keeps the
        dtype/sharding/committedness of the leaf it replaces, so all arms hit the arm-A executable."""
        leaves, tdef = jax.tree_util.tree_flatten(carry)
        out = []
        for x in leaves:
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) and getattr(x, "size", 0) > 0:
                xn = np.asarray(x)
                v = xn * (1 + 1e-2 * rng.standard_normal(xn.shape).astype(xn.dtype))
                out.append(like(x, v) if perturbable(x) else x)
            else:
                out.append(x)
        c = jax.tree_util.tree_unflatten(tdef, out)
        st = c.state
        add = {}
        for f in ("theta", "u", "v"):
            x = getattr(st, f)
            v = np.asarray(x) + rng.standard_normal(x.shape).astype(x.dtype)
            add[f] = like(x, v) if perturbable(x) else x
        return c.replace(state=st.replace(**add))

    def leaf_sig(tree_):
        leaves, tdef = jax.tree_util.tree_flatten(tree_)
        return tdef, [(tuple(x.shape), str(x.dtype), bool(jax.typeof(x).weak_type), bool(x.committed), str(x.sharding))
                      if isinstance(x, jax.Array) else (type(x).__name__,) for x in leaves]

    def sig_diff(ta, tb):
        (da, la), (db, lb) = leaf_sig(ta), leaf_sig(tb)
        names = ("shape", "dtype", "weak_type", "committed", "sharding")
        kinds = {}
        for x, y in zip(la, lb):
            for nm, u, w in zip(names, x, y):
                if u != w:
                    kinds[nm] = kinds.get(nm, 0) + 1
        return {"treedef_equal": da == db, "n_leaves": len(la), "leaves_differing": sum(x != y for x, y in zip(la, lb)),
                "by_attribute": kinds}

    m = n // 2
    log["perturbation"] = "all float carry leaves x(1+1e-2 N(0,1)); theta/u/v + N(0,1) K or m/s"

    def build(idx, tag, mutant=False):
        kw = dict(parent_name="d01", parent_namelist=pnl, parent_cadence=pcad,
                  child_names=tuple(f"{tag}{i}" for i in idx), child_namelists=tuple(NL[i] for i in idx),
                  child_weights=tuple(W[i] for i in idx), child_bdy_widths=tuple(BW[i] for i in idx),
                  child_ratios=tuple(R[i] for i in idx), child_cadences=tuple(CAD[i] for i in idx))
        if not mutant:
            return dt._build_fused_cascade_program(**kw), None
        orig = dt.build_child_boundary_package
        prev = {"theta": None}

        def coupled(child_state, parent_state, weights, **k2):
            out = orig(child_state, parent_state, weights, **k2)
            if prev["theta"] is not None:
                out = out.replace(theta=(out.theta + 1e-3 * jnp.mean(prev["theta"])).astype(out.theta.dtype))
            prev["theta"] = out.theta
            return out

        dt.build_child_boundary_package = coupled   # active while the mutant traces (first call), restored after
        return dt._build_fused_cascade_program(**kw), orig

    def digest(tree_):
        leaves = jax.tree_util.tree_leaves(jax.device_get(tree_))
        return [hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16] if hasattr(x, "shape")
                else repr(x) for x in leaves], leaves

    def run(prog, kids, ps, cs):
        newp, newc = prog(P, tuple(kids), ps, tuple([cs] * len(kids)))
        jax.block_until_ready((newp, newc))
        return newp, newc

    def maxdiff(la, lb):
        out = 0.0
        for x, y in zip(la, lb):
            xa = np.asarray(x)
            if np.issubdtype(xa.dtype, np.floating) and xa.size:
                out = max(out, float(np.nanmax(np.abs(xa.astype(np.float64) - np.asarray(y, np.float64)))))
        return out

    full = list(range(n))
    prog, _ = build(full, "s")
    stamp("built")

    if a.lower_only:
        fused_call = _closure(prog, "_fused_call")
        fused_jit, args_fn = _closure(fused_call, "fused_jit"), _closure(fused_call, "_args")
        prep = _closure(prog, "_prepare_windows")
        pc, kids, clocks = prep(P, tuple(K), 2, tuple([4] * n))
        lowered = fused_jit.lower(*args_fn(pc, kids, 2, tuple([4] * n), clocks))
        stamp("lowered")
        txt = lowered.as_text()
        log["lowered"] = {"stablehlo_chars": len(txt), "stablehlo_lines": txt.count("\n"),
                          "sha256": hashlib.sha256(txt.encode()).hexdigest()}
        log["verdict"] = "LOWERED"
        stamp("end")
        return 0

    S = {i: perturb(K[i], np.random.default_rng(148 + i)) for i in sorted({m, 0})}
    log["perturb_signature"] = {str(i): sig_diff(K[i], S[i]) for i in S}
    sig_ok = all(d["treedef_equal"] and d["leaves_differing"] == 0 for d in log["perturb_signature"].values())
    stamp("perturbed")
    if a.reuse_check:
        S1 = {i: perturb_v1(K[i], np.random.default_rng(148 + i)) for i in S}
        log["perturb_v1_signature"] = {str(i): sig_diff(K[i], S1[i]) for i in S1}
        fused_call = _closure(prog, "_fused_call")
        args_fn, prep = _closure(fused_call, "_args"), _closure(prog, "_prepare_windows")
        def reuse_proxy(*xs):
            return [jnp.sum(x) for x in jax.tree_util.tree_leaves(xs) if jnp.issubdtype(x.dtype, jnp.floating)][:4]

        proxy = jax.jit(reuse_proxy)
        rc = {}
        for label, SS in (("v2", S), ("v1", S1)):
            entries = []
            for ph, (ps, cs) in {"ORD": (2, 4), "RAD": (pcad + 1, CAD[0])}.items():
                for arm, pert in {"A": None, "B": m, "C": 0}.items():
                    kids = tuple(SS[i] if i == pert else K[i] for i in full)
                    pc, kk, clocks = prep(P, kids, ps, tuple([cs] * n))
                    before = counts["proxy_trace"]
                    jax.block_until_ready(proxy(*args_fn(pc, kk, ps, tuple([cs] * n), clocks)))
                    entries.append({"phase": ph, "arm": arm, "new_traces": counts["proxy_trace"] - before,
                                    "cache_size": proxy._cache_size()})
            rc[label] = {"calls": entries, "final_cache_size": proxy._cache_size()}
            proxy.clear_cache()
        log["reuse_check"] = rc
        log["verdict"] = "REUSE_OK" if sig_ok and rc["v2"]["final_cache_size"] == 1 else "REUSE_FAIL"
        stamp("end")
        print(json.dumps({k: log[k] for k in ("perturb_signature", "perturb_v1_signature", "reuse_check")}, indent=1))
        return 0 if log["verdict"] == "REUSE_OK" else 1
    assert sig_ok, f"perturbation changed a leaf signature: {log['perturb_signature']}"
    iso = None if a.skip_isolated else build([0], "i")[0]
    phases = {"ORD": (2, 4), "RAD": (pcad + 1, CAD[0])}
    arms_def = {"A": None, "B": m, "C": 0}
    res = {}
    for ph in a.phases.split(","):
        ps, cs = phases[ph]
        outs, kidsets = {}, {}
        for arm, pert in arms_def.items():
            kids = [S[i] if i == pert else K[i] for i in full]
            kidsets[arm] = pert
            outs[arm] = run(prog, kids, ps, cs)
            stamp(f"{ph}_{arm}")
            if a.abort_on_recompile and counts["fused_compile"] > 1:
                log["verdict"] = "RECOMPILE"
                stamp("end")
                return 3
        dig = {k: {"parent": digest(v[0])[0], "children": [digest(c)[0] for c in v[1]]} for k, v in outs.items()}
        checks, fails = 0, []
        for arm in ("B", "C"):
            for i in full:
                if i == kidsets[arm]:
                    continue
                for j, (x, y) in enumerate(zip(dig[arm]["children"][i], dig["A"]["children"][i])):
                    checks += 1
                    if x != y:
                        fails.append({"arm": arm, "child": i, "leaf": j})
            for j, (x, y) in enumerate(zip(dig[arm]["parent"], dig["A"]["parent"])):
                checks += 1
                if x != y:
                    fails.append({"arm": arm, "parent": True, "leaf": j})
        r = {"parent_start": ps, "child_start": cs, "child_steps": [cs, cs + R[0] - 1],
             "radiation_step_in_child_steps": any((k - 1) % CAD[0] == 0 for k in range(cs, cs + R[0])),
             "radiation_step_in_parent_step": (ps - 1) % pcad == 0,
             "n_leaves_per_child": len(dig["A"]["children"][0]), "gate_checks": checks, "gate_failures": fails[:50],
             "gate_pass": not fails,
             "perturbed_child_leaves_differing_B": sum(x != y for x, y in zip(dig["B"]["children"][m], dig["A"]["children"][m])),
             "perturbed_child_leaves_differing_C": sum(x != y for x, y in zip(dig["C"]["children"][0], dig["A"]["children"][0])),
             "child0_leaves_changed_by_step": sum(x != y for x, y in zip(digest(K[0])[0], dig["A"]["children"][0])),
             "child0_step_maxabs_change": maxdiff(digest(K[0])[1], digest(outs["A"][1][0])[1])}
        rd = getattr(K[0], "radiation_diagnostics", None)
        if rd is not None:
            r["child0_radiation_diag_leaves_changed"] = sum(
                x != y for x, y in zip(digest(rd)[0], digest(outs["A"][1][0].radiation_diagnostics)[0]))
        if iso is not None:
            oi = run(iso, [K[0]], ps, cs)
            stamp(f"{ph}_I")
            di = digest(oi[1][0])
            r["A_vs_isolated_identical_leaves"] = sum(x == y for x, y in zip(di[0], dig["A"]["children"][0]))
            r["A_vs_isolated_maxabs"] = maxdiff(di[1], digest(outs["A"][1][0])[1])
        if not a.skip_mutant:
            mprog, orig = build(full, "m", mutant=True)
            try:
                mo = {arm: run(mprog, [S[i] if i == pert else K[i] for i in full], ps, cs) for arm, pert in arms_def.items()}
            finally:
                dt.build_child_boundary_package = orig
            md = {k: [digest(c)[0] for c in v[1]] for k, v in mo.items()}
            r["mutant_gate_failures"] = sum(x != y for arm in ("B", "C") for i in full if i != kidsets[arm]
                                            for x, y in zip(md[arm][i], md["A"][i]))
            r["mutant_killed"] = r["mutant_gate_failures"] > 0
            stamp(f"{ph}_mutant")
        res[ph] = r
        log["results"] = res
        stamp(f"{ph}_done")
    ok = all(r["gate_pass"] for r in res.values()) and all(r.get("mutant_killed", True) for r in res.values())
    log["verdict"] = "PASS" if ok else "FAIL"
    stamp("end")
    print(json.dumps(res, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
