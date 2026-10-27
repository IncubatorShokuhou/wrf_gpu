"""F2N01 replay: captured port operands (capture.py) through the verbatim pristine chain.

For every captured step and species, two pristine evaluations on the port's OWN operands:
  chain    from scratch: physics sc (port operand) + pristine hdiff + pristine sixth (dt/3)
           [+ port QV relax/spec, WRF overwrite semantics] -> rk_update_scalar_pd -> advect_scalar_pd
           -> rk_update_scalar -> flow_dep_bdy
  isolated every operator receives the port's own input (sc_tend, preload, advect tendency, update)
both with -ffp-contract=off and =fast (rounding envelope).  Port negatives are classified with the
pre-registered rule in replay_lib.  CPU only, light (< 1 GB).
usage: replay.py <run_dir> [--steps 1,2,...] --out out.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import pristine_chain as PC  # noqa: E402
import replay_lib as RL  # noqa: E402

BUILD = Path("<USER_HOME>/wrf_gpu2_lanes/o5-f2neg/build")


def step_eval(path, consts, work, libs, flow_names):
    z = np.load(path)
    keys = set(z.files)
    species = [k[5:] for k in z.files if k.startswith("qold_")]
    f32 = lambda k: np.asarray(z[k], np.float32)  # noqa: E731
    mu1, mu2 = f32("mu1"), f32("mu2")
    mut3, mu_old_pd = f32("pdk_mut"), f32("pdk_mu_old")
    ru, rv, rom = f32("pdk_ru"), f32("pdk_rv"), f32("pdk_rom")
    nz, ny, nx = z[f"qold_{species[0]}"].shape
    dt = float(work["dt_s"])
    rk = int(work.get("rk_order", 3))
    diff6 = {}
    j = 0
    while f"diff6call{j}_field" in keys:
        diff6[j] = (f32(f"diff6call{j}_field"), f32(f"diff6call{j}_mu"), f32(f"diff6call{j}_out"),
                    float(z[f"diff6call{j}_dt"]))
        j += 1
    checks = {"pdk_mu_old_equals_mu1": bool(np.array_equal(mu_old_pd, mu1)),
              "xkhh_present": "xkhh" in keys}
    # WRF caller wiring (solve_em.F): rk_scalar_tend at RK1 gets grid%muts AFTER the RK1 acoustic loop
    # (:2303); rk_update_scalar_pd uses grid%mu_1 = step-start mass (:1946); advect_scalar_pd at RK3 gets
    # the RK3 post-acoustic muts and mu_1; rk_update_scalar's mu_new = mu_2 (RK3); the RK3 'scalar' is the
    # RK2 result after its flow_dep_bdy.
    if "upd1_mu_new" in keys:
        checks["hdiff_muts_is_rk1_post_acoustic"] = bool(np.array_equal(f32("muts1"), f32("upd1_mu_new"))) if "muts1" in keys else None
        checks["preadd_mu1_is_step_start"] = bool(np.array_equal(mu1, f32("upd1_mu_old")))
        checks["pd_mut_is_rk3_post_acoustic"] = bool(np.array_equal(mut3, mu2))
        diff6_mu = [bool(np.array_equal(v[1], f32("upd1_mu_new"))) for v in diff6.values() if abs(v[3] - dt / rk) < 1e-9]
        checks["sixth_mu_is_rk1_post_acoustic"] = bool(diff6_mu) and all(diff6_mu)
        qc_ok = []
        for n in species:
            ref = f32(f"flow2_{n}") if f"flow2_{n}" in keys else (f32(f"upd2_{n}") if f"upd2_{n}" in keys else None)
            if ref is not None:
                qc_ok.append(bool(np.array_equal(f32(f"qcur_{n}"), ref)))
            if f"upd1qold_{n}" in keys:
                qc_ok.append(bool(np.array_equal(f32(f"qold_{n}"), f32(f"upd1qold_{n}"))))
        checks["rk3_field_is_rk2_result_and_qold_is_step_start"] = bool(qc_ok) and all(qc_ok)
    muts1 = f32("muts1") if "muts1" in keys else mu1
    xkhh = f32("xkhh") if "xkhh" in keys else np.zeros((nz, ny, nx), np.float32)
    c = {k: np.asarray(consts[k], np.float32) for k in consts.files}
    rows = {}
    for n in species:
        qold, qcur, pre = f32(f"qold_{n}"), f32(f"qcur_{n}"), f32(f"pre_{n}")
        phys = f32(f"src0_{n}") if f"src0_{n}" in keys else None
        other = f32(f"src1_{n}") if f"src1_{n}" in keys else None
        hd = f32(f"hdiff_{n}") if f"hdiff_{n}" in keys else np.zeros_like(qold)
        d6s = [v for v in diff6.values() if np.array_equal(v[0], qold) and abs(v[3] - dt / rk) < 1e-9]
        d6 = d6s[0][2] if d6s else np.zeros_like(qold)
        bdy = f32(f"bdy_{n}") if f"bdy_{n}" in keys else np.zeros_like(qold)
        sc = phys if phys is not None else None
        if other is not None:
            sc = other if sc is None else (sc + other).astype(np.float32)  # port order: physics + other
        sc = np.zeros_like(qold) if sc is None else sc
        adv, upd = f32(f"adv_{n}"), f32(f"upd_{n}")
        fin = f32(f"fin_{n}") if f"fin_{n}" in keys else upd
        flow = n in flow_names
        args = dict(nx=nx, ny=ny, nz=nz, spec_zone=int(work["spec_zone"]), flow=flow, ownqv=int(f"bdy_{n}" in keys),
                    opt=int(work["diff_6th_opt"]), qold=qold, qcur=qcur, xkhh=xkhh,
                    phys=phys if phys is not None else np.zeros_like(qold), bdy=bdy, ru=ru, rv=rv, ww=rom,
                    muts1=muts1, mu1=mu1, mut3=mut3, mu2=mu2, mx=c["msftx"], my=c["msfty"], ux=c["msfux"],
                    uy=c["msfuy"], vx=c["msfvx"], vy=c["msfvy"], fnm=c["fnm"], fnp=c["fnp"], rdnw=c["rdnw"],
                    c1=c["c1h"], c2=c["c2h"], rdx=float(z["pdk_rdx"]), rdy=float(z["pdk_rdy"]), dt=dt,
                    dt1=dt / rk, factor=float(work["diff_6th_factor"]))
        full = {k: libs[k](**args) for k in libs}
        # the CPU capture ran under XLA:CPU denormal flushing (E117, F2N04): same-semantics oracle
        full_ftz = libs["off"](**args, ftz=True)
        iso = {k: libs[k](**args, sc_in=sc, pre_in=f32(f"qoldpd_{n}"), adv_in=adv, upd_in=upd) for k in libs}
        port = {"hdiff": hd, "diff6": d6, "sc": sc, "pre": pre, "adv": adv, "upd": upd, "fin": fin}
        scale = RL.footprint_max(qold, qcur, pre)
        mass1 = c["c1h"][:, None, None] * mu1[None] + c["c2h"][:, None, None]
        parts = {"diff6": d6, "hdiff": hd}
        if phys is not None:
            parts["physics"] = phys
        rows[n] = {
            "operand_checks": {"pd_kernel_saw_preload": bool(np.array_equal(f32(f"qoldpd_{n}"), pre)),
                               "diff6_found": bool(d6s), "has_physics_sc": phys is not None,
                               "has_other_sc": other is not None, "flow_dep": flow},
            "isolated": {k: RL.compare_field(port[k], iso["off"][k], iso["fast"][k]) for k in port},
            "chain": {k: RL.compare_field(port[k], full["off"][k], full["fast"][k]) for k in port},
            "classify_fin_chain": RL.classify(fin, full["off"]["fin"], full["fast"]["fin"], scale),
            "classify_fin_isolated": RL.classify(fin, iso["off"]["fin"], iso["fast"]["fin"], scale),
            "classify_fin_chain_ftz": RL.classify(fin, full_ftz["fin"], full["fast"]["fin"], scale),
            "chain_ftz": {k: RL.compare_field(port[k], full_ftz[k]) for k in port},
            "classify_pre_chain": RL.classify(pre, full["off"]["pre"], full["fast"]["pre"], scale),
            "birth": RL.birth(qold, pre, fin, mass1, dt, parts),
            "allcell_fin_ieee": RL.allcell_diff(fin, full["off"]["fin"], qold, qcur, pre),
            "allcell_fin_ftz": RL.allcell_diff(fin, full_ftz["fin"], qold, qcur, pre),
        }
    return {"file": str(path), "checks": checks, "species": rows}


def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    want = None
    if "--steps" in sys.argv:
        want = {int(s) for s in sys.argv[sys.argv.index("--steps") + 1].split(",")}
    consts = np.load(run / "constants.npz")
    report = json.loads((run / "report.json").read_text())
    work = report["work"]
    summary = [json.loads(line) for line in (run / "summary.jsonl").read_text().splitlines() if line.strip()]
    flow_names = {}
    for r in summary:
        if "captured" in r:
            flow_names[r["step"]] = set(r.get("flow_names", []))
    libs = {k: PC.Chain(BUILD / f"chain_{k}.so") for k in ("off", "fast")}
    res = {"scope": __doc__, "run": str(run), "source": report.get("source"), "inputs": report.get("inputs"),
           "work": work, "steps": []}
    for step in sorted(flow_names):
        if want and step not in want:
            continue
        path = run / f"step{step:05d}.npz"
        r = step_eval(path, consts, work, libs, flow_names[step])
        r["step"] = step
        res["steps"].append(r)
        print(step, {n: (v["classify_fin_chain"]["verdict"], v["classify_fin_chain"]["port_negative"],
                         v["classify_fin_chain"]["port_min"], v["classify_fin_chain"]["wrf_min"])
                     for n, v in r["species"].items()}, flush=True)
        out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
