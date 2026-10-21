"""Assemble the w43-teacher qualification evidence (TEACHER_PREREG v3) into analysis/TEACHER_QUAL.{json,md} parts.

usage: python3 build_qual.py OUT_DIR
Reads only finished receipts (score dirs, extracts, case inputs, CPU run receipts); computes no new model numbers. Writes:
  OUT_DIR/TEACHER_QUAL.json        - machine-readable: identity, per case x domain verdicts (teacher_qual.py), twins, G-INT, extracts
  OUT_DIR/QUAL_TABLE.md            - teacher_qual.py table (case x domain x field class, wind tau>40 h row)
  OUT_DIR/GINT_ITEMS.md            - gint_tables.py (FAIL stays FAIL)
  OUT_DIR/IDENTITY.md, TWINS.md, EXTRACTS.md - tables included into TEACHER_QUAL.md by hand-written prose
"""
import hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher")
L = Path("<USER_HOME>/wrf_gpu2_lanes/w43-teacher")
SC = L / "score"
WN3 = ["20260227", "20260502", "20260614", "20260220", "20260608", "20260120"]
ERA5 = ["20240302", "20240218"]
TWIN = {"wn3_20260227": SC / "C_0227", "era5_20240302": SC / "C_era5_20240302"}
CPU_RUN = {"era5_20240302": S / "cpu/era5_20240302_ref_3x1", "era5_20240218": S / "cpu/era5_20240218_ref_2x2"}
TWIN_RUN = {"era5_20240302": S / "cpu/era5_20240302_twin_2x2"}
BUNDLE = {"tag": "v0.4.2 (private tag object 08cdd85b -> 5c34b4189d; public ae0a17df -> 59f5f69755)",
          "bundle_pin_commit": "b979188128e0b6038d79a81e02d1dd89f273e969",
          "src_tree": "b3e9b92a8ce2f1aa4a96f39b0b03424a3c1ebb42", "gpuwrf_tree": "0dc759849113b8c204081c9c4f22e698c07e024d",
          "bundle_sha256": "bfa2dc90c034e6e057c23694d64ae1dbd7c4579808975a540be42caac5d616d2",
          "source": "cross/FROM_WRF_GPU2_V042_TUPLE_20261019.md"}
INPUTS = ["namelist.input", "wrfbdy_d01", "wrfinput_d01", "wrfinput_d02", "wrfinput_d03", "wrflowinp_d01", "wrflowinp_d02",
          "wrflowinp_d03"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    return json.loads(Path(p).read_text())


def set_sha(files):
    return hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def case_identity(name):
    cd = S / "cases" / name
    files = {f: sha(cd / f) for f in INPUTS}
    rec = jl(SC / name / "gpu_receipt.json")
    out = {"case_dir": str(cd), "inputs_sha256": files, "input_set_sha256": set_sha(files),
           "namelist_sha256": files["namelist.input"],
           "gpu": {"source": rec["source"], "command": rec["command"], "aot_cache_warm": rec["aot_cache_warm"], "rc": rec["rc"],
                   "start_utc": rec["start_utc"], "end_utc": rec["end_utc"], "wall_s": rec["wall_s"],
                   "vram_peak_mib": rec["vram_peak_mib"], "receipt_sha256": sha(SC / name / "gpu_receipt.json")},
           "gpu_wrfout_manifest_sha256": sha(SC / name / "GPU_WRFOUT_MANIFEST.json") if (SC / name / "GPU_WRFOUT_MANIFEST.json").exists() else None}
    if name in CPU_RUN:  # ERA5: our CPU-WRF runs on the same staged inputs; prove binary identity, list namelist diffs
        for role, rd in [("cpu_ref", CPU_RUN[name])] + ([("cpu_twin", TWIN_RUN[name])] if name in TWIN_RUN else []):
            r = jl(rd / "cpu_run.json")
            fi = {f: sha(rd / "input" / f) for f in INPUTS}
            diff = [f for f in INPUTS if fi[f] != files[f]]
            nl = subprocess.run(["diff", str(cd / "namelist.input"), str(rd / "input/namelist.input")], capture_output=True, text=True).stdout
            out[role] = {"run_dir": str(rd), "rc": r["rc"], "success_complete": r["success_complete"], "wrfout_files": r["wrfout_files"],
                         "executable": r.get("executable"), "executable_sha256": r.get("executable_sha256"), "wall_s": r.get("wall_s"),
                         "input_files_differing_from_gpu_case": diff, "namelist_diff_vs_gpu_case": nl.strip().splitlines()}
    else:  # WN3: CPU ref = ALISIOS WT2 run dir, staged via the guarded facade (receipts WT2_<issue>_try1 / G04_ref0227)
        out["cpu_ref"] = {"source": "ALISIOS wg_<issue>_18z_a1 CPU-WRF 4.7.1 run (staged guarded, receipts analysis/receipts/WT2_*/G04_ref0227)",
                          "gpu_inputs_receipt": "analysis/receipts/G02_wn3_inputs/sha.json (inputs staged from the same run dir)"}
    return out


import importlib.util
_spec = importlib.util.spec_from_file_location("verdict", HERE / "verdict.py")
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)
BLOCKS = [("0-23", range(0, 24)), ("24-47", range(24, 48)), ("48-72", range(48, 73))]


def worst_table(name, sdir, cdir):
    """PREREG v3 section 6: per domain x field class x tau block the row with the largest X/L, with X, L, C (own twin) and X/C."""
    v = jl(sdir / "VERDICT.json")
    rows = []
    for dom, dv in v["domains"].items():
        X, _ = V.rows_of(sdir / "x", dom)
        C = V.rows_of(cdir / "x", dom)[0] if cdir else {}
        best = {}
        for (f, t, m), (x, lim) in X.items():
            if lim is None or lim == 0 or m in V.LOWER or not V.fin(x, lim):
                continue
            for b, rg in BLOCKS:
                if t in rg:
                    k = (V.cls(f), b)
                    if k not in best or x / lim > best[k][0]:
                        best[k] = (x / lim, f, t, m, x, lim, (C.get((f, t, m)) or (None,))[0])
        for (c, b), (r, f, t, m, x, lim, cv) in sorted(best.items()):
            st = dv["classes"].get(c, {}).get("status", "PASS")
            rows.append({"case": name, "domain": dom, "class": c, "class_status": st, "block": b, "field": f, "lead_h": t,
                         "metric": m, "X": x, "L": lim, "X_over_L": r, "C": cv,
                         "X_over_C": (x / cv) if cv not in (None, 0) and V.fin(cv) else None})
    return rows


def twin0227_validity():
    T = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1")
    r = jl(T / "cpu_run_native_times.json")
    cd = S / "cases/wn3_20260227"
    return {"run_dir": str(T / "cpu"), "lane": "w43-twin0227 (4x1 on cores 1,4,5,7)", "receipt": str(T / "cpu_run_native_times.json"),
            "receipt_sha256": sha(T / "cpu_run_native_times.json"), "rc": r["rc"], "status": r["status"], "wrfout": r["wrfout"],
            "complete_times": r["complete_times"], "completion_check": r["completion_check"],
            "supersedes_bookkeeping_only": r["supersedes_bookkeeping_only"], "original_receipt_sha256": r["original_receipt_sha256"],
            "input_identity": r["input_identity"], "executable_sha256": r["executable_sha256"],
            "output_manifest_sha256": r["output_manifest_sha256"], "field_count_per_file": r["field_count_per_file"],
            "namelist_diff": (T / "namelist.diff").read_text().strip().splitlines(),
            "gpu_case_namelist_sha256": sha(cd / "namelist.input"),
            "twin_original_namelist_equals_gpu_case": sha(T / "original_namelist.input") == sha(cd / "namelist.input"),
            "scorer_preconditions": jl(SC / "C_0227/PRECONDITIONS.json").get("pass")}


def twin_check(cdir):
    """All-field finiteness/schema of the twin-vs-ref scorer output (critic requirement)."""
    res = {}
    for d in ("d01", "d02", "d03"):
        x = jl(cdir / "x" / f"{d}.json")
        fs = x["field_summaries"]
        bad, nlead = [], set()
        for f, v in fs.items():
            for r in v.get("by_lead", []):
                nlead.add(r["lead_h"])
                if r.get("finite_pair_fraction") not in (1, 1.0):
                    bad.append((f, r["lead_h"], r.get("finite_pair_fraction")))
        res[d] = {"fields": len(fs), "leads": len(nlead), "non_finite_pairs": bad[:10], "n_non_finite": len(bad),
                  "missing_vs_cpu": x.get("missing_vs_cpu", x.get("integrity", {}).get("missing_variables_vs_cpu"))}
    return res


def extract_row(xdir, role):
    rb, man = jl(xdir / "READBACK.json"), jl(xdir / "MANIFEST.json")
    return {"role": role, "dir": str(xdir), "status": rb["status"], "manifest_sha256": sha(xdir / "MANIFEST.json"),
            "artifacts": {k: v["sha256"] for k, v in man["artifacts"].items()},
            "domains": {d: {"frames": v["frames"], "allowed_nan": v["allowed_nan_total"], "foreign_nonfinite": v["foreign_nonfinite_total"]}
                        for d, v in rb["domains"].items()},
            "profile_distinct_tau": rb.get("profile", {}).get("distinct_tau")}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    names = [f"wn3_{i}" for i in WN3] + [f"era5_{i}" for i in ERA5]
    specs = []
    for n in names:
        forcing = "WN3" if n.startswith("wn3") else "ERA5"
        specs.append(f"{n}={forcing}:{SC / n}" + (f":{TWIN[n] / 'x'}" if n in TWIN else ""))
    subprocess.run([sys.executable, str(HERE / "teacher_qual.py"), str(out / "QUAL_CASES.json"), str(out / "QUAL_TABLE.md"), *specs],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(HERE / "gint_tables.py"), str(out / "GINT_ITEMS.md"), str(SC), *names], check=True,
                   stdout=subprocess.DEVNULL)
    qual = jl(out / "QUAL_CASES.json")
    # cr-w43-teacher stage2 E200 finding: verdict_v2 (coverage + both-side finiteness over all fields) is the binding verdict;
    # teacher_qual.py aggregates the pinned v1 VERDICT.json, so require v1 == v2 per domain and attach the v2 coverage receipt.
    cov = {}
    for c in qual["cases"]:
        n = c["case"]
        v1, v2 = jl(SC / n / "VERDICT.json"), jl(SC / n / "VERDICT_v2.json")
        for d in v2["domains"]:
            a1 = (v1["domains"][d]["teacher"], v1["domains"][d]["bounded_worst"])
            a2 = (v2["domains"][d]["teacher"], v2["domains"][d]["bounded_worst"])
            if a1 != a2 or v1["breaches"] != v2["breaches"] or v1["case_teacher"] != v2["case_teacher"]:
                raise SystemExit(f"verdict_v2 differs from v1 for {n} {d}: {a1} vs {a2} - aggregate from VERDICT_v2 instead")
            c["domains"][d]["coverage_v2"] = v2["domains"][d]["coverage_v2"]
        c["twin_v2"] = v2["twin_v2"]
        cov[n] = {"verdict_v2_sha256": sha(SC / n / "VERDICT_v2.json"), "case_teacher_v1": v1["case_teacher"],
                  "case_teacher_v2": v2["case_teacher"],
                  "violations": sum(len(x["coverage_v2"]["fail"]) + len(x["coverage_v2"]["incomplete"]) for x in v2["domains"].values())}
    for c in qual["cases"]:  # cr-w43-teacher stage2: per-domain wind value is the bounded diagnostic, final row is HELD
        for dv in c["domains"].values():
            dv["wind_tau_gt40_bounded_diagnostic"] = dv.pop("wind_tau_gt40")
    t = (out / "QUAL_TABLE.md").read_text().replace("| wind τ>40 h |\n", "| wind τ>40 h bounded (diagnostic; final row HELD) |\n", 1)
    (out / "QUAL_TABLE.md").write_text("\n".join(x for x in t.splitlines() if not x.startswith("Wind τ>40 h (leads")).rstrip("\n") + "\n")
    ident = {n: case_identity(n) for n in names}
    twins = {n: {"dir": str(c), "d6": jl(c / "PRECONDITIONS.json") if (c / "PRECONDITIONS.json").exists() else None,
                 "finiteness_schema": twin_check(c)} for n, c in TWIN.items()}
    extracts = {}
    for n in names:
        i = n.split("_", 1)[1]
        cpu = (L / "te_repro" / f"{i}_cpu") if n.startswith("wn3") else (L / "te_cpu" / f"{i}_cpu")
        extracts[n] = {"gpu": extract_row(L / "te_gpu" / f"{i}_gpu", "GPU wrf_gpu v0.4.2"),
                       "cpu": extract_row(cpu, "CPU-WRF 4.7.1 reference")}
    gint = {n: jl(SC / n / "integ/INTEGRITY.json") for n in names}
    worst = [r for n in names for r in worst_table(n, SC / n, TWIN.get(n))]
    twins["wn3_20260227"]["validity"] = twin0227_validity()
    res = {"schema": "w43-teacher.teacher_qual.v1", "rule": "analysis/TEACHER_PREREG.md v3 (sha256 %s)" % sha(HERE / "TEACHER_PREREG.md"),
           "tools": {p: sha(HERE / p) for p in ["score_case_v2.sh", "post_batch_v2.sh", "verdict.py", "verdict_v2.py", "test_verdict_v2.py", "teacher_qual.py",
                                                "teacher_extract.py", "gint_tables.py", "build_qual.py"]},
           "identity": BUNDLE, "cases": qual["cases"],
           "wind_tau_gt40_h": {
               "verdict": "HELD", "final": True,
               "rule": "ALISIOS acceptance item 5: stays HELD until explicitly qualified (decision ALISIOS/manager, not this lane)",
               "reasons_not_promoted": [
                   "4 of 8 cases teacher-FAIL on G-INT (non-wind trace-zero items); their case-domains are not TEACHER-OK",
                   "only 2 of 8 cases have an own CPU twin (one twin per forcing)",
                   "deterministic one-twin rule, no statistical guarantee (PREREG v3 section 2)",
                   "ERA5 E-C 10 m wind X/C up to 2.48 (d03 U10@68) while under D6 (coastwind context)"],
               "diagnostic_bounded_wind_criterion": {
                   "note": "teacher_qual.py criterion (wind10+wind3d D6 bounds, leads 41-72, coverage complete); NOT a final "
                           "qualification: explicit G-INT False does not hold this diagnostic",
                   "computed": qual["wind_tau_gt40_h"]["verdict"],
                   "per_case_domain": qual["wind_tau_gt40_h"]["per_case_domain"]}},
           "case_identity": ident,
           "twins": twins, "gint": gint, "extracts": extracts, "worst_lead": worst,
           "verdict_v2_coverage": cov}
    (out / "TEACHER_QUAL.json").write_text(json.dumps(res, indent=1) + "\n")
    # small markdown tables
    I = ["| case | forcing | input set sha256 (8 files) | namelist sha256 | GPU start–end UTC | GPU wall s | VRAM MiB | CPU ref |",
         "|---|---|---|---|---|---|---|---|"]
    for n in names:
        c = ident[n]; g = c["gpu"]
        cpu = c["cpu_ref"].get("run_dir", "ALISIOS WT2 run (guarded stage)")
        I.append(f"| {n} | {'WN3' if n.startswith('wn3') else 'ERA5'} | `{c['input_set_sha256'][:16]}` | `{c['namelist_sha256'][:16]}` | "
                 f"{g['start_utc'][11:16]}–{g['end_utc'][11:16]} | {g['wall_s']:.0f} | {g['vram_peak_mib']} | {Path(cpu).name} |")
    (out / "IDENTITY.md").write_text("\n".join(I) + "\n")
    T = ["| case | twin | domain | fields | leads | non-finite pairs | missing vs CPU |", "|---|---|---|---|---|---|---|"]
    for n, t in twins.items():
        for d, v in t["finiteness_schema"].items():
            T.append(f"| {n} | {Path(t['dir']).name} | {d} | {v['fields']} | {v['leads']} | {v['n_non_finite']} | {v['missing_vs_cpu'] or 'none'} |")
    (out / "TWINS.md").write_text("\n".join(T) + "\n")
    E = ["| case | side | readback | d01 frames/allowed NaN/foreign | d02 | d03 | profile τ | MANIFEST sha256 |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        for side in ("cpu", "gpu"):
            e = extracts[n][side]; dd = e["domains"]
            cell = lambda d: f"{dd[d]['frames']}/{dd[d]['allowed_nan']}/{dd[d]['foreign_nonfinite']}"
            E.append(f"| {n} | {side.upper()} | {e['status']} | {cell('d01')} | {cell('d02')} | {cell('d03')} | {e['profile_distinct_tau']} | `{e['manifest_sha256'][:16]}` |")
    (out / "EXTRACTS.md").write_text("\n".join(E) + "\n")
    g = lambda v: "-" if v is None else f"{v:.4g}"
    W = ["| case | dom | class | class verdict | τ block | worst field@lead | metric | X | L | X/L | C (own twin) | X/C |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in worst:
        W.append(f"| {r['case']} | {r['domain']} | {r['class']} | {r['class_status']} | {r['block']} | {r['field']}@{r['lead_h']} | "
                 f"{r['metric']} | {g(r['X'])} | {g(r['L'])} | {r['X_over_L']:.3f} | {g(r['C'])} | {g(r['X_over_C'])} |")
    (out / "WORST_LEAD.md").write_text("\n".join(W) + "\n")
    K = ["| case | v1 teacher | v2 teacher | coverage/finiteness violations (all fields × 73 leads × 3 domains, both sides) | VERDICT_v2 sha256 |",
         "|---|---|---|---|---|"]
    for n, v in cov.items():
        K.append(f"| {n} | {v['case_teacher_v1']} | {v['case_teacher_v2']} | {v['violations']} | `{v['verdict_v2_sha256'][:16]}` |")
    (out / "VERDICT_V2.md").write_text("\n".join(K) + "\n")
    print(json.dumps({"wind_final": "HELD", "wind_diagnostic": qual["wind_tau_gt40_h"]["verdict"], "cases": {c["case"]: c["case_teacher"] for c in qual["cases"]}}))


if __name__ == "__main__":
    main()
