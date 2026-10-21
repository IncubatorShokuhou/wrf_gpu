"""Teacher verdict v2 = verdict.py (pinned, unchanged) + a complete-coverage / both-side-finiteness gate over ALL fields.

cr-w43-teacher stage-2 finding 2026-10-21T18:35Z (E200 class): verdict.py reads only tolerance checks, so NaN/Inf in an
unbounded field or a deleted bounded field still produced TEACHER-OK. PREREG v3 sections 2/3 already require: every field
finite on both sides, complete exact lead coverage, no missing required field, compatible metadata - never floor-limited.
This file enforces that from the frozen scorer's own receipts BEFORE the bounded aggregation:
  - inventory: cpu_only empty, incompatible_common empty, non_numeric_common empty, CPU variable count == REQUIRED_FIELDS (375);
  - every manifest mandatory-presence field is in the CPU file (a field deleted on one side shows up in cpu_only / no summary);
  - every numeric common field: by_lead leads exactly 0..72, missing_leads / incompatible_leads empty, CPU == GPU metadata, and
    at every lead n > 0, finite_cpu == n, finite_gpu == n (the scorer counts np.isfinite: NaN, +Inf, -Inf all reduce it);
  - Times (non-numeric): PRECONDITIONS.json cpu_exact, gpu_exact, times_ok (exact 219 names + Times, both sides);
  - the same gate on the own twin's scorer output; a twin that fails it supplies no C (case treated as no-twin, PREREG section 5).
A violation -> that domain FAIL (reason listed). A fact the receipts cannot show (keys absent, receipt unreadable) -> INCOMPLETE.

usage: python3 verdict_v2.py CASE X_DIR [C_DIR|-] INTEGRITY_JSON|- PRECONDITIONS_JSON OUT_JSON
"""
import importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("verdict_v1", HERE / "verdict.py")
V1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V1)

REQUIRED_FIELDS = 375
LEADS = list(range(73))
MANIFEST = Path("<USER_HOME>/src/wrf_gpu2/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json")
RANK = {"PASS": 0, "FLOOR-LIMITED": 1, "FAIL": 2, "INCOMPLETE": 3}


def manifest_fields():
    m = json.loads(MANIFEST.read_text())
    return set(m["fields"]), set(m["score_policy"]["mandatory_presence_fields_for_final_runs"])


def coverage(score_dir, dom):
    """-> (fail_reasons, incomplete_reasons) for one domain's frozen-scorer receipt."""
    fail, inc = [], []
    try:
        d = json.loads((Path(score_dir) / f"{dom}.json").read_text())
        inv, fs = d["inventory"], d["field_summaries"]
    except (OSError, ValueError, KeyError) as e:
        return [], [f"{dom}: scorer receipt unreadable ({type(e).__name__})"]
    for k in ("cpu_only", "incompatible_common", "non_numeric_common"):
        if inv.get(k) is None:
            inc.append(f"{dom}: inventory.{k} absent")
        elif inv[k]:
            fail.append(f"{dom}: inventory.{k} = {sorted(inv[k])[:8]}")
    if inv.get("cpu_variable_count") != REQUIRED_FIELDS:
        fail.append(f"{dom}: CPU variable count {inv.get('cpu_variable_count')} != {REQUIRED_FIELDS}")
    cpu_vars = set(inv.get("common") or []) | set(inv.get("cpu_only") or [])
    _, mandatory = manifest_fields()
    for f in sorted(mandatory - cpu_vars):
        fail.append(f"{dom}: mandatory field {f} absent from the CPU file")
    for f in sorted(set(inv.get("common") or [])):
        v = fs.get(f)
        if v is None:
            fail.append(f"{dom}: common field {f} has no summary"); continue
        if v.get("classification") == "time_metadata" and not v.get("by_lead"):
            continue  # Times: covered by PRECONDITIONS (exact names + Times)
        if v.get("missing_leads") or v.get("incompatible_leads"):
            fail.append(f"{dom}: {f} missing/incompatible leads {v.get('missing_leads')}/{v.get('incompatible_leads')}")
        md = v.get("metadata") or {}
        if md.get("cpu") != md.get("gpu"):
            fail.append(f"{dom}: {f} metadata CPU != GPU")
        rows = v.get("by_lead") or []
        if [r.get("lead_h") for r in rows] != LEADS:
            fail.append(f"{dom}: {f} lead coverage {len(rows)} rows != 0..72"); continue
        for r in rows:
            if not all(isinstance(r.get(k), int) for k in ("n", "finite_cpu", "finite_gpu")):
                inc.append(f"{dom}: {f}@{r['lead_h']} finite counts absent from receipt"); break
            if not (r["n"] > 0 and r["finite_cpu"] == r["n"] and r["finite_gpu"] == r["n"]):
                fail.append(f"{dom}: {f}@{r['lead_h']} non-finite (n {r['n']}, finite cpu {r['finite_cpu']} gpu {r['finite_gpu']})")
    return fail, inc


def preconditions(pre):
    if pre is None:
        return [], ["PRECONDITIONS.json absent"]
    bad = [k for k in ("cpu_exact", "gpu_exact", "times_ok") if pre.get(k) is not True]
    return ([f"PRECONDITIONS {k} not true" for k in bad] if "pass" in pre else []), ([] if "pass" in pre else ["PRECONDITIONS.pass absent"])


def verdict(case, xdir, cdir, integ, pre):
    twin_note = None
    if cdir is not None:
        tf, ti = [], []
        for dom in ("d01", "d02", "d03"):
            a, b = coverage(cdir, dom); tf += a; ti += b
        if tf or ti:
            twin_note = {"twin_dir": str(cdir), "invalid": tf + ti, "consequence": "no C (case treated as no-twin)"}
            cdir = None
    r = V1.verdict(case, xdir, cdir, integ)
    r["rule"] = "TEACHER_PREREG v3 via verdict_v2: complete-coverage + both-side finiteness gate over all fields, then v1 bounded rule"
    pf, pi = preconditions(pre)
    for dom, dv in r["domains"].items():
        f, i = coverage(xdir, dom)
        f, i = f + pf, i + pi
        dv["coverage_v2"] = {"fail": f, "incomplete": i, "fields_checked": "all common numeric fields x 73 leads + Times via PRECONDITIONS"}
        if i:
            dv["teacher"] = "INCOMPLETE"
        elif f:
            dv["teacher"] = "FAIL"; dv["bounded_worst"] = "FAIL"
    r["twin_v2"] = twin_note or ({"twin_dir": str(cdir), "valid": True} if cdir else None)
    ds = [d["teacher"] for d in r["domains"].values()]
    r["case_teacher"] = "INCOMPLETE" if "INCOMPLETE" in ds else "TEACHER-OK" if all(t == "TEACHER-OK" for t in ds) else "FAIL"
    return r


if __name__ == "__main__":
    case, xdir, cdir, ij, pj, out = sys.argv[1:7]
    integ = None if ij == "-" else json.loads(Path(ij).read_text())
    pre = json.loads(Path(pj).read_text()) if Path(pj).exists() else None
    res = verdict(case, xdir, None if cdir == "-" else cdir, integ, pre)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"case": case, "case_teacher": res["case_teacher"],
                      **{d: v["teacher"] + "/" + v["bounded_worst"] for d, v in res["domains"].items()}}))
