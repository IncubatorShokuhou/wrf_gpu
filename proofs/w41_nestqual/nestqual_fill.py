"""Lane w41-nestqual: collect the per-arm NEST_GATE.json verdicts into the repo receipt NESTQUAL_SUMMARY.json and the
text for the v0.4.1 docs placeholders (w41-relnotes: {{NESTQUAL_*}} in SUPPORT_MATRIX_v0.4.1.md / RELEASE_NOTES_v0.4.1.md).
Numbers come only from NEST_GATE.json (gate_summary.py over the frozen-manifest release scorer); an arm without a
NEST_GATE.json is reported as not run / not scored, never as a pass. Copies the small receipts (NEST_GATE, PREPARE,
GPU_STAGE, cpu_run) into proofs/w41_nestqual/<arm>/ and the D6 jsons (gzip) into the lane dir for the critic.
usage: python nestqual_fill.py <cases_dir> [--issue 20260707_18z] [--write]
"""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ARMS = {"C2": ("YSU", "YSU"), "C3": ("SHINHONG", "Shin-Hong"), "C4": ("ACM2", "ACM2"), "C7": ("EEPS", "EEPS")}
SINGLE_DOMAIN_E41 = "FAIL"  # v0.4.0 composites c02/c07/c05/c10 (SUPPORT_MATRIX rows 1/11/7/16): E41-disclosed
HERE = Path(__file__).resolve().parent
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-nestqual")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def arm_result(case):
    gate = case / "NEST_GATE.json"
    if not gate.exists():
        status = ("CPU twin sealed, not scored" if (case / "cpu_run.json").exists()
                  else "CPU twin not run" if case.exists() else "no case")
        return dict(verdict="NOT_RUN", status=status)
    g = json.loads(gate.read_text())
    doms, worst, fails = {}, None, []
    for d, v in g["domains"].items():
        if v.get("verdict") == "MISSING":
            doms[d] = dict(verdict="MISSING")
            continue
        ratios = {n: w for n, w in v["worst_lead_rmse"].items() if w.get("ratio") is not None}
        top = max(ratios.items(), key=lambda kv: kv[1]["ratio"]) if ratios else None
        doms[d] = dict(verdict=v["verdict"], paired_frames=v["paired_frames"], compared_fields=v["compared_fields"],
                       bounded_fields=v["bounded_fields"], failures=v["failures"], nonfinite=v["nonfinite"],
                       coverage_complete=v["coverage_complete"], missing_vs_cpu=v["missing_vs_cpu"],
                       worst_ratio=dict(field=top[0], **top[1]) if top else None,
                       t2_worst=v["worst_lead_rmse"].get("T2"),
                       report_only_constant_vs_varying=sorted(v["report_only_constant_vs_varying"]),
                       output_integrity_pass=v.get("output_integrity_pass"),
                       global_attrs_missing=v.get("global_attrs_missing"),
                       d6_json_sha256=v["d6_json_sha256"])
        if top and (worst is None or top[1]["ratio"] > worst["ratio"]):
            worst = dict(domain=d, field=top[0], **top[1])
        fails += [dict(domain=d, field=n, **w) for n, w in v["worst_lead_rmse"].items()
                  if w.get("ratio") is not None and w["ratio"] > 1]
    return dict(verdict=g["verdict"], input_identity=g["input_identity"], domains=doms, worst=worst,
                over_limit=sorted(fails, key=lambda f: -f["ratio"]), nest_gate_sha256=sha(gate),
                cpu_timing=g["cpu"]["timing"], gpu=g["gpu"])


def cell(r):
    if r["verdict"] == "NOT_RUN":
        return f"Not run ({r['status']}); not validated inside nests"
    d = r["domains"]
    if r["verdict"] == "PASS":
        w = r["worst"]
        t2 = "/".join(f"{d[k]['t2_worst']['rmse']:.3f}" for k in ("d01", "d02", "d03"))
        nb = d["d01"]["bounded_fields"]
        deg = "; ".join(f"{k}: {', '.join(v['report_only_constant_vs_varying'])}" for k, v in d.items()
                        if v.get("report_only_constant_vs_varying"))
        return (f"**PASS** d01–d03: {d['d01']['paired_frames']} hourly frames per domain, 0 of {nb} bounded fields "
                f"over limit, NF0; highest worst-hour RMSE/limit {w['ratio']:.3f} ({w['field']} {w['domain']} +{w['lead_h']} h); "
                f"T2 worst-hour RMSE {t2} K"
                + (f"; output fields constant on the GPU where CPU-WRF varies (disclosed, not bounded): {deg}" if deg else ""))
    if r["verdict"] == "FAIL":
        lst = "; ".join(f"{f['domain']} {f['field']} {f['rmse']:.3g} > {f['limit']:g} (+{f['lead_h']} h)"
                        for f in r["over_limit"][:4]) or "; ".join(  # non-RMSE failures (NF, coverage, inventory)
            f"{k} {v['verdict']} (failures {v.get('failures')}, non-finite {v.get('nonfinite')}, "
            f"coverage {'complete' if v.get('coverage_complete') else 'INCOMPLETE'})" for k, v in d.items()
            if v["verdict"] != "PASS")
        return f"**FAIL**, stays experimental inside nests: {lst}"
    return f"Incomplete ({', '.join(k + ' ' + v['verdict'] for k, v in d.items())}); not validated inside nests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=Path)
    ap.add_argument("--issue", default="20260707_18z")
    ap.add_argument("--write", action="store_true", help="copy receipts + write NESTQUAL_SUMMARY.json")
    a = ap.parse_args()
    res, fill = {}, {}
    for arm, (key, name) in ARMS.items():
        case = a.cases / f"{a.issue}_{arm}"
        r = res[arm] = dict(scheme=name, case=str(case), **arm_result(case))
        fill[f"NESTQUAL_{key}"] = cell(r)
        fill[f"NESTQUAL_{key}_E41"] = (f"Not measured for the nested programs [U] (compiled and run by ALISIOS G2); "
                                       f"single-domain program {SINGLE_DOMAIN_E41}, disclosed in v0.4.0")
        if a.write and (case / "NEST_GATE.json").exists():
            dst = HERE / arm
            dst.mkdir(exist_ok=True)
            for f in ("NEST_GATE.json", "PREPARE.json", "GPU_STAGE.json", "cpu_run.json"):
                shutil.copy2(case / f, dst / f)
            score = Path(json.loads((case / "NEST_GATE.json").read_text())["scorer"]["score_dir"])
            for f in ("d01.md", "d02.md", "d03.md", "resources.json"):     # small scorer reports -> repo
                shutil.copy2(score / f, dst / f"wn3_{f}")
            keep = LANE / "receipts" / arm                                  # full scorer jsons (~10 MB each) -> lane, gzip
            keep.mkdir(parents=True, exist_ok=True)
            for f in ("summary.json", "d01.json", "d02.json", "d03.json",
                      "d01_early_hours.json", "d02_early_hours.json", "d03_early_hours.json"):
                with open(score / f, "rb") as i, gzip.open(keep / f"wn3_{f}.gz", "wb") as o:
                    shutil.copyfileobj(i, o)
    fill["NESTQUAL_RECEIPT"] = "[w41-nestqual](../../proofs/w41_nestqual/NESTQUAL_SUMMARY.json)"
    eng = sorted({e.get("manifest_engine") or e.get("source_head") for r in res.values()
                  for e in [(r.get("gpu") or {}).get("engine") or {}] if e})
    passed = [res[a_]["scheme"] for a_ in ARMS if res[a_]["verdict"] == "PASS"]
    other = [f"{res[a_]['scheme']} {res[a_]['verdict'].replace('_', ' ').lower()}" for a_ in ARMS
             if res[a_]["verdict"] != "PASS"]
    fill["NESTQUAL_RESULT"] = (
        f"Nested 24 h comparison vs original CPU-WRF twins (G2 Tenerife 3-nest, issue {a.issue}"
        + (f", GPU {'; '.join(eng)}" if eng else "") + "): "
        + (f"{', '.join(passed)} pass on d01–d03 and are validated inside nests in this configuration" if passed
           else "no scheme scored yet" if all(res[a_]["verdict"] == "NOT_RUN" for a_ in ARMS) else "no scheme passed")
        + (f"; {', '.join(other)}" if other else "") + ".")
    out = dict(schema="w41-nestqual.summary.v1", issue=a.issue, gpu_engine=eng,
               gate="D6 24 h d01-d03 vs original CPU-WRF 4.7.1 twin (same namelist/inputs), merged bounded scorer "
                    "scripts/wn3_score.py --sequential-domains, frozen D6 manifest, 0 bounded misses, NF0, every CPU "
                    "variable on all 25 frames, E184 input + output identity (rd-s6 SHA manifest)",
               arms=res, fill=fill)
    if a.write:
        (HERE / "NESTQUAL_SUMMARY.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(fill, indent=1))


if __name__ == "__main__":
    main()
