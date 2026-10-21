"""Aggregate per-case teacher verdicts (verdict.py, TEACHER_PREREG v3) into TEACHER_QUAL.json + the markdown table body.

usage: python3 teacher_qual.py OUT_JSON OUT_MD CASE=forcing:SCORE_DIR[:C_DIR] ...
SCORE_DIR = score_case.sh output (x/, VERDICT.json); C_DIR = twin-vs-reference scorer output or '-'.
Per case x domain x field class: verdict status (PASS / FLOOR-LIMITED / FAIL) plus worst X/L per tau block 0-23, 24-47, 48-72.
Wind tau>40 h row (ALISIOS acceptance item 5): classes wind10 + wind3d, leads 41-72 only, status per case x domain;
QUALIFIED only if every case x domain is PASS or FLOOR-LIMITED there and coverage/integrity are complete, else HELD.
"""
import json, sys
from pathlib import Path

import importlib.util
spec = importlib.util.spec_from_file_location("verdict", Path(__file__).with_name("verdict.py"))
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

BLOCKS = [("0-23", range(0, 24)), ("24-47", range(24, 48)), ("48-72", range(48, 73))]
RANK = {"PASS": 0, "FLOOR-LIMITED": 1, "FAIL": 2, "INCOMPLETE": 3}
WIND = {"wind10", "wind3d"}


def worst(a, b):
    return max(a, b, key=RANK.get)


def case_rows(case, forcing, sdir, cdir):
    v = json.loads((Path(sdir) / "VERDICT.json").read_text())
    out = {"case": case, "forcing": forcing, "case_teacher": v["case_teacher"], "has_twin": cdir is not None, "domains": {}}
    for dom, dv in v["domains"].items():
        X, _ = V.rows_of(Path(sdir) / "x", dom)
        ratio = {}
        for (f, t, m), (x, lim) in X.items():
            if lim is None or m in V.LOWER or not V.fin(x, lim) or lim == 0:
                continue
            for b, rg in BLOCKS:
                if t in rg:
                    k = (V.cls(f), b)
                    ratio[k] = max(ratio.get(k, 0.0), x / lim)
        classes = {}
        for c, _ in V.CLASSES + [("other", None)]:
            st = dv["classes"].get(c, {}).get("status")
            if st is None and not any(k[0] == c for k in ratio):
                continue
            classes[c] = {"status": st or "PASS",
                          **{f"max_X_over_L_{b}": round(ratio[(c, b)], 4) for b, _ in BLOCKS if (c, b) in ratio}}
        wind = "PASS"
        for br in v["breaches"]:
            if br["domain"] == dom and V.cls(br["field"]) in WIND and br["lead_h"] > 40:
                wind = worst(wind, br["status"])
        if not dv["coverage_complete"] or dv["integrity_pass"] is None:
            wind = "INCOMPLETE"
        out["domains"][dom] = {"teacher": dv["teacher"], "bounded_worst": dv["bounded_worst"], "integrity_pass": dv["integrity_pass"],
                               "coverage_complete": dv["coverage_complete"], "classes": classes, "wind_tau_gt40": wind}
    return out


def main():
    oj, om, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    cases = []
    for s in specs:
        case, rest = s.split("=", 1)
        parts = rest.split(":")
        forcing, sdir, cdir = parts[0], parts[1], (parts[2] if len(parts) > 2 and parts[2] != "-" else None)
        if not (Path(sdir) / "VERDICT.json").exists():
            cases.append({"case": case, "forcing": forcing, "case_teacher": "INCOMPLETE", "domains": {}, "missing": str(sdir)})
            continue
        cases.append(case_rows(case, forcing, sdir, cdir))
    wind_all = [d["wind_tau_gt40"] for c in cases for d in c["domains"].values()] + ["INCOMPLETE" for c in cases if not c["domains"]]
    wind_q = "QUALIFIED" if wind_all and all(w in ("PASS", "FLOOR-LIMITED") for w in wind_all) else "HELD"
    res = {"rule": "TEACHER_PREREG v3", "cases": cases, "wind_tau_gt40_h": {"verdict": wind_q, "per_case_domain": {
        f"{c['case']}/{d}": dv["wind_tau_gt40"] for c in cases for d, dv in c["domains"].items()}}}
    Path(oj).write_text(json.dumps(res, indent=1) + "\n")
    L = ["| case | forcing | twin | domain | teacher | " + " | ".join(c for c, _ in V.CLASSES) + " | wind τ>40 h |",
         "|" + "---|" * (6 + len(V.CLASSES))]
    for c in cases:
        if not c["domains"]:
            L.append(f"| {c['case']} | {c['forcing']} | - | all | INCOMPLETE |" + " |" * (len(V.CLASSES) + 1))
            continue
        for d, dv in c["domains"].items():
            cells = [dv["classes"].get(k, {}).get("status", "n/a") for k, _ in V.CLASSES]
            L.append(f"| {c['case']} | {c['forcing']} | {'yes' if c['has_twin'] else 'no'} | {d} | {dv['teacher']} | "
                     + " | ".join(cells) + f" | {dv['wind_tau_gt40']} |")
    L.append(f"\nWind τ>40 h (leads 41–72, wind10 + wind3d, all cases × domains): **{wind_q}**")
    Path(om).write_text("\n".join(L) + "\n")
    print(json.dumps({"wind_tau_gt40_h": wind_q, "cases": {c["case"]: c["case_teacher"] for c in cases}}))


if __name__ == "__main__":
    main()
