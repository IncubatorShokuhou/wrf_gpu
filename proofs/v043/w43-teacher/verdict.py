"""Teacher-qualification verdict (pre-registered rule TEACHER_PREREG.md v2, §2) from frozen-scorer per-lead rows.

usage: python3 verdict.py CASE X_DIR [C_DIR|-] INTEGRITY_JSON|- OUT_JSON
X_DIR = wn3_score.py --hours 72 output for GPU vs CPU-reference (d01.json d02.json d03.json + summary.json);
C_DIR = the same scorer output for CPU-twin vs CPU-reference ('-' = the case has no valid twin);
INTEGRITY_JSON = {"pass": bool, ...} from the frozen all-frame G-INT + annex run ('-' = not run -> INCOMPLETE).
Per (domain, field, lead, metric) with a D6 limit: strict24 (tau 0-24, no floor), then tau 25-72 floor X <= (4/3) C.
Upper-bound metrics get the floor; finite_pair_fraction_min (a lower bound) must always pass. Non-finite X/C/limit -> FAIL (E200).
"""
import json, math, sys
from fractions import Fraction
from pathlib import Path

K = Fraction(4, 3)
LOWER = {"finite_pair_fraction_min"}
CLASSES = [("T2/Q2", {"T2", "Q2", "TH2", "RH2"}), ("wind10", {"U10", "V10"}), ("wind3d", {"U", "V", "W"}),
           ("T/theta", {"T", "THM"}), ("pressure/mass", {"PSFC", "P", "PB", "PH", "PHB", "MU", "MUB", "P_HYD"}),
           ("precip", {"RAINNC", "RAINC", "SNOWNC", "GRAUPELNC", "HAILNC", "RAINSH"}),
           ("moisture/hydrometeors", {"QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN"}),
           ("radiation/surface fluxes", {"SWDOWN", "GLW", "OLR", "HFX", "LH", "QFX", "UST", "PBLH", "SWNORM", "GRDFLX"}),
           ("land", {"TSK", "TSLB", "SMOIS", "SH2O", "SNOW", "SNOWH", "CANWAT", "SST"})]


def cls(field):
    return next((c for c, s in CLASSES if field in s), "other")


def fin(*v):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def rows_of(score_dir, dom):
    d = json.loads((Path(score_dir) / f"{dom}.json").read_text())
    out = {}
    for f, fs in d["field_summaries"].items():
        for bl in fs.get("by_lead") or []:
            tr = bl.get("tolerance_result") or {}
            for m, chk in (tr.get("checks") or {}).items():
                out[(f, bl["lead_h"], m)] = (chk.get("value"), chk.get("limit"))
    leads = d["pairing"]["common_leads_h"]
    return out, leads


def verdict(case, xdir, cdir, integ):
    res, table = {"case": case, "rule": "TEACHER_PREREG v2: strict24 + X<=max(L,4/3 C) own twin", "domains": {}}, []
    # pass None = G-INT tool error / not run -> INCOMPLETE (PREREG section 1), never FAIL
    integ_ok = None if integ is None or integ.get("pass") is None else bool(integ["pass"])
    for dom in ("d01", "d02", "d03"):
        X, xl = rows_of(xdir, dom)
        C, cl = rows_of(cdir, dom) if cdir else ({}, None)
        complete = xl == list(range(73)) and (cdir is None or cl == list(range(73)))
        fields = {}
        for (f, t, m), (x, lim) in sorted(X.items()):
            if lim is None:
                continue
            st = fields.setdefault(f, {"class": cls(f), "status": "PASS", "worst": None})
            c = C.get((f, t, m), (None, None))[0] if cdir else None
            if not fin(x, lim) or (c is not None and not fin(c)):
                s = "FAIL"; why = "nonfinite"
            elif m in LOWER:
                s = "PASS" if x >= lim else "FAIL"; why = "lower-bound metric"
            elif x <= lim:
                continue
            elif t <= 24:
                s = "FAIL"; why = "strict24"
            elif c is None:
                s = "FAIL"; why = "no own twin"
            elif Fraction(x) <= K * Fraction(c):
                s = "FLOOR-LIMITED"; why = "X<=4/3 C"
            else:
                s = "FAIL"; why = "X>max(L,4/3 C)"
            if s == "PASS":
                continue
            row = dict(field=f, lead_h=t, metric=m, X=x, L=lim, C=c, X_over_C=(x / c if c else None), status=s, why=why,
                       frozen_A3=(None if c is None else ("FLOOR-LIMITED" if (c > lim or c / x >= 0.75) else "REAL FAIL")))
            table.append(dict(domain=dom, **row))
            rank = {"PASS": 0, "FLOOR-LIMITED": 1, "FAIL": 2}
            if rank[s] > rank[st["status"]] or st["worst"] is None or (rank[s] == rank[st["status"]] and
                                                                         (row["X_over_C"] or 0) > (st["worst"]["X_over_C"] or 0)):
                st["worst"] = row
            st["status"] = max(st["status"], s, key=rank.get)
        by_class = {}
        for f, st in fields.items():
            cur = by_class.setdefault(st["class"], {"status": "PASS", "fields": {}})
            cur["fields"][f] = st["status"]
            cur["status"] = max(cur["status"], st["status"], key={"PASS": 0, "FLOOR-LIMITED": 1, "FAIL": 2}.get)
        worst = max([v["status"] for v in by_class.values()] or ["PASS"], key={"PASS": 0, "FLOOR-LIMITED": 1, "FAIL": 2}.get)
        teacher = ("INCOMPLETE" if (not complete or integ_ok is None) else
                   "TEACHER-OK" if worst != "FAIL" and integ_ok else "FAIL")
        res["domains"][dom] = dict(coverage_complete=complete, bounded_worst=worst, integrity_pass=integ_ok,
                                   teacher=teacher, classes=by_class,
                                   fields={f: {k: v for k, v in st.items()} for f, st in fields.items() if st["status"] != "PASS"})
    res["breaches"] = table
    res["case_teacher"] = ("INCOMPLETE" if any(d["teacher"] == "INCOMPLETE" for d in res["domains"].values()) else
                           "TEACHER-OK" if all(d["teacher"] == "TEACHER-OK" for d in res["domains"].values()) else "FAIL")
    return res


if __name__ == "__main__":
    case, xdir, cdir, ij, out = sys.argv[1:6]
    integ = None if ij == "-" else json.loads(Path(ij).read_text())
    r = verdict(case, xdir, None if cdir == "-" else cdir, integ)
    Path(out).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({"case": case, "case_teacher": r["case_teacher"],
                      **{d: v["teacher"] + "/" + v["bounded_worst"] for d, v in r["domains"].items()}}))
