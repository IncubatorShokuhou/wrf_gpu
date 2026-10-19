"""RW3 72 h D6 floor reading with the FROZEN v0.3.3 A3 rule on the NEW scorer output (w42-scorer scripts/wn3_score.py).

usage: python3 floor72_new.py SCORE_OUT_DIR OUT_JSON
Same rule and C reference as the v0.4.1 floor72.py (w41-bundle041; A3 pre-registered 2026-10-07): a frozen-D6 breach
X > limit is FLOOR-LIMITED iff C > limit OR C/X >= 0.75, where C = original CPU-WRF 0227 vs its own +-1-ulp IC twin for
the same domain/field/lead, taken VERBATIM from FLOOR72_FROM_EXISTING_C.json; a breach without a recorded C is REAL.
Only input format changes: breaches = summary.json domains[d].policy_failures of kind 'frame_tolerance' (detail
[{metric, value, limit}]); any other policy-failure kind, a missing required field, a GPU-only nonfinite or an incomplete
pairing (paired_file_count != 73) is never floor-limited. binding72_A3 = PASS iff every breach is FLOOR-LIMITED and
coverage is complete; output_integrity is reported beside it (separate gate).
"""
import hashlib
import json
import sys
from pathlib import Path

REF = Path("<USER_HOME>/wrf_gpu2_lanes/mass-opus/SCIENCE_G2_W3P0227_ACTUAL_20261009/FLOOR72_FROM_EXISTING_C.json")
case, out = Path(sys.argv[1]), Path(sys.argv[2])
ref = json.loads(REF.read_text())
C = {(r["domain"], r["field"], r["lead_h"]): r["C"] for r in ref["rows"]}
summary = json.loads((case / "summary.json").read_text())
rows, real, complete, other = [], 0, True, []
for dom in ("d01", "d02", "d03"):
    d = summary["domains"][dom]
    complete &= d["pairing"]["paired_file_count"] == 73 and not d["missing_required_fields"] and not d["gpu_only_nonfinite"]
    for f in d["policy_failures"]:
        if f.get("kind") != "frame_tolerance":
            other.append(dict(domain=dom, **f))
            real += 1
            continue
        for chk in f["detail"]:
            x, lim = chk["value"], chk["limit"]
            c = C.get((dom, f["field"], f["lead_h"]))
            floor = c is not None and (c > lim or c / x >= 0.75)
            real += not floor
            rows.append(dict(domain=dom, field=f["field"], lead_h=f["lead_h"], metric=chk.get("metric"), X=x, limit=lim,
                             C=c, C_over_X=(c / x if c is not None else None),
                             status="FLOOR-LIMITED" if floor else "REAL FAIL"))
res = dict(case=str(case), scorer="scripts/wn3_score.py (w42-scorer) summary.json",
           rule="originalA3 C>limit OR C/X>=.75 (frozen); C verbatim from the v0.3.3 W3P 0227 receipt",
           C_reference=dict(path=str(REF), sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
                            original_source=ref["C_reference"]),
           rows=rows, other_policy_failures=other,
           counts={"FLOOR-LIMITED": len(rows) - sum(r["status"] == "REAL FAIL" for r in rows), "REAL FAIL": real},
           coverage_complete=bool(complete), raw72="PASS" if not rows and not other else "FAIL",
           binding72_A3="PASS" if complete and real == 0 else "FAIL",
           output_integrity_pass=summary.get("output_integrity_pass"))
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("raw72", "binding72_A3", "counts", "coverage_complete", "output_integrity_pass")}))
for r in rows:
    print(r["domain"], r["field"], r["lead_h"], round(r["X"], 4), "C", r["C"] and round(r["C"], 4), r["status"])
for o in other:
    print("OTHER", json.dumps(o)[:200])
