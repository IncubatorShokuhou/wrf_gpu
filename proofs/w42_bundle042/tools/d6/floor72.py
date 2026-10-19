"""RW3 72 h D6 floor reading with the FROZEN v0.3.3 A3 rule (no new threshold, no recomputed C).

usage: python3 floor72.py CASE_DIR OUT_JSON
Rule (mass-opus A3, pre-registered 2026-10-07, as applied in FLOOR72_FROM_EXISTING_C.json of the v0.3.3 W3P 0227 gate):
a frozen-D6 breach X > limit is FLOOR-LIMITED iff C > limit OR C/X >= 0.75, where C = original CPU-WRF 0227 vs its own
+-1-ulp IC twin (same executable) for the same domain/field/lead; else REAL FAIL. C is taken VERBATIM from that receipt
(its source file cpupair_P_vs_R.json fcd8b975 was removed in the 10-09 cleanup); a breach without a recorded C is REAL.
binding72_A3 = PASS iff every breach is FLOOR-LIMITED and the frozen D6 has 0 nonfinite and complete coverage.
"""
import hashlib
import json
import sys
from pathlib import Path

REF = Path("<USER_HOME>/wrf_gpu2_lanes/mass-opus/SCIENCE_G2_W3P0227_ACTUAL_20261009/FLOOR72_FROM_EXISTING_C.json")
case, out = Path(sys.argv[1]), Path(sys.argv[2])
ref = json.loads(REF.read_text())
C = {(r["domain"], r["field"], r["lead_h"]): r["C"] for r in ref["rows"]}
rows, real, complete = [], 0, True
for dom in ("d01", "d02", "d03"):
    s = json.loads((case / f"SCORING_IO_{dom}.json").read_text())
    complete &= s["paired_frames"] == 73 and not s["incomplete"] and s["nonfinite"] == 0
    for f in s["failures"]:
        for chk in f["checks"]:
            x, lim = chk["value"], chk["limit"]
            c = C.get((dom, f["field"], f["lead_h"]))
            floor = c is not None and (c > lim or c / x >= 0.75)
            real += not floor
            rows.append(dict(domain=dom, field=f["field"], lead_h=f["lead_h"], metric=chk.get("metric"), X=x, limit=lim,
                             C=c, C_over_X=(c / x if c is not None else None),
                             status="FLOOR-LIMITED" if floor else "REAL FAIL"))
res = dict(case=str(case), rule="originalA3 C>limit OR C/X>=.75 (frozen); C verbatim from the v0.3.3 W3P 0227 receipt",
           C_reference=dict(path=str(REF), sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
                            original_source=ref["C_reference"]),
           rows=rows, counts={"FLOOR-LIMITED": len(rows) - real, "REAL FAIL": real}, coverage_complete=bool(complete),
           raw72="PASS" if not rows else "FAIL",
           binding72_A3="PASS" if complete and real == 0 else "FAIL")
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("raw72", "binding72_A3", "counts", "coverage_complete")}))
for r in rows:
    print(r["domain"], r["field"], r["lead_h"], round(r["X"], 4), "C", r["C"] and round(r["C"], 4), r["status"])
