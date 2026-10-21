"""w43-release: binding v0.4.3 RW1-RW4 table = the frozen verdict.py output + the manager's registered adjudications.

usage: python3 rw_final.py FOLLOWUP_DIR MAILBOX DECISION_LINE OUT_DIR
Reads (never rewrites, E212): FOLLOWUP_DIR/RW_VERDICT.json (frozen tools_session verdict.py), d6_N72/summary.json,
QGRAUP_D03_CENSUS.json, RW3T_VERDICT.json, RW3T_ACCEPT.json, FLOOR72_N72.json; MAILBOX line DECISION_LINE must be the
manager's 'DECISION output_integrity' line for this lane (quoted verbatim). Writes OUT_DIR/INTEGRITY_ADJUDICATION.json
(scope = exactly the raw integrity records the decision names; any other degenerate/missing/global finding stays a FAIL)
and OUT_DIR/RW_TABLE_FINAL.json (RW3 binding = 72 h pair no >10 % regression AND RW3T PASS AND frozen-A3 D6 binding PASS
AND integrity binding PASS; raw D6 / raw integrity FAIL are disclosed, never rewritten). CPU only, reads files.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

F, MB, LINE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
PREC_DOC = "docs/release/evidence/v033/0227_QGRAUP_integrity.md"   # in the pin tree (blob 21e541aa9f67)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name):
    return json.loads((F / name).read_text())


verdict, summary, census = load("RW_VERDICT.json"), load("d6_N72/summary.json"), load("QGRAUP_D03_CENSUS.json")
rw3t, accept, floor = load("RW3T_VERDICT.json"), load("RW3T_ACCEPT.json"), load("FLOOR72_N72.json")
line = MB.read_text().splitlines()[LINE - 1]
assert re.match(r"^20\d\d-\d\d-\d\dT[\d:.]+Z manager -> w43-release: DECISION output_integrity:", line), line
assert "REGISTERED EXCEPTION" in line and "QGRAUP" in line, line

# the raw integrity findings, exactly as the frozen scorer wrote them
raw = {}
for dom, v in summary["domains"].items():
    oi = v.get("output_integrity") or {}
    ga = oi.get("global_attrs") or {}
    raw[dom] = dict(pass_=oi.get("pass"), degenerate=oi.get("degenerate_fields") or {},
                    missing_vs_cpu=oi.get("missing_variables_vs_cpu") or {},
                    globals_missing=ga.get("missing_vs_cpu") or [], globals_by_frame=ga.get("by_frame") or {})
covered = [("d03", "QGRAUP", "wrfout_d03_2026-03-03_00:00:00")]   # the decision's scope: d03 QGRAUP at t72 only
uncovered = []
for dom, r in raw.items():
    for fld, recs in r["degenerate"].items():
        for rec in recs:
            if (dom, fld, rec.get("file")) not in covered:
                uncovered.append(dict(domain=dom, field=fld, record=rec))
    if r["missing_vs_cpu"] or r["globals_missing"] or r["globals_by_frame"]:
        uncovered.append(dict(domain=dom, missing_vs_cpu=r["missing_vs_cpu"], globals_missing=r["globals_missing"],
                              globals_by_frame=r["globals_by_frame"]))
seen = [(d, f, rec.get("file")) for d, r in raw.items() for f, recs in r["degenerate"].items() for rec in recs]
integrity_binding = "PASS" if not uncovered and seen == covered else "FAIL"

rows = census["rows"]
lead = {k: {i: rows[k][i][1:] for i in (70, 71, 72)} for k in rows}   # [min, max, nonzero cells] per source and lead
adj = dict(
    what="RW3 output-integrity adjudication (registered exception) for the v0.4.3 bundle; the frozen RW_VERDICT.json "
         "and the scorer summary stay byte-unchanged (E212); detector and thresholds unchanged",
    decision=dict(mailbox=str(MB), line=LINE, text=line),
    scope=dict(case="WN3 20260227_18z_a1 72 h, arm N72 (v0.4.3 sealed bundle, warm replay == A3 cold, 219 files)",
               covered=[dict(domain=d, field=f, frame=fr) for d, f, fr in covered],
               note="no other field, frame, domain, case or arm; any further finding stays a FAIL"),
    raw=dict(file=str(F / "d6_N72/summary.json"), sha256=sha(F / "d6_N72/summary.json"),
             output_integrity_pass=summary.get("output_integrity_pass"), findings=raw),
    census=dict(file=str(F / "QGRAUP_D03_CENSUS.json"), sha256=sha(F / "QGRAUP_D03_CENSUS.json"),
                columns="[min, max, nonzero cells] of d03 QGRAUP (kg/kg)", N72_v043=lead["N72"], O72_v042=lead["O72"],
                CPU_WRF=lead["CPU"], frames_with_gpu_constant_cpu_varying=census["degenerate"]),
    precedent=dict(
        cpu_vs_cpu="MAILBOX 2026-10-06T10:42:21Z wn3 (line ~5850): original CPU-WRF twin P0227 vs reference R through the "
                   "frozen wn3_integrity tool -> the IDENTICAL d03 QGRAUP flag at 2026-03-03_00 (P0227 exactly 0 in every "
                   "cell); manager 10:42:49Z EXPLAINED-predictability",
        doc=f"{PREC_DOC} (pin b979188128)", receipt_sha256="9767e9d30d0ff6e17e636b99110e7780d369d5dd950f4e1b3ba860346c95068c",
        receipt_present=False,
        v040="v0.4.0 B7 in the v0.4.2 gate: identical record (QGRAUP t72 GPU 0 vs CPU [0, 6.57752e-12])",
        v042="v0.4.2 B1: passed by one cell 1.2874e-12 kg/kg"),
    binding=dict(RW3_integrity_raw="FAIL", RW3_integrity_binding=integrity_binding, uncovered_findings=uncovered),
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "INTEGRITY_ADJUDICATION.json").write_text(json.dumps(adj, indent=1) + "\n")

r3 = verdict["RW3"]
timing = {k: dict(v043=v.get("v043"), v042=v.get("v042"), ratio=v.get("ratio"), regression_gt_10pct=v.get("regression_gt_10pct"))
          for k, v in r3["metrics_72h"].items()}
cover = {a: dict(samples=accept["slots"][a]["info"].get("samples"), wall_s=accept["slots"][a]["info"].get("wall_s"),
                 max_gap_s=accept["slots"][a]["info"].get("max_gap_s"), watch_cover=accept["slots"][a]["checks"].get("watch_cover"))
         for a in ("N72", "O72")}
m = rw3t["all_counted_24h"]["metrics"]["d03_step_median"]
rw3t_row = dict(verdict=rw3t.get("verdict"), prereg_sha256=rw3t.get("prereg_sha256"), counted=rw3t.get("counted"),
                primary="d03_step_median", N=m["N"], O=m["O"], pooled_N_over_O=m["pooled_N_over_O"], S=m["S"],
                rule="pooled + S <= 1.10", not_counted=sorted(set(accept["slots"]) - set(rw3t.get("counted") or [])
                                                               - {"N4", "O4", "N5", "O5"}),
                rejected_attempts=sorted(accept.get("rejected_attempts") or {}))
d6_binding = floor.get("binding72_A3")
rw3_ok = (not r3["regression_72h"] and rw3t.get("verdict") == "PASS" and d6_binding == "PASS"
          and integrity_binding == "PASS" and r3["b1_replay_ok"] and r3["o72_replay_ok"])
table = dict(
    what="v0.4.3 bundle RW1-RW4 real-world gate, binding reading (derived; sources hashed below)",
    sources={n: sha(F / n) for n in ("RW_VERDICT.json", "RW3T_VERDICT.json", "RW3T_ACCEPT.json", "FLOOR72_N72.json",
                                     "d6_N72/summary.json", "QGRAUP_D03_CENSUS.json")}
            | {"INTEGRITY_ADJUDICATION.json": sha(OUT / "INTEGRITY_ADJUDICATION.json")},
    RW1=verdict["RW1"]["verdict"], RW2_RW4=verdict["RW2_RW4"]["verdict"],
    RW3=dict(verdict="PASS" if rw3_ok else "FAIL", timing_72h_N72_vs_O72=timing,
             timing_72h_disclosure=dict(basis="O72 = v0.4.2 sealed bundle, same session", watch_cover=cover,
                                        note="the 72 h arms miss the 24 h accept rule's watch_cover (samples < wall - 5 s); "
                                             "disclosed secondary, RW3T decides"),
             rw3t_interleaved_24h=rw3t_row, d6_72h_raw=r3["d6_72h_raw"], d6_72h_binding_A3=d6_binding,
             d6_72h_floor_rows=floor.get("rows"), integrity_raw="FAIL" if r3["output_integrity"] is False else r3["output_integrity"],
             integrity_binding=integrity_binding),
    ICLOUD0=verdict["ICLOUD0"]["verdict"],
    E41=dict(default=verdict["E41"], icloud0=verdict["ICLOUD0"]["e41"]),
)
(OUT / "RW_TABLE_FINAL.json").write_text(json.dumps(table, indent=1) + "\n")
print(json.dumps(dict(RW1=table["RW1"], RW2_RW4=table["RW2_RW4"], RW3=table["RW3"]["verdict"],
                      RW3_d6_raw=table["RW3"]["d6_72h_raw"], RW3_d6_binding=d6_binding, RW3_integrity_raw=table["RW3"]["integrity_raw"],
                      RW3_integrity_binding=integrity_binding, RW3T=rw3t_row["verdict"], ICLOUD0=table["ICLOUD0"],
                      uncovered=len(uncovered))))
