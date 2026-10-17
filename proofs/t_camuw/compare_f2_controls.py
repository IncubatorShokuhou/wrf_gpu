"""Compare complete same-work F2 receipts; does not assign scientific verdicts."""
import json
from pathlib import Path
import sys

candidate, control, output = (Path(x).resolve() for x in sys.argv[1:4])
a, b = (json.loads(p.read_text()) for p in (candidate, control))
original = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
assert a["input_sha256"] == b["input_sha256"]
assert a["work"] == b["work"]
assert a["platform"] == b["platform"] == "cpu"
assert a["work"]["disable_guards"] is True
assert [r["end_step"] for r in a["steps"]] == [r["end_step"] for r in b["steps"]]
comparisons = []
endpoints = []
if original:
    assert all(not r.get("negative_count", 0) for r in original["records"] if r["hour"] == 0)
for x, y in zip(a["steps"], b["steps"]):
    assert x["work_counts"] == y["work_counts"]
    assert x["nonfinite"] == y["nonfinite"] == 0
    rows = []
    for p, q in zip(x["f2"]["records"], y["f2"]["records"]):
        assert (p["field"], p["phase"]) == (q["field"], q["phase"])
        if p["negative_count"] or q["negative_count"] or p["nonfinite_count"] or q["nonfinite_count"]:
            rows.append(dict(phase=p["phase"], field=p["field"], candidate=p, control=q,
                exact_count_location_match=all(p[k] == q[k] for k in
                    ("negative_count", "ring0_count", "interior_count", "nonfinite_count"))))
    guards = []
    for name in sorted(x["guard_candidates"].keys() | y["guard_candidates"].keys()):
        zero = dict(nonfinite=0, out_of_range=0, repaired=0)
        guards.append(dict(field=name, candidate=x["guard_candidates"].get(name, zero),
                           control=y["guard_candidates"].get(name, zero)))
    for row in (x, y):
        assert all(not v for repair in row["f2_events_and_water"]["guard_water"].values()
                   for field in repair.values() for v in field.values())
    comparisons.append(dict(end_step=x["end_step"], candidate_guard_total=x["guard_total"],
        control_guard_total=y["guard_total"], guards=guards, f2=rows))
    if original and x["end_step"] % 200 == 0:
        # Strict mode leaves pre_boundary_guard state unchanged until the next
        # step_entry. Cumulative B(1..N)-E(0..N-1) telescopes to endpoint N
        # because the independently checked initial state has zero negatives.
        by_arm = [{(r["phase"], r["field"]): r for r in z["f2"]["records"]} for z in (x, y)]
        for ref in original["records"]:
            if ref["hour"] != x["end_step"] // 200 or "negative_count" not in ref:
                continue
            counts = []
            for values in by_arm:
                start = values["step_entry", ref["field"]]
                end = values["pre_boundary_guard", ref["field"]]
                assert start["present_observations"] == end["present_observations"] == x["end_step"]
                counts.append({k: end[k] - start[k] for k in
                    ("negative_count", "ring0_count", "interior_count", "nonfinite_count")})
                assert all(v >= 0 for v in counts[-1].values())
            endpoints.append(dict(hour=ref["hour"], field=ref["field"], candidate=counts[0],
                control=counts[1], original={k:ref[k] for k in counts[0]}))
output.write_text(json.dumps(dict(candidate_source=a["source"], control_source=b["source"],
    input_sha256=a["input_sha256"], work=a["work"], comparisons=comparisons,
    endpoint_negative_counts=endpoints,
    scope="same-work regression attribution, not WRF fidelity; scientific disposition required"), indent=2) + "\n")
for r in comparisons:
    print("STEP", r["end_step"], "aggregate", r["candidate_guard_total"], r["control_guard_total"])
for r in comparisons[-1]["f2"]:
    print(r["phase"], r["field"], *[(v["negative_count"], v["ring0_count"], v["interior_count"],
        v["minimum_negative"]) for v in (r["candidate"], r["control"])])
