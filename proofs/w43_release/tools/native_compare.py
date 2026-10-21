"""C1/C2/C2c of RELEASE_PLAN.md: compare two native_ids.py reports (v0.4.2 bundle vs v0.4.3 bundle), per program.

usage: python3 native_compare.py V042_JSON V043_JSON OUT_JSON
Programs are matched by (family dir, optimized-HLO SHA) and, independently, by native record-2 SHA; the lowered-HLO name
(AotMeta.hlo_sha256) and the cheap key are expected to differ (Pallas source paths / gpuwrf_version in the key) and are
only reported. PASS = same program families with the same program count, and for every v0.4.2 program a v0.4.3 program
with the SAME optimized HLO, the SAME record-2 (native cubin) SHA and the same path-normalized SHA for every Riegeli record.
"""
import json, sys
from pathlib import Path

a, b = (json.loads(Path(x).read_text()) for x in sys.argv[1:3])
out = Path(sys.argv[3])
res = {"v042": sys.argv[1], "v043": sys.argv[2], "families": {}}
ok = set(a["programs"]) == set(b["programs"])
res["family_sets_equal"] = ok
for fam in sorted(set(a["programs"]) | set(b["programs"])):
    pa, pb = a["programs"].get(fam, []), b["programs"].get(fam, [])
    by_hlo = {x.get("optimized_hlo_sha256"): x for x in pb}
    rows = []
    for x in pa:
        y = by_hlo.get(x.get("optimized_hlo_sha256"))
        row = {"v042_lowered": x["aliases"][0][:16], "optimized_hlo_sha256": x.get("optimized_hlo_sha256"),
               "v042_native_record2": x.get("native_record2_sha256"), "match": y is not None}
        if y is not None:
            ra = [r["sha256_pathnorm"] for r in x.get("records", [])]
            rb = [r["sha256_pathnorm"] for r in y.get("records", [])]
            row.update(v043_lowered=y["aliases"][0][:16], v043_native_record2=y.get("native_record2_sha256"),
                       native_equal=x.get("native_record2_sha256") == y.get("native_record2_sha256"),
                       elfs_equal=x.get("native_elfs") == y.get("native_elfs"),
                       n_records=(len(ra), len(rb)), records_pathnorm_equal=ra == rb,
                       records_raw_differ=[i for i, (r, s) in enumerate(zip(x.get("records", []), y.get("records", [])))
                                           if r["sha256"] != s["sha256"]],
                       cheap_key=(x.get("meta", {}).get("cheap_key"), y.get("meta", {}).get("cheap_key")))
        rows.append(row)
    native_only = sorted({x.get("native_record2_sha256") for x in pa} ^ {y.get("native_record2_sha256") for y in pb})
    fam_ok = (len(pa) == len(pb) and all(r["match"] and r["native_equal"] and r["elfs_equal"] and r["records_pathnorm_equal"]
                                         for r in rows))
    res["families"][fam] = {"count": (len(pa), len(pb)), "programs": rows, "native_set_symmetric_diff": native_only,
                            "pass": fam_ok}
    ok = ok and fam_ok
res["verdict"] = "PASS" if ok else "FAIL"
out.write_text(json.dumps(res, indent=1) + "\n")
print("NATIVE_IDENTITY", res["verdict"], {f: (v["count"], v["pass"]) for f, v in res["families"].items()})
