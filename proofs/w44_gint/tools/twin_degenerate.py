"""Is the CPU-WRF twin ALSO all-zero where the GPU is degenerate? From retained receipts only (no raw wrfout), CPU only.

INFERENCE ONLY (critic cr-w44-gint F2): binary64 sums can erase a positive REAL32 value (a 1e-30 twin value passes the test
below), so `twin_all_zero` is "statistics consistent with an all-zero twin", never proof of exact zero; use raw values or the
direct frozen integrity (gint_cached.py) for exact claims.

For each case with an own CPU twin (w43-teacher score/C_<case>: twin vs the same CPU reference, frozen wn3_score) and each
(domain, field, frame) the frozen all-frame G-INT lists as GPU-degenerate (score/<case>/integ/<dom>.json, GPU constant 0 while
the reference varies): for a field that is >= 0 everywhere, GPU == 0 everywhere gives mean(ref) = mae_GPU (= -bias_GPU, checked),
and then mean(twin) = bias_twin + mean(ref). The twin is all-zero iff mean(twin) == 0 (and then mae_twin == mean(ref) and
max_abs_twin == max_abs_GPU, both checked). Fields that can be negative are skipped (reported). Exact float equality: the
scorer accumulates both pairs with the same code over the same reference, so identical candidate arrays give identical sums.
usage: python3 twin_degenerate.py OUT_JSON
"""
import json, sys
from pathlib import Path

S = Path("<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score")
TWINS = {"wn3_20260227": "C_0227", "era5_20240302": "C_era5_20240302"}
NONNEG = {"QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "QNSNOW", "QNGRAUPEL", "CANLIQ", "CANICE", "CANWAT",
          "FWET", "QRAINXY", "QSNOWXY", "RAINNC", "SNOWNC", "GRAUPELNC", "SNOW", "SNOWH", "SNOWC", "ACSNOM", "ECAN", "EVC"}


def lead_of(fname, t0):
    from datetime import datetime
    t = datetime.strptime(fname[11:30], "%Y-%m-%d_%H:%M:%S")
    return round((t - t0).total_seconds() / 3600)


def main():
    from datetime import datetime
    out = {}
    for case, twin in TWINS.items():
        t0 = datetime.fromisoformat(json.load(open(S / case / "PRECONDITIONS.json"))["t0"])
        rows = []
        for dom in ("d01", "d02", "d03"):
            integ = json.load(open(S / case / "integ" / f"{dom}.json"))["domains"][dom]
            X = json.load(open(S / case / "x" / f"{dom}.json"))["field_summaries"]
            C = json.load(open(S / twin / "x" / f"{dom}.json"))["field_summaries"]
            for fld, hits in integ.get("degenerate_fields", {}).items():
                for h in hits:
                    L = lead_of(h["file"], t0)
                    row = {"domain": dom, "field": fld, "file": h["file"], "lead": L, "gpu_value": h["gpu_value"],
                           "cpu_range": h["cpu_range"]}
                    if fld not in NONNEG or h["gpu_value"] != 0.0 or h["cpu_range"][0] < 0:
                        row["twin_all_zero"] = "SKIP (field not provably >= 0 / GPU constant != 0)"
                        rows.append(row); continue
                    x = [r for r in X[fld]["by_lead"] if r["lead_h"] == L][0]
                    c = [r for r in C[fld]["by_lead"] if r["lead_h"] == L][0]
                    assert x["finite_pair"] == x["n"] and c["finite_pair"] == c["n"], (case, dom, fld, L)
                    assert x["mae"] == -x["bias"], (case, dom, fld, L, "GPU not <= ref everywhere")
                    mref = x["mae"]
                    tmean = c["bias"] + mref
                    row.update({"mean_ref": mref, "mean_twin": tmean,
                                "twin_all_zero": bool(tmean == 0.0 and c["mae"] == mref and c["max_abs"] == x["max_abs"])})
                    rows.append(row)
        out[case] = {"twin": twin, "t0": t0.isoformat(), "gpu_degenerate_frames": len(rows),
                     "twin_also_all_zero": sum(r["twin_all_zero"] is True for r in rows), "rows": rows}
        print(case, twin, "GPU-degenerate frames", len(rows), "-> twin also all-zero", out[case]["twin_also_all_zero"])
        for r in rows:
            print(f"   {r['domain']} {r['field']:8s} lead {r['lead']:2d} CPU max {r['cpu_range'][1]:.2e} twin_all_zero={r['twin_all_zero']}")
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
