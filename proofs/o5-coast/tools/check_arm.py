"""Checker phase of coast_gpu.sh (CPU, seconds): validates one arm's streamed frames.

usage: python check_arm.py ARM_DIR EXPECTED_FRAMES [CPU_REF_DIR]
PASS needs: stream_extract MANIFEST with EXPECTED_FRAMES records, every frame npz present, no nonfinite field, every
record deleted (raw history not left behind).  With CPU_REF_DIR (our own CPU-WRF run of the same case) it also reports
the d01 ring-mean MU difference at the first common forecast frame per boundary ring (released clock: rings 1-9 sit
about one root step of boundary tendency below CPU-WRF; WRF walk: ~0) -- reported, not a pass criterion.
Writes ARM_DIR/CHECK.json; exit 0 = PASS, 1 = FAIL."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

arm, expected = Path(sys.argv[1]), int(sys.argv[2])
ref = Path(sys.argv[3]) if len(sys.argv) > 3 else None
man = arm / "frames" / "MANIFEST.jsonl"
recs = [json.loads(x) for x in man.read_text().splitlines() if x.strip()] if man.exists() else []
res = {"arm": str(arm), "expected_frames": expected, "frames": len(recs),
       "missing_npz": [r["name"] for r in recs if not (arm / "frames" / f"{r['name']}.npz").exists()],
       "nonfinite": [r["name"] for r in recs if r["nonfinite_fields"]],
       "not_deleted": [r["name"] for r in recs if not r["deleted"]],
       "raw_left": sorted(os.path.basename(p) for p in glob.glob(str(arm / "wrfout" / "wrfout_d0*")))}
res["pass"] = (len(recs) == expected and not res["missing_npz"] and not res["nonfinite"] and not res["not_deleted"]
               and not res["raw_left"])
if ref is not None and res["pass"]:
    import netCDF4 as nc
    names = sorted(r["name"] for r in recs if r["name"].startswith("wrfout_d01_"))
    for name in names[1:2]:  # first forecast frame
        g = np.load(arm / "frames" / f"{name}.npz")["MU"].astype(np.float64)
        rp = ref / name
        if rp.exists():
            with nc.Dataset(rp) as f:
                c = np.asarray(f["MU"][0], np.float64)
            ny, nx = c.shape
            jj, ii = np.mgrid[0:ny, 0:nx]
            dist = np.minimum(np.minimum(jj, ny - 1 - jj), np.minimum(ii, nx - 1 - ii))
            res["d01_mu_ring_mean_diff_pa"] = {"frame": name, **{f"ring{k}": float((g - c)[dist == k].mean())
                                                                  for k in range(6)},
                                               "interior": float((g - c)[dist >= 10].mean())}
(arm / "CHECK.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("frames", "pass")} | ({"mu": res.get("d01_mu_ring_mean_diff_pa")} if ref else {})))
sys.exit(0 if res["pass"] else 1)
