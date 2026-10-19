"""Old-vs-new comparator identity on synthetic EDGE cases the real 24/72 h data never exercise (w42-scorer, CPU only):
unpaired NaN cells (both sides, varying per lead), staggered U/V splits, a field missing in one GPU frame, a field whose
shape changes in one lead (incompatible lead), dimension mismatch from the start (incompatible common), a char field
(non-numeric), CPU-only and GPU-only fields, a Times mismatch, an int field, a static field. The new comparator runs with
the default budget and with --field-batch-mib 1 (one field per pass here) and without spatial splits.
usage: python edge_identity.py REF_TREE NEW_TREE WORK_DIR OUT_JSON"""
import json
import re
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np

ref_tree, new_tree, work, out = (Path(a) for a in sys.argv[1:5])
cpu_dir, gpu_dir = work / "cpu", work / "gpu"
cpu_dir.mkdir(parents=True, exist_ok=True)
gpu_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
nz, ny, nx = 12, 40, 50
times = [f"2026-02-28_{h:02d}:00:00" for h in range(4)]


def write(path, t, side, lead):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("bottom_top", nz), ("south_north", ny),
                           ("west_east", nx), ("west_east_stag", nx + 1), ("south_north_stag", ny + 1), ("chars", 4)):
            ds.createDimension(name, size)
        tt = t if not (side == "gpu" and lead == 2) else t.replace("02:00", "02:01")  # Times mismatch in one lead
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = [c.encode() for c in tt]
        yy, xx = np.meshgrid(np.arange(ny, dtype="f4"), np.arange(nx, dtype="f4"), indexing="ij")
        statics = {"HGT": yy * 60.0 + xx * 20.0 - 100.0, "LANDMASK": ((xx + yy) % 7 > 2).astype("f4"),
                   "XLAT": 28.0 + yy * 0.01, "XLONG": -17.0 + xx * 0.01}
        for name, values in statics.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = values
        base = np.random.default_rng(100 + lead)
        def fld(shape, scale):
            a = base.standard_normal(shape).astype("f4") * scale
            return a if side == "cpu" else a + rng.standard_normal(shape).astype("f4") * 0.05 * scale
        t3 = fld((nz, ny, nx), 2.0) + 280.0
        t3[rng.uniform(size=t3.shape) < (0.02 if side == "cpu" else 0.03)] = np.nan
        ds.createVariable("T", "f4", ("Time", "bottom_top", "south_north", "west_east"))[0] = t3
        ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))[0] = fld((nz, ny, nx + 1), 5.0)
        ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))[0] = fld((nz, ny + 1, nx), 5.0)
        ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))[0] = fld((ny, nx), 1.0) + 290.0
        ds.createVariable("XTIME", "f4", ("Time",))[0] = 60.0 * lead
        ds.createVariable("LU_INDEX", "i4", ("Time", "south_north", "west_east"))[0] = ((xx + yy) % 21).astype("i4")
        if not (side == "gpu" and lead == 2):  # missing in one GPU frame
            ds.createVariable("QX", "f4", ("Time", "south_north", "west_east"))[0] = fld((ny, nx), 1e-3)
        if side == "gpu" and lead == 1:  # shape changes in one lead
            ds.createVariable("SHP", "f4", ("Time", "south_north_stag", "west_east"))[0] = fld((ny + 1, nx), 1.0)
        else:
            ds.createVariable("SHP", "f4", ("Time", "south_north", "west_east"))[0] = fld((ny, nx), 1.0)
        if side == "cpu":  # dims differ from the start
            ds.createVariable("IV", "f4", ("Time", "south_north", "west_east"))[0] = fld((ny, nx), 1.0)
            ds.createVariable("CPUONLY", "f4", ("Time", "south_north", "west_east"))[0] = fld((ny, nx), 1.0)
        else:
            ds.createVariable("IV", "f4", ("Time", "south_north", "west_east_stag"))[0] = fld((ny, nx + 1), 1.0)
            ds.createVariable("GPUONLY", "f4", ("Time", "south_north", "west_east"))[0] = fld((ny, nx), 1.0)
        ds.createVariable("CHARV", "S1", ("Time", "chars"))[0, :] = [b"a", b"b", b"c", b"d"]


for lead, t in enumerate(times):
    for side, d in (("cpu", cpu_dir), ("gpu", gpu_dir)):
        write(d / f"wrfout_d01_{t}", t, side, lead)
manifest = ref_tree / "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"
VOL = [(re.compile(rb'"generated_utc": "[^"]*"'), b'"g"'), (re.compile(rb'"elapsed_seconds": [-0-9.eE+]+'), b'"e"'),
       (re.compile(rb"Generated UTC: `[^`]*`"), b"G")]


def run(tree, tag, extra):
    o = work / tag
    cmd = [sys.executable, str(tree / "scripts/wn3_fast_compare.py"), "--cpu-dir", str(cpu_dir), "--gpu-dir", str(gpu_dir),
           "--domain", "d01", "--init", "2026-02-28T00:00:00Z", "--min-lead", "0", "--max-lead", "3",
           "--tolerance-json", str(manifest), "--boundary-width", "3", "--progress", "1",
           "--out-json", str(o.with_suffix(".json")), "--out-md", str(o.with_suffix(".md")), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    data = []
    for suffix in (".json", ".md"):
        b = o.with_suffix(suffix).read_bytes()
        for pat, rep in VOL:
            b = pat.sub(rep, b)
        data.append(b)
    return data, p.stderr


results = {}
for splits in ([], ["--no-spatial-splits"]):
    ref, _ = run(ref_tree, "ref" + "".join(splits), splits)
    for budget in ([], ["--field-batch-mib", "1"]):
        tag = "new" + "".join(splits) + "".join(budget)
        new, err = run(new_tree, tag, splits + budget)
        passes = re.search(r"in (\d+) passes", err)
        results[tag] = {"identical_json": ref[0] == new[0], "identical_md": ref[1] == new[1],
                        "passes": int(passes.group(1)) if passes else None}
report = json.loads((work / "ref.json").read_text())
fs = report["field_summaries"]
coverage = {"missing_leads_QX": fs["QX"]["missing_leads"], "incompatible_leads_SHP": [x["lead_h"] for x in fs["SHP"]["incompatible_leads"]],
            "non_numeric": [x["field"] for x in report["inventory"]["non_numeric_common"]],
            "incompatible_common": [x["field"] for x in report["inventory"]["incompatible_common"]],
            "cpu_only": report["inventory"]["cpu_only"], "gpu_only": report["inventory"]["gpu_only"],
            "times_all_equal": fs["Times"]["all_equal"], "T_finite_pair_fraction": fs["T"]["overall"]["finite_pair_fraction"],
            "split_kinds_U_V_T": [bool(fs[n]["spatial_splits"]) for n in ("U", "V", "T")]}
ok = all(r["identical_json"] and r["identical_md"] for r in results.values())
out.write_text(json.dumps({"identical": ok, "runs": results, "coverage": coverage}, indent=1) + "\n")
print(json.dumps({"identical": ok, "runs": results, "coverage": coverage}))
sys.exit(0 if ok else 1)
