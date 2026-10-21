"""Compare two runs (max_dom 3 vs 9) frame by frame for the shared domains d01-d03.

usage: python cmp_maxdom.py OUT.json DIR_A DIR_B   (each dir holds wrfout_d0N_<time>; searched recursively)
Per domain and valid time: number of variables compared, number bitwise-equal (array bytes), list of differing
variables, max |diff| and RMSE of U10/V10/T2/U/V/W/T/PH/MU/QVAPOR. Missing frames are reported, never skipped silently."""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

KEY = ("U10", "V10", "T2", "U", "V", "W", "T", "PH", "MU", "QVAPOR", "PSFC", "UST", "HFX")


def index(d):
    return {p.name: p for p in Path(d).rglob("wrfout_d0*")}


def main():
    out, a, b = sys.argv[1], index(sys.argv[2]), index(sys.argv[3])
    res = {"a": sys.argv[2], "b": sys.argv[3], "frames": [], "missing": []}
    for name in sorted(set(a) | set(b)):
        if name[7:10] not in ("d01", "d02", "d03"):
            continue
        if name not in a or name not in b:
            res["missing"].append([name, name in a, name in b]); continue
        with nc.Dataset(a[name]) as fa, nc.Dataset(b[name]) as fb:
            fa.set_auto_mask(False); fb.set_auto_mask(False)
            common = sorted(set(fa.variables) & set(fb.variables))
            diff, stats = [], {}
            for v in common:
                x, y = fa[v][:], fb[v][:]
                if x.shape != y.shape or x.tobytes() != y.tobytes():
                    diff.append(v)
                    if v in KEY and x.shape == y.shape:
                        d = y.astype(np.float64) - x.astype(np.float64)
                        stats[v] = [float(np.abs(d).max()), float(np.sqrt((d ** 2).mean()))]
            res["frames"].append({"file": name, "n_vars": len(common), "n_equal": len(common) - len(diff),
                                  "only_a": sorted(set(fa.variables) - set(fb.variables)), "only_b": sorted(set(fb.variables) - set(fa.variables)),
                                  "diff_vars": diff, "key_maxabs_rmse": stats})
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    for f in res["frames"]:
        print(f["file"], f"equal {f['n_equal']}/{f['n_vars']}", {k: round(v[0], 6) for k, v in f["key_maxabs_rmse"].items() if k in ("U10", "T2", "U", "MU")})
    print("missing", res["missing"])


if __name__ == "__main__":
    main()
