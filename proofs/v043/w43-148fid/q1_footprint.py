"""Parent (d02) error over a leaf footprint vs the leaf's own error, per lead (U10, W). CPU numpy only."""
import json, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
EDGES = {"d03": (92, 36), "d04": (58, 54), "d05": (79, 37), "d06": (52, 23), "d07": (129, 26), "d08": (173, 38), "d09": (192, 69)}
z2 = np.load(H / "data/d02.npz")
P = {k: z2[k] for k in ("gpu_U10", "cpu_U10", "gpu_W", "cpu_W", "leads")}
res = {}
for dom, (i0, j0) in EDGES.items():
    zz = np.load(H / f"data/{dom}.npz")
    C = {k: zz[k] for k in ("gpu_U10", "cpu_U10", "gpu_W", "cpu_W", "HGT")}
    ny, nx = C["HGT"].shape
    sj, si = slice(j0 - 1, j0 - 1 + ny // 3 + 1), slice(i0 - 1, i0 - 1 + nx // 3 + 1)
    r = lambda a: float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    rows = []
    for i, L in enumerate(P["leads"]):
        rows.append({"lead": int(L), "parent_U10": r(P["gpu_U10"][i][sj, si] - P["cpu_U10"][i][sj, si]),
                     "child_U10": r(C["gpu_U10"][i] - C["cpu_U10"][i]),
                     "parent_W": r(P["gpu_W"][i][:, sj, si] - P["cpu_W"][i][:, sj, si]),
                     "child_W": r(C["gpu_W"][i] - C["cpu_W"][i])})
    res[dom] = rows
json.dump(res, open(H / "q1_parent_footprint.json", "w"), indent=0)
for dom in ("d04", "d05", "d06"):
    print(dom, "lead parentU10 childU10 parentW childW")
    for x in res[dom]:
        if x["lead"] in (4, 6, 8, 9, 10, 12, 16, 18, 20, 22, 24, 30, 34, 54, 55, 59, 63, 66, 72):
            print("  ", x["lead"], *(round(x[k], 3) for k in ("parent_U10", "child_U10", "parent_W", "child_W")))
