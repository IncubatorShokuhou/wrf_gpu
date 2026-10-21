"""Q3 mask-radius sensitivity: outside-mask RMSE of W/U10/V10 vs dilation radius (0,1,2,3,5 km), 1 km leaves."""
import json, numpy as np
from pathlib import Path
from scipy import ndimage
H = Path(__file__).resolve().parent
st = ndimage.generate_binary_structure(2, 1)
out = {}
for dom in ("d04", "d05", "d06"):
    z = np.load(H / f"data/{dom}.npz")
    A = {k: z[k].astype(np.float64) for k in ("cpu_RAINNC", "gpu_RAINNC", "cpu_W", "gpu_W", "cpu_U10", "gpu_U10", "cpu_V10", "gpu_V10")}
    leads = z["leads"].tolist()
    ic = np.diff(A["cpu_RAINNC"], axis=0, prepend=A["cpu_RAINNC"][:1]); ig = np.diff(A["gpu_RAINNC"], axis=0, prepend=A["gpu_RAINNC"][:1])
    dW = A["gpu_W"] - A["cpu_W"]; dU = A["gpu_U10"] - A["cpu_U10"]; dV = A["gpu_V10"] - A["cpu_V10"]
    res = {}
    for rad in (0, 1, 2, 3, 5):
        rows = []
        for i, L in enumerate(leads):
            heavy = (ic[i] >= 1) | (ig[i] >= 1)
            if i: heavy |= (ic[i - 1] >= 1) | (ig[i - 1] >= 1)
            near = ndimage.binary_dilation(heavy, st, iterations=rad) if (rad and heavy.any()) else heavy
            o = ~near
            wall = float(np.sqrt(np.mean(dW[i] ** 2)))
            wo = float(np.sqrt(np.mean(dW[i][:, o] ** 2))) if o.any() else None
            uo = float(np.sqrt(np.mean(dU[i][o] ** 2))) if o.any() else None
            vo = float(np.sqrt(np.mean(dV[i][o] ** 2))) if o.any() else None
            sse = (dW[i] ** 2).sum(axis=0)
            rows.append({"lead": L, "W_all": wall, "W_out": wo, "U10_out": uo, "V10_out": vo, "area_frac": float(near.mean()),
                         "W_sse_frac": float(sse[near].sum() / sse.sum()) if sse.sum() > 0 else None})
        q = [r for r in rows if r["W_all"] > 0.1]
        res[rad] = {"max_W_out": max((r["W_out"] or 0) for r in rows),
                    "median_W_sse_frac_qualifying": float(np.median([r["W_sse_frac"] for r in q])) if q else None,
                    "median_area_frac_qualifying": float(np.median([r["area_frac"] for r in q])) if q else None,
                    "max_U10_out": max((r["U10_out"] or 0) for r in rows), "max_V10_out": max((r["V10_out"] or 0) for r in rows),
                    "rows": rows}
    out[dom] = res
    print(dom, " ".join(f"r{r}km: maxWout {v['max_W_out']:.4f} medSSE {v['median_W_sse_frac_qualifying']:.3f}/area {v['median_area_frac_qualifying']:.3f} maxU10out {v['max_U10_out']:.3f}" for r, v in res.items()))
json.dump(out, open(H / "Q3_radius.json", "w"), indent=0)
