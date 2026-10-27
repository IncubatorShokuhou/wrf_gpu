"""#148 integrity rows on the nest spec zone: is a leaf's ring-0 hydrometeor trace explained by its parent (d02) at the same time?

At an hourly frame the leaf ring 0 equals SINT(parent coupled value)/(c1h*muts+c2h) (WRF spec_bdy_final; GPU release default
GPUWRF_NEST_SCALAR_SPEC_FINAL=1, microphysics skips the spec zone, no positivity floor), so for each run separately:
  child ring-0 max > 0  ==>  parent max over the SINT support of those cells > 0          (necessary)
  parent support max > R  ==>  child ring-0 max > 0                                       (expected for R well above round-off)
Both runs are checked with the same code; CPU-WRF is the control of the method. Column maxima (npz *_colmax) are used, i.e.
level information is dropped: a necessary-condition test, not a reconstruction of the values.

usage: JAX_PLATFORMS=cpu python proofs/o5-9nest/ring0_trace.py <npz_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

# namelist a2 (#148): i/j_parent_start of the 1 km leaves in d02, ratio 3
GEOM = {"d03": (92, 36), "d04": (58, 54), "d05": (79, 37), "d06": (52, 23), "d07": (129, 26), "d08": (173, 38), "d09": (192, 69)}
RATIO = 3
SUPPORT = 2  # parent cells either side of the registered position (SINT 5-point stencil half-width)
FIELDS = ("QRAIN", "QNRAIN", "QGRAUP", "QSNOW", "QICE", "QCLOUD")
R_EXPECT = {"QRAIN": 1e-10, "QNRAIN": 1e-3, "QGRAUP": 1e-10, "QSNOW": 1e-10, "QICE": 1e-10, "QCLOUD": 1e-10}


def ring0_cells(ny, nx):
    y, x = np.mgrid[0:ny, 0:nx]
    return np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x)) == 0


def parent_support(ny, nx, ips, jps, pny, pnx):
    """boolean parent mask covering the SINT support of the child's ring-0 mass cells (WRF registration i -> ips-1 + i/3 - 1/3)."""
    m = np.zeros((pny, pnx), bool)
    r0 = ring0_cells(ny, nx)
    for j, i in zip(*np.nonzero(r0)):
        pi = ips - 1 + i / RATIO - 1.0 / RATIO
        pj = jps - 1 + j / RATIO - 1.0 / RATIO
        i0, i1 = int(np.floor(pi)) - SUPPORT, int(np.ceil(pi)) + SUPPORT
        j0, j1 = int(np.floor(pj)) - SUPPORT, int(np.ceil(pj)) + SUPPORT
        m[max(j0, 0):min(j1, pny - 1) + 1, max(i0, 0):min(i1, pnx - 1) + 1] = True
    return m


def main(npz_dir, out_path):
    npz_dir = Path(npz_dir)
    par = dict(np.load(npz_dir / "d02.npz"))
    pny, pnx = par["HGT"].shape
    plead = par["leads"].tolist()
    out = {"method": __doc__.split("\n\n")[0], "support_half_width_parent_cells": SUPPORT, "r_expect": R_EXPECT, "leaves": {}}
    for dom, (ips, jps) in GEOM.items():
        p = npz_dir / f"{dom}.npz"
        if not p.exists():
            continue
        ch = dict(np.load(p))
        ny, nx = ch["HGT"].shape
        r0 = ring0_cells(ny, nx)
        sup = parent_support(ny, nx, ips, jps, pny, pnx)
        leads = [L for L in ch["leads"].tolist() if L in plead]
        res = {}
        for f in FIELDS:
            key = f"{f}_colmax"
            if f"cpu_{key}" not in ch or f"cpu_{key}" not in par:
                continue
            rows, viol = [], {"cpu": {"child_without_parent": [], "parent_without_child": []},
                              "gpu": {"child_without_parent": [], "parent_without_child": []}}
            for L in leads:
                ic, ip = ch["leads"].tolist().index(L), plead.index(L)
                row = {"lead": L}
                for side in ("cpu", "gpu"):
                    c = ch[f"{side}_{key}"][ic][r0]
                    pv = par[f"{side}_{key}"][ip][sup]
                    row[side] = {"child_ring0_nonzero": int((c > 0).sum()), "child_ring0_max": float(c.max()),
                                 "parent_support_nonzero": int((pv > 0).sum()), "parent_support_max": float(pv.max())}
                    if not np.isfinite(c).all() or not np.isfinite(pv).all():
                        raise ValueError(f"non-finite {side} {dom} {f} lead {L}")
                    if c.max() > 0 and pv.max() <= 0:
                        viol[side]["child_without_parent"].append(L)
                    if pv.max() > R_EXPECT[f] and c.max() <= 0:
                        viol[side]["parent_without_child"].append(L)
                rows.append(row)
            res[f] = {"violations": viol, "by_lead": rows}
        out["leaves"][dom] = {"shape": [ny, nx], "parent_support_cells": int(sup.sum()), "fields": res}
    Path(out_path).write_text(json.dumps(out, indent=1))
    for dom, d in out["leaves"].items():
        for f, r in d["fields"].items():
            v = r["violations"]
            print(dom, f, "cpu", {k: len(x) for k, x in v["cpu"].items()}, "gpu", {k: len(x) for k, x in v["gpu"].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
