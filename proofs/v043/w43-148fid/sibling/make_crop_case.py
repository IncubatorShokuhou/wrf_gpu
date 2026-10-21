"""Real-structured small sibling case from OUR PROD inputs (no ALISIOS data): d01 = PROD d01 (unchanged, with its
wrfbdy), d02..d04 = three different crops of the PROD d02 wrfinput (real terrain, land use, initial state), each a
one-way child of d01 (parent_id 1, ratio 3). Crop starts are multiples of 3 so every crop sits on the parent grid
exactly as a WRF nest would (i_parent_start = 16 + x0/3).

usage: python make_crop_case.py OUTDIR
"""
import os
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

SRC = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
# (x0, y0, nx, ny) in PROD d02 mass cells, 0-based; nx, ny, x0, y0 all multiples of 3
CROPS = ((30, 30, 60, 60), (105, 36, 54, 48), (180, 27, 57, 63))


def crop_wrfinput(src, dst, x0, y0, nx, ny, grid_id):
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w", format=a.data_model) as b:
        for name, dim in a.dimensions.items():
            size = {"west_east": nx, "west_east_stag": nx + 1, "south_north": ny, "south_north_stag": ny + 1}.get(name, len(dim))
            b.createDimension(name, None if dim.isunlimited() else size)
        for name, var in a.variables.items():
            out = b.createVariable(name, var.datatype, var.dimensions)
            out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            sl = []
            for d in var.dimensions:
                if d == "west_east":
                    sl.append(slice(x0, x0 + nx))
                elif d == "west_east_stag":
                    sl.append(slice(x0, x0 + nx + 1))
                elif d == "south_north":
                    sl.append(slice(y0, y0 + ny))
                elif d == "south_north_stag":
                    sl.append(slice(y0, y0 + ny + 1))
                else:
                    sl.append(slice(None))
            out[:] = var[tuple(sl)]
        att = {k: a.getncattr(k) for k in a.ncattrs()}
        att.update({"WEST-EAST_GRID_DIMENSION": np.int32(nx + 1), "SOUTH-NORTH_GRID_DIMENSION": np.int32(ny + 1),
                    "WEST-EAST_PATCH_END_UNSTAG": np.int32(nx), "WEST-EAST_PATCH_END_STAG": np.int32(nx + 1),
                    "SOUTH-NORTH_PATCH_END_UNSTAG": np.int32(ny), "SOUTH-NORTH_PATCH_END_STAG": np.int32(ny + 1),
                    "GRID_ID": np.int32(grid_id), "PARENT_ID": np.int32(1),
                    "I_PARENT_START": np.int32(int(a.getncattr("I_PARENT_START")) + x0 // 3),
                    "J_PARENT_START": np.int32(int(a.getncattr("J_PARENT_START")) + y0 // 3)})
        lat = a.variables["XLAT"][0, y0:y0 + ny, x0:x0 + nx]
        lon = a.variables["XLONG"][0, y0:y0 + ny, x0:x0 + nx]
        att["CEN_LAT"] = np.float32(lat[ny // 2, nx // 2])
        att["CEN_LON"] = np.float32(lon[ny // 2, nx // 2])
        b.setncatts(att)


def namelist(src, dst, crops):
    n = 1 + len(crops)
    text = open(src).read()
    per = {"e_we": [121] + [c[2] + 1 for c in crops], "e_sn": [71] + [c[3] + 1 for c in crops],
           "i_parent_start": [1] + [16 + c[0] // 3 for c in crops], "j_parent_start": [1] + [16 + c[1] // 3 for c in crops],
           "parent_id": [1] * n, "grid_id": list(range(1, n + 1)), "max_dom": [n]}
    out = []
    for line in text.splitlines():
        m = re.match(r"^(\s*)(\w+)(\s*=\s*)(.*?)\s*$", line)
        if m:
            key, vals = m.group(2), m.group(4)
            if key in per:
                v = per[key]
                line = f"{m.group(1)}{key}{m.group(3)}" + ", ".join(str(x) for x in v) + ","
            else:
                items = [x.strip() for x in vals.rstrip(",").split(",")]
                if len(items) == 2 and vals.rstrip().endswith(","):   # per-domain list of the 2-domain PROD case
                    items = items + [items[1]] * (n - 2)
                    line = f"{m.group(1)}{key}{m.group(3)}" + ", ".join(items) + ","
        out.append(line)
    open(dst, "w").write("\n".join(out) + "\n")


def main(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    for f in ("wrfinput_d01", "wrfbdy_d01"):
        os.symlink(SRC / f, out / f)
    for k, (x0, y0, nx, ny) in enumerate(CROPS, start=2):
        crop_wrfinput(SRC / "wrfinput_d02", out / f"wrfinput_d0{k}", x0, y0, nx, ny, k)
    namelist(SRC / "namelist.input", out / "namelist.input", CROPS)
    print("case", out, "domains", 1 + len(CROPS))


if __name__ == "__main__":
    main(sys.argv[1])
