"""Write a real-structured ``wrffdda_d01`` (real.exe auxinput10 layout) from CPU-WRF history frames.

usage: python make_wrffdda.py <cpu_wrfout_dir> <domain> <start YYYY-mm-dd_HH:MM:SS> <interval_min> <n_records> <out>

Record n: ``*_NDG_OLD`` = frame(t_n), ``*_NDG_NEW`` = frame(t_n + interval) like main/real_em.F:904-1110
(T_NDG = dry th_phy - t0 from THM/QVAPOR in WRF REAL form; MU_NDG = MU; PH_NDG = PH).  Used as the
"synthetic-but-real-structured" analysis file for CPU smoke runs: no real.exe grid_fdda=1 output exists here.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

RVRD = np.float32(np.float32(461.6) / np.float32(287.0))


def frame(path: Path) -> dict:
    with Dataset(path) as d:
        g = lambda n: np.asarray(d[n][0], np.float32)  # noqa: E731
        qv = g("QVAPOR")
        th = (g("THM") + np.float32(300.0)) / (np.float32(1.0) + RVRD * qv)
        return {"U": g("U"), "V": g("V"), "T": th - np.float32(300.0), "Q": qv, "PH": g("PH"), "MU": g("MU")}


def wrf_metadata(ds, source: Path, provenance: str) -> None:
    """Supply WRF's ingestion metadata without changing analysis values.

    external/io_netcdf/wrf_io.F90 ext_ncd_read_field returns before reading a
    variable without FieldType. A successful wrf.exe run can otherwise nudge
    toward zero analyses. WRF_REAL=104 in wrf_io_flags.h.
    """

    with Dataset(source) as original:
        ds.setncatts({k: original.getncattr(k) for k in original.ncattrs() if k != "TITLE"})
        title = str(original.getncattr("TITLE"))
    ds.TITLE = title + "; replay FDDA analyses derived from original CPU-WRF history"
    ds.GPUWRF_ANALYSIS_PROVENANCE = provenance
    for name, variable in ds.variables.items():
        if name == "Times":
            continue
        variable.FieldType = np.int32(104)
        variable.MemoryOrder = "XYZ" if variable.ndim == 4 else "XY"
        variable.stagger = {"U": "X", "V": "Y", "PH": "Z"}.get(name.split("_NDG_")[0], "")


def main() -> None:
    src, dom, start, interval, nrec, out = sys.argv[1:7]
    t0 = datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
    interval, nrec = int(interval), int(nrec)
    times = [t0 + timedelta(minutes=interval * k) for k in range(nrec + 1)]
    frames = [frame(Path(src) / f"wrfout_{dom}_{t:%Y-%m-%d_%H:%M:%S}") for t in times]
    nz, ny, nx = frames[0]["T"].shape
    with Dataset(out, "w", format="NETCDF4") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", nx), ("south_north", ny),
                           ("bottom_top", nz), ("west_east_stag", nx + 1), ("south_north_stag", ny + 1),
                           ("bottom_top_stag", nz + 1), ("fdda_levels", 1)):
            ds.createDimension(name, size)
        tv = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for k in range(nrec):
            tv[k] = np.frombuffer(f"{times[k]:%Y-%m-%d_%H:%M:%S}".encode(), dtype="S1")
        dims = {"U": ("bottom_top", "south_north", "west_east_stag"), "V": ("bottom_top", "south_north_stag",
                "west_east"), "T": ("bottom_top", "south_north", "west_east"), "Q": ("bottom_top", "south_north",
                "west_east"), "PH": ("bottom_top_stag", "south_north", "west_east"),
                "MU": ("fdda_levels", "south_north", "west_east")}
        for key, dd in dims.items():
            for suffix, shift in (("OLD", 0), ("NEW", 1)):
                v = ds.createVariable(f"{key}_NDG_{suffix}", "f4", ("Time", *dd))
                for k in range(nrec):
                    # Registry fdda2d has i-v-j storage (v=fdda_levels=1),
                    # unlike history MU's ordinary i-j storage. Grid nudging's
                    # RMUNDGDTEN is zero, but WRF must still read a valid file.
                    value = frames[k + shift][key]
                    v[k] = value[None] if key == "MU" else value
        wrf_metadata(ds, Path(src) / f"wrfout_{dom}_{times[0]:%Y-%m-%d_%H:%M:%S}", str(Path(src).resolve()))


if __name__ == "__main__":
    main()
