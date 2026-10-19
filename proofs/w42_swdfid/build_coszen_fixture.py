"""w42-swdfid: build the CPU-WRF history-COSZEN oracle fixture (original WRF 4.7.1, #146 WN3 0420 C arm, 72 h).

usage: python build_coszen_fixture.py C_RUN_DIR OUT_NPZ OUT_RECEIPT_JSON
Per domain d01/d02/d03, every 4th mass point in x and y: XLAT/XLONG (frame 0) and the history COSZEN + XTIME of all 73
hourly frames, plus DT/RADT/JULDAY/GMT/START_DATE.  WRF history COSZEN is the radiation driver's HELD value (the last
radiation call's calc_coszen at xtime + radt/2).  The receipt pins every source file by sha256.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

src, out_npz, out_receipt = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
arrays, receipt = {}, {"source_dir": str(src), "files": {}, "stride": 4}
for dom in ("d01", "d02", "d03"):
    names = sorted(p.name for p in src.glob(f"wrfout_{dom}_*"))
    assert len(names) == 73, (dom, len(names))
    with Dataset(src / names[0]) as ds:
        lat = np.asarray(ds["XLAT"][0], np.float32)[::4, ::4]
        lon = np.asarray(ds["XLONG"][0], np.float32)[::4, ::4]
        attrs = {k: getattr(ds, k) for k in ("DT", "RADT", "JULDAY", "GMT", "START_DATE", "SIMULATION_START_DATE")}
    cz, xt = [], []
    for n in names:
        p = src / n
        receipt["files"][n] = hashlib.sha256(p.read_bytes()).hexdigest()
        with Dataset(p) as ds:
            cz.append(np.asarray(ds["COSZEN"][0], np.float32)[::4, ::4])
            xt.append(float(ds["XTIME"][0]))
    arrays[f"{dom}_xlat"], arrays[f"{dom}_xlong"] = lat, lon
    arrays[f"{dom}_coszen"], arrays[f"{dom}_xtime_min"] = np.stack(cz), np.asarray(xt, np.float64)
    arrays[f"{dom}_dt_s"], arrays[f"{dom}_radt_min"] = np.float64(attrs["DT"]), np.float64(attrs["RADT"])
    receipt[dom] = {k: (v.item() if hasattr(v, "item") else v) for k, v in attrs.items()}
    receipt[dom]["frames"] = [names[0], names[-1]]
np.savez(out_npz, **arrays)
receipt["npz_sha256"] = hashlib.sha256(out_npz.read_bytes()).hexdigest()
out_receipt.write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps({d: receipt[d] for d in ("d01", "d02", "d03")}), receipt["npz_sha256"])
