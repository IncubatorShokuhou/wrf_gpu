"""Seal the original CPU-WRF output names, values, units and file checksums."""
import hashlib
import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

RUN=Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h")
ROOT=Path(__file__).resolve().parents[2]
FIELDS=("PREC_ACC_C","PREC_ACC_NC","SNOW_ACC_NC","WSPD10MAX","W_UP_MAX","W_DN_MAX",
        "UP_HELI_MAX","W_MEAN","GRPL_MAX","HAIL_MAXK1","HAIL_MAX2D","REFD_MAX","REFL_10CM","SWDDNI","SWDDIF")
manifest=dict(receipt=json.loads((RUN/"receipt.json").read_text()),fields=FIELDS,frames=[],
              qualification="Original CPU-WRF reference ONLY; no GPU comparison; solar inactive at night")
for path in sorted(RUN.glob("wrfout*")):
    rec=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fields={})
    with Dataset(path) as ds:
        for name in FIELDS:
            v=ds[name];a=np.asarray(v[:]);assert np.isfinite(a).all(),(path,name)
            rec["fields"][name]=dict(shape=list(a.shape),dtype=str(a.dtype),units=getattr(v,"units",""),
                                    min=float(a.min()),max=float(a.max()),nonzero=int(np.count_nonzero(a)))
    manifest["frames"].append(rec)
assert len(manifest["frames"]) == 4 and manifest["receipt"]["success"]
(ROOT/"proofs/u2_diag/cpu_reference_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(success=True,frames=len(manifest["frames"]),fields=FIELDS,
                      wall_s=manifest["receipt"]["wall_s"])))
