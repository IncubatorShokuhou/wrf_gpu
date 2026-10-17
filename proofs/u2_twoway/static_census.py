"""Measure original WRF static feedback at initialization and the first step."""
from pathlib import Path
import json
import sys
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

case=Path(sys.argv[1])
out=Path(sys.argv[2])
names=("HGT","PB","PHB","MUB","MAPFAC_MX","MAPFAC_MY","MAPFAC_UX","MAPFAC_UY",
       "MAPFAC_VX","MAPFAC_VY","F","E","SINALPHA","COSALPHA","LU_INDEX","XLAND")
files={"input":case/"wrfinput_d01","t0":case/"wrfout_d01_2026-02-28_00:00:00",
       "t54":case/"wrfout_d01_2026-02-28_00:00:54","t108":case/"wrfout_d01_2026-02-28_00:01:48"}
values={}
for key,file in files.items():
    with Dataset(file) as ds:
        values[key]={name:np.asarray(ds[name][0]) for name in names if name in ds.variables}
result={"reference":"original CPU-WRF feedback1/smooth2; no port/self reference","fields":{}}
for name in values["t0"]:
    row={}
    for left,right in (("input","t0"),("t0","t54"),("t54","t108")):
        if name not in values[left]:continue
        a,b=values[left][name],values[right][name]
        assert np.isfinite(a).all() and np.isfinite(b).all()
        diff=b.astype(np.float64)-a.astype(np.float64)
        row[f"{left}_to_{right}"]={"changed":int(np.count_nonzero(diff)),"max_abs":float(np.abs(diff).max()),
                                    "rmse":float(np.sqrt(np.mean(diff*diff)))}
    result["fields"][name]=row
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
