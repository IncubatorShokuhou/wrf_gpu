"""Read-only original CPU-WRF export/edited-IC dry-run check (no forecast).

Run in an immutable checkout: python real_check.py CPU_CASE ARTIFACT_DIR.
Copies only wrfinput; source case and WRF tables are never edited.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import netCDF4
import numpy as np
import xarray as xr

case, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
source = case / "wrfinput_d01"
sha = hashlib.sha256(source.read_bytes()).hexdigest()
receipts = {"source": str(source), "source_sha256": sha, "forecast_scope": "dry-run only; edited geo_em/real.exe is a preprocessing responsibility", "compatibility": []}
working = out / "edited_case"
working.mkdir()
try:
    for path in case.iterdir():
        if path.is_file() and path.name not in {"namelist.input", "wrfinput_d01"} and not path.name.startswith("wrfout"):
            (working/path.name).symlink_to(path.resolve())
    shutil.copyfile(source, working/"wrfinput_d01")
    text = (case/"namelist.input").read_text()
    with netCDF4.Dataset(source) as ds:
        old_hgt = np.asarray(ds["HGT"][:])
        land = np.asarray(ds["LANDMASK"][:]).astype(bool)
        original_lu = np.asarray(ds["LU_INDEX"][:])
        mminlu = ds.MMINLU
    for count in (21,24,40,61):
        (working/"namelist.input").write_text(re.sub(r"(num_land_cat\s*=\s*)\d+",rf"\g<1>{count}",text))
        with netCDF4.Dataset(working/"wrfinput_d01","a") as ds:
            ds.NUM_LAND_CAT=count
            ds.MMINLU=mminlu
            ds["HGT"][:]=old_hgt+np.where(land,17.,0.)
            for key in ("LU_INDEX","IVGTYP"):
                ds[key][:]=np.where(land,5,original_lu)
        start=time.perf_counter()
        run=subprocess.run([sys.executable,"-m","gpuwrf.cli","run","--input-dir",str(working),"--output-dir",str(out/"never_run"),"--hours","1","--dry-run"],
                           cwd=root,env={**os.environ,"PYTHONPATH":str(root/"src")},capture_output=True,text=True,timeout=30)
        if run.returncode:
            raise AssertionError(run.stderr)
        plan=json.loads(run.stdout)
        assert len(plan["land_category_check"]["checked"])==1
        receipts["compatibility"].append({"NUM_LAND_CAT":count,"status":"PASS","wall_s":time.perf_counter()-start})
    with netCDF4.Dataset(working/"wrfinput_d01","a") as ds:
        ds.MMINLU="MODIFIED_IGBP_MODIS_NOAH_LCZ"
        for key in ("LU_INDEX","IVGTYP"):
            ds[key][:]=np.where(land,31,original_lu)
    run=subprocess.run([sys.executable,"-m","gpuwrf.cli","run","--input-dir",str(working),"--output-dir",str(out/"never_run"),"--hours","1","--dry-run"],
                       cwd=root,env={**os.environ,"PYTHONPATH":str(root/"src")},capture_output=True,text=True,timeout=30)
    assert run.returncode==2 and "LCZ_WITHOUT_URBAN" in run.stderr,run.stderr
    receipts["lcz_refusal"]="PASS"
    from gpuwrf.export.core import load
    from gpuwrf.export.writers import write
    history=sorted(case.glob("wrfout_d01_*"))[0]
    start=time.perf_counter()
    exported=load([history],["T2","PSFC","HGT"])
    for fmt,suffix in (("netcdf","nc"),("zarr","zarr")):
        target=out/f"cpu_export.{suffix}"
        write(exported,target,fmt)
        with (xr.open_dataset(target) if fmt=="netcdf" else xr.open_zarr(target)) as ds, xr.open_dataset(history) as cpu:
            for name,key in (("temperature_2m","T2"),("surface_pressure","PSFC"),("terrain_height","HGT")):
                np.testing.assert_array_equal(ds[name].values,cpu[key].values)
    receipts["export"]={"source":str(history),"source_sha256":hashlib.sha256(history.read_bytes()).hexdigest(),
                       "status":"PASS","formats":["netcdf","zarr"],"fields_exact":["T2","PSFC","HGT"],
                       "grid_shape":dict(exported.sizes),"wall_s":time.perf_counter()-start}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    (out/"real_check.json").write_text(json.dumps(receipts,indent=2))
    print(json.dumps(receipts))
finally:
    shutil.rmtree(working)
    for path in (out/"cpu_export.nc",out/"cpu_export.zarr"):
        if path.is_dir(): shutil.rmtree(path)
        elif path.exists(): path.unlink()
