"""Fortran-decoded OBS_DOMAIN specimen and pristine time/VIF helper fixtures."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def generate(build, save):
    manifest = json.loads((save / "MANIFEST.json").read_text())
    reports = [
        ("20230115000000",46.1234,7.4567,"STATION 001","Surface record", "FM-12 SYNOP", "TEST",456.,False,False,1,
         [[(101325.,0.),(100000.,0.),(458.,0.),(274.5,0.),(3.125,129.),(-2.5,129.),(82.5,0.),(96000.,0.),(-888888.,-888888.)]]),
        ("20230115003000",46.5678,7.1234,"WIND 002","SATWNDS record","SATWNDS","TEST",1000.,True,False,1,
         [[(85000.,0.),(1500.,0.),(-888888.,-888888.),(10.,0.),(12.,0.),(-888888.,-888888.)]]),
        ("20230115010000",46.9876,7.6789,"SOUND 003","Upper air profile","FM-35 TEMP","TEST",500.,True,True,2,
         [[(90000.,0.),(1000.,0.),(280.,-777777.),(4.,256.),(5.,256.),(60.,0.)],
          [(80000.,0.),(2000.,0.),(270.,0.),(8.,0.),(9.,0.),(50.,0.)]]),
    ]
    specimen = save / "OBS_DOMAIN101"
    with specimen.open("w", encoding="ascii") as f:
        for date,lat,lon,station,name,platform,source,elev,sounding,bogus,count,levels in reports:
            f.write(f" {date}\n  {lat:9.4f} {lon:9.4f} \n  {station:40s}   {name:40s}   \n")
            f.write(f"  {platform:16s}  {source:16s}  {elev:8.0f}  "
                    f"{'T' if sounding else 'F':>4}  {'T' if bogus else 'F':>4}  {count:5d}\n")
            for level in levels:
                f.write(" " + "".join(f"{value:11.3f} {qc:11.3f} " for value,qc in level) + "\n")
    for kind in ("r4","r8"):
        dtype = np.float32 if kind == "r4" else np.float64
        decoded = build / "decoded.txt"
        subprocess.run([str(build/kind/"decode"),str(specimen),str(decoded)],check=True,timeout=10)
        lines = iter(decoded.read_text().splitlines())
        records = []
        for _ in reports:
            date = next(lines).strip()
            latitude,longitude,elevation = map(float,next(lines).split())
            station_id,name,platform,source = [next(lines).rstrip() for _ in range(4)]
            sounding,bogus,count = next(lines).split()
            sounding,bogus,count = sounding == "T",bogus == "T",int(count)
            measurements = [np.asarray(list(map(float,next(lines).split()))).reshape(-1,2).tolist()
                            for _ in range(count if sounding else 1)]
            records.append(dict(time=datetime.strptime(date,"%Y%m%d%H%M%S").isoformat(),
                                latitude=latitude,longitude=longitude,elevation=elevation,
                                station_id=station_id,name=name,platform=platform,source=source,
                                is_sounding=sounding,bogus=bogus,measurement_count=count,measurements=measurements))
        (save/f"decoded_{kind}.json").write_text(json.dumps(records,indent=2)+"\n")
        decoded.unlink()
        # Nonbinary fractions, both directions of the taper, exact boundaries,
        # and PBL-biased VIF parameters before and after the hmax cap.
        n = 24
        times = np.asarray([-.2,0,.1,.25,.5,.75,.9,1.,1.1,1.5,2.,3.]*2,dtype)
        args = np.stack([np.full(n,1.3,dtype),np.full(n,.3,dtype),np.full(n,1.,dtype),
                         np.asarray([1.]*12+[.7]*12,dtype),times],axis=1)
        full = np.asarray([50.,-5000.,-4800.,6000.,10.,-5100.]*4,dtype)
        ramp = np.asarray([50.,-5000.,-5100.,100.,10.,-4900.]*4,dtype)
        pblh = np.asarray([20.,100.,1500.,5000.,75.,600.]*4,dtype)
        inp,out = build/"helpers_in.bin",build/"helpers_out.bin"
        with inp.open("wb") as f:
            f.write(np.asarray([n],np.int32).tobytes())
            for arr in (args,full,ramp,pblh,np.asarray([3000.,50.,50.],dtype)):
                f.write(arr.tobytes())
        subprocess.run([str(build/kind/"helpers"),str(inp),str(out)],check=True,timeout=10)
        result = np.fromfile(out,dtype).reshape(3,n)
        np.savez_compressed(save/f"helpers_{kind}.npz",args=args,full=full,ramp=ramp,pblh=pblh,
                            time_weight=result[0],height1=result[1],height2=result[2])
        inp.unlink(); out.unlink()
    manifest["oracle"] = json.loads((build/"build.json").read_text())
    manifest["helper_tolerance"] = "4*eps absolute time weight; 4*eps*max(1,abs(expected)) VIF heights"
    for name in ("OBS_DOMAIN101","decoded_r4.json","decoded_r8.json","helpers_r4.npz","helpers_r8.npz"):
        manifest["files"][name] = hashlib.sha256((save/name).read_bytes()).hexdigest()
    (save/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
