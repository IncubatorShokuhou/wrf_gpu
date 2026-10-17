"""Additional original-nudob cases bracketing NINT(rko+.45)'s surface cutoff.

Usage: python make_boundary_savepoints.py <build-dir> <savepoint-dir>.
Reuses the qualified Swiss operands; acquires only previously missing fractional
RKO boundary cases, without regenerating or revalidating the original fixtures.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def generate(build, save):
    manifest = json.loads((save/"MANIFEST.json").read_text())
    for kind in ("r4","r8"):
        dtype = np.float32 if kind == "r4" else np.float64
        base_name = f"regime_3_{kind}.npz"
        with np.load(save/base_name) as source:
            base = {k:source[k] for k in source.files}
        o = manifest["cases"][base_name]["options"]
        cutoff = dtype(1.05)
        for index, rko0 in enumerate((np.nextafter(cutoff,dtype(-np.inf)),cutoff,np.nextafter(cutoff,dtype(np.inf)))):
            data = {k:v.copy() for k,v in base.items()}
            data["rko"][0] = rko0
            nobs = len(data["rio"])
            nz,nx = data["pbase"].shape
            varobs,errf = np.zeros((nobs,6),dtype),np.zeros((nobs,10),dtype)
            varobs[:,4] = data["pressure_cb"]
            errf[:,2],errf[:,4] = data["innovation"],data["pbl_level"]
            errf[:,5:8],errf[:,8] = data["base_pressure_cb"][:,None],data["rko"]
            inp,out = build/"boundary_in.bin",build/"boundary_out.bin"
            with inp.open("wb") as f:
                def write(a,dtype=dtype):
                    f.write(np.asarray(a,dtype).tobytes())
                write([nx,nz,o["jde"]-1,nobs,len(data["calls"]),3,o["j"],o["ionf"],o["enabled"],int(o["nudge_pbl"]),o["sfc_scheme_horiz"],o["sfc_scheme_vert"]],np.int32)
                write([o[k] for k in ("dx","rinxy","twindo","dtmin","coefficient","sfcfact","sfcfacr","dpsmx")])
                for a in (data["rio"],data["rjo"],data["rko"],data["time_hours"],varobs,errf):write(a)
                for key in ("mu","c1h","c2h","msfx","msfy","pbase","pp","kpbl","vih1","vih2","terrain","z"):
                    write(data[key],np.int32 if key=="kpbl" else dtype)
                write(data["initial_aten"])
                write(np.repeat(data["initial_saved"][:,:,None],6,axis=2))
                for ints,reals in zip(data["calls"],data["times"]):write(ints,np.int32);write(reals)
            subprocess.run([str(build/kind/"oracle"),str(inp),str(out)],check=True,timeout=20)
            result = np.fromfile(out,dtype).reshape(len(data["calls"]),2,nz,nx)
            assert np.isfinite(result).all()
            data["oracle_aten"],data["oracle_saved"] = result[:,0],result[:,1]
            name = f"rko_cutoff_{index}_3_{kind}.npz"
            np.savez_compressed(save/name,**data)
            sha = hashlib.sha256((save/name).read_bytes()).hexdigest()
            census = {"surface":int((data["rko"]==1).sum()),"upper":int((data["rko"]>=2).sum()),
                      "fractional_cutoff":1,"active_tendency_values":int(np.count_nonzero(result[:,0]-data["initial_aten"])),
                      "saved_nonzero":int(np.count_nonzero(result[:,1]))}
            manifest["cases"][name] = {"options":o,"sha256":sha,"census":census,
                                     "rko0":float(rko0),"purpose":"NINT(rko+.45) surface/upper cutoff"}
            manifest["files"][name] = sha
            inp.unlink();out.unlink()
    (save/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
