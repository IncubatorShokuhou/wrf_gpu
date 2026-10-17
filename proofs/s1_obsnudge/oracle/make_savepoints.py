"""Generate single-level nudob fixtures from original CPU-WRF Swiss columns.

Usage: python make_savepoints.py <oracle-build> <savepoint-dir>.
The synthetic innovations exercise the errob->nudob boundary; they do not
establish reader, errob or coupled-forecast fidelity. No CPU-WRF refs are edited.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gpuwrf.assimilation.obs_nudging import SingleLevelNudgingOptions  # noqa: E402

SOURCE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_01:00:00")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_array(f, a, dtype):
    # Model data use (k,i), Fortran stream uses (i,k) with i fastest.
    f.write(np.asarray(a, dtype=dtype).tobytes(order="C"))


def generate(build, out):
    out.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(SOURCE) as ds:
        def slab(name):
            a = np.asarray(ds[name][0])
            return a[:, 20, 15:27] if a.ndim == 3 else a[20, 15:27]
        base = slab("PB")
        pp = slab("P")
        zw = slab("PHB") + slab("PH")
        z = (zw[1:] + zw[:-1]) * np.float32(.5) / np.float32(9.81)
        terrain = slab("HGT")
        mu = slab("MU") + slab("MUB")
        c1h = np.asarray(ds["C1H"][0])
        c2h = np.asarray(ds["C2H"][0])
        msfx = slab("MAPFAC_MX")
        msfy = slab("MAPFAC_MY")
        dx = float(ds.DX)
    nz, nx = base.shape
    nobs, ny = 8, 10
    manifest = {
        "fixture_id": "s1-obsnudge-single-level-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "source_sha256": sha(SOURCE),
        "source_commit": "pristine WRF sources bound by build hashes", "oracle": json.loads((build / "build.json").read_text()),
        "scenario": "Original CPU-WRF Swiss h1 columns with synthetic independent observation innovations",
        "license_notes": "WRF source routines remain external; fixtures generated from original CPU-WRF reference fields",
        "boundary": "single-level nudob only; not OBS_DOMAIN ingestion, errob, coupled forecast or restart",
        "tolerance": "16*eps*(abs(input ATEN)+coefficient*max(coupling)*nobs*max(abs(valid innovation))) for ATEN; "
                     "16*eps*nobs*max(abs(valid innovation)) for SAVWT (FMA/reassociation envelope fixed before comparison)",
        "variables": {"aten": {"units": "rho/map-coupled m/s2 or hybrid-mass-coupled K/s or kg/kg/s",
                                 "shape": [6, nz, nx], "dtype": "float32/float64", "absolute_tolerance": "see tolerance", "relative_tolerance": 0},
                      "saved": {"units": "m/s, K or kg/kg innovation", "shape": [6, nz, nx],
                                "dtype": "float32/float64", "absolute_tolerance": "see tolerance", "relative_tolerance": 0}},
        "cases": {}, "files": {},
    }
    # Refresh, hold, refresh, restart-refresh at a non-ionf step, refresh, hold.
    calls = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 3, 1], [4, 0, 0], [5, 0, 0]], np.int32)
    for kind in ("r4", "r8"):
        dtype = np.float32 if kind == "r4" else np.float64
        for ivar in (1, 2, 3, 4):
            for scenario in ("regime", "mm5", "no_pbl", "boundary"):
                name = f"{scenario}_{ivar}_{kind}.npz"
                o = SingleLevelNudgingOptions(ivar=ivar, j=2 if scenario == "boundary" else 5, jde=ny+1,
                    dx=dx, rinxy=dx/1000*2.5, twindo=.1, dtmin=.3, coefficient=3e-4, ionf=2,
                    nudge_pbl=scenario != "no_pbl", sfcfact=.75, sfcfacr=1.2,
                    sfc_scheme_horiz=int(scenario == "mm5"), sfc_scheme_vert=int(scenario == "mm5"))
                # Both active surface and upper-air obs; missing, outside radius,
                # and outside time window controls. Pressure distance is in cb.
                rio = np.asarray([5.5, 6.25, 7.5, 6.5, -99., 5.5, 5.5, 8.5], dtype)
                rjo = np.asarray([5.5, 5.5, 4.75, 200., 5.5, 5.5, 5.5, 6.5], dtype)
                rko = np.asarray([1., 1., 14., 1., 1., 18., 16., 8.], dtype)
                timeob = np.asarray([0., .03, .05, 0., 0., 9., 0., -.04], dtype)
                pind = np.asarray(.001*(base+pp), dtype)
                pressure = np.asarray([90.,90.,pind[13,6],90.,90.,pind[17,5],pind[15,5],pind[7,7]], dtype)
                factor = 1e-3 if ivar == 4 else 1.
                innovation = np.asarray([2., -1., 3., 8., 8., 8., 999999., -2.], dtype) * dtype(factor)
                innovation[6] = dtype(999999.)
                psurf = np.asarray([base[0,4],base[0,5],base[0,6],base[0,5],base[0,5],base[0,5],base[0,5],base[0,7]], dtype)*dtype(.001)
                obpbl = np.full(nobs, 7, dtype)
                errf = np.zeros((nobs,10), dtype)
                errf[:,ivar-1] = innovation
                errf[:,4] = obpbl
                errf[:,5:8] = psurf[:,None]
                errf[:,8] = rko
                varobs = np.zeros((nobs,6), dtype)
                varobs[:,4] = pressure
                kpbl = np.asarray([1,3,4,5,6,7,8,9,12,26,4,6], np.int32)
                vih1 = np.asarray([50,100,500,1000,300,200,800,500,200,100,60,50], dtype)
                vih2 = vih1 + dtype(500)
                aten = np.arange(nx*nz, dtype=dtype).reshape(nz,nx) * dtype(1e-6)
                saved = np.full_like(aten, dtype(.125*factor))
                times = np.asarray([[.3,1.], [1.,.8], [3.,.6], [4.,.4], [8.,.2], [9.,0.]], dtype)
                data = dict(rio=rio,rjo=rjo,rko=rko,time_hours=timeob,pressure_cb=pressure,
                            innovation=innovation,base_pressure_cb=psurf,pbl_level=obpbl,
                            mu=mu.astype(dtype),c1h=c1h.astype(dtype),c2h=c2h.astype(dtype),
                            msfx=msfx.astype(dtype),msfy=msfy.astype(dtype),pbase=base.astype(dtype),pp=pp.astype(dtype),
                            kpbl=kpbl,vih1=vih1,vih2=vih2,terrain=terrain.astype(dtype),z=z.astype(dtype),
                            initial_aten=aten, initial_saved=saved,calls=calls,times=times)
                inpath, outpath = build / "input.bin", build / "output.bin"
                with inpath.open("wb") as f:
                    write_array(f, [nx,nz,ny,nobs,len(calls),ivar,o.j,o.ionf,o.enabled,int(o.nudge_pbl),o.sfc_scheme_horiz,o.sfc_scheme_vert], np.int32)
                    write_array(f, [o.dx,o.rinxy,o.twindo,o.dtmin,o.coefficient,o.sfcfact,o.sfcfacr,o.dpsmx], dtype)
                    for arr in (rio,rjo,rko,timeob,varobs,errf):
                        write_array(f, arr, dtype)
                    for key in ("mu","c1h","c2h","msfx","msfy","pbase","pp","kpbl","vih1","vih2","terrain","z"):
                        write_array(f, data[key], np.int32 if key == "kpbl" else dtype)
                    write_array(f, aten, dtype)
                    # Fortran SAVWT(6,nx,nz), variable index fastest.
                    write_array(f, np.repeat(saved[:,:,None], 6, axis=2), dtype)
                    for ints,reals in zip(calls,times):
                        write_array(f, ints, np.int32)
                        write_array(f, reals, dtype)
                subprocess.run([str(build/kind/"oracle"),str(inpath),str(outpath)], check=True, timeout=20)
                result = np.fromfile(outpath,dtype).reshape(len(calls),2,nz,nx)
                if not np.isfinite(result).all():
                    raise ValueError(f"nonfinite oracle {name}")
                data.update(oracle_aten=result[:,0],oracle_saved=result[:,1])
                np.savez_compressed(out/name,**data)
                manifest["cases"][name] = {"options": asdict(o), "sha256": sha(out/name),
                    "census": {"surface": int((rko == 1).sum()), "upper": int((rko > 1).sum()),
                               "active_tendency_values": int(np.count_nonzero(result[:,0]-aten)),
                               "saved_nonzero": int(np.count_nonzero(result[:,1]))}}
                manifest["files"][name] = sha(out/name)
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for path in (build/"input.bin",build/"output.bin"):
        path.unlink()


if __name__ == "__main__":
    generate(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
