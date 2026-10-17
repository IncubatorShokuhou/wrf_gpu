"""Freeze two consecutive pristine MP calls on actual Swiss CPU-WRF columns."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng", "Nn", "Nc")
OPTIONS = (1, 3, 4, 6, 10, 13, 14, 97)
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_06:00:00")


def generate(build, destination):
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists()
    with Dataset(CASE) as ds:
        assert ds.USE_THETA_M == 1
        get = lambda key: np.asarray(ds[key][0], np.float32)
        qfields = {k: get(v) for k, v in zip(FIELDS[1:7], ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"))}
        points = []
        for field in ("qv", "qc", "qr", "qi", "qs", "qg"):
            # Interior real columns; no alteration to observed thermodynamic/moisture fields.
            colscore = qfields[field][:, 2:-2, 2:-2].max(axis=0)
            y, x = np.unravel_index(np.argmax(colscore), colscore.shape)
            point = (int(y)+2, int(x)+2)
            if point not in points:
                points.append(point)
        ys, xs = np.array(points).T
        pick = lambda a: a[:, ys, xs].T.copy()
        phi = get("PH")+get("PHB")
        data = {"theta": pick(get("THM")+np.float32(300)), "p": pick(get("P")+get("PB")),
                "ph": pick(phi), "w": pick(get("W"))}
        data.update({k: pick(v) for k, v in qfields.items()})
        data["Ni"] = pick(get("QNICE")); data["Nr"] = pick(get("QNRAIN"))
        for field in ("Ns", "Ng"):
            data[field] = np.zeros_like(data["theta"])
        # Additional scheme-specific number fields are initialized, not observed Thompson values.
        data["Nn"] = np.full_like(data["theta"], 1.e8)
        data["Nc"] = np.full_like(data["theta"], 3.e8)
    ncol, nz = data["theta"].shape
    interface_z = data["ph"].astype(np.float64)/9.80665
    z = ((interface_z[:, :-1]+interface_z[:, 1:])*.5).astype(np.float32)
    dz = np.maximum(np.diff(interface_z, axis=1), 1.).astype(np.float32)
    pii = ((data["p"]/np.float32(1.e5))**np.float32(2/7)).astype(np.float32)
    w = (np.float32(.5)*(data["w"][:, :-1]+data["w"][:, 1:])).astype(np.float32)
    fixture = {"input_"+k: v for k,v in data.items()}
    for option in OPTIONS:
        records = []
        for c in range(ncol):
            payload = np.zeros((nz,20),dtype=np.float32,order="F")
            for f, name in enumerate(FIELDS):
                payload[:,f] = data[name][c]
            for f, array in enumerate((data["p"], pii, z, dz, w), start=13):
                payload[:,f] = array[c]
            inp, out = build/"input.bin", build/"output.bin"
            with inp.open("wb") as stream:
                np.array([option,nz],np.int32).tofile(stream)
                np.array([18.,interface_z[c,0]],np.float32).tofile(stream)
                stream.write(payload.tobytes(order="F"))
            subprocess.run([str(build/"caller"),str(inp),str(out)], check=True, timeout=30)
            raw = np.fromfile(out,np.float32).reshape(2,nz*15+3)
            assert np.isfinite(raw).all(), (option,c)
            records.append(raw)
        raw = np.stack(records,axis=1)
        fixture[f"mp{option}_out"] = raw[:,:,:nz*15].reshape(2,ncol,15,nz)
        fixture[f"mp{option}_precip"] = raw[:,:,nz*15:]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/"swiss_two_step.npz"
    np.savez_compressed(path,**fixture)
    manifest = json.loads((build/"build.json").read_text())
    manifest.update(fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_file=str(CASE), input_sha256=hashlib.sha256(CASE.read_bytes()).hexdigest(),
        points_y_x=points, fields=FIELDS, options=OPTIONS, steps=2, dt_s=18., ncol=ncol, nz=nz,
        active_max={k:float(v.max()) for k,v in data.items() if k in FIELDS},
        number_initialization="Ns/Ng=0 (not observed); Nn=1e8 and Nc=3e8 initialized for WDM5; Ni/Nr observed QNICE/QNRAIN",
        wdm5_itimestep=[1201,1202],
        geometry="port caller ph/9.80665 and Exner held equal to isolate theta seam; not a geometry/forecast gate")
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("ncol","nz","active_max","fixture_sha256")}))


if __name__ == "__main__":
    generate(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
