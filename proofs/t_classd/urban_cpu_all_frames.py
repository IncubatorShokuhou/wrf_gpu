"""Original CPU-WRF c57 (BEP+BEM, sf_urban_physics=3) green-roof/PV history, every cell/all frames.

Usage: python proofs/t_classd/urban_cpu_all_frames.py <c57_case_dir> <out.json> [<extract.npz>]
Checks WRF's semantics with PV/green roof off (URBPARM GR_FLAG=PV_FRAC_ROOF=GR_FRAC_ROOF=0):
TGR_URB4D/DRAINGR_URB3D/EP_PV_URB3D are +0 bits; QGR_URB4D(t0) == SMOIS(t0, top layer) bitwise;
QGR_URB4D(t>0) == QGR_URB4D(t0) with every IVGTYP==ISURBAN column set to +0.
The optional extract keeps the 42x42 frame-0 SMOIS top / QGR / IVGTYP and one later frame for tests.
"""
import glob, hashlib, json, sys
import numpy as np, netCDF4

case, out = sys.argv[1], sys.argv[2]
extract = sys.argv[3] if len(sys.argv) > 3 else None
files = sorted(glob.glob(f'{case}/cpu/wrfout_d01_*'))
with netCDF4.Dataset(files[0]) as ds:
    isurban = int(ds.getncattr('ISURBAN'))
    smois0 = np.asarray(ds['SMOIS'][0, 0], np.float32)
    qgr0 = np.asarray(ds['QGR_URB4D'][0], np.float32)
    ivg = np.asarray(ds['IVGTYP'][0])
urban = ivg == isurban
expected_later = np.where(urban, np.float32(0), qgr0)
frames, ok = [], True
for path in files:
    rec = dict(file=path.split('/')[-1], sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest(), fields={})
    with netCDF4.Dataset(path) as ds:
        for name in ('TGR_URB4D', 'DRAINGR_URB3D', 'EP_PV_URB3D', 'QGR_URB4D'):
            v = ds[name]
            a = np.asarray(v[0], np.float32)
            if name != 'QGR_URB4D':
                good = bool(np.all(a.view(np.uint32) == 0))
            elif path == files[0]:
                good = a.tobytes() == smois0.tobytes()
            else:
                good = a.tobytes() == expected_later.tobytes()
            ok &= good
            rec['fields'][name] = dict(match=good, dims=list(v.dimensions), dtype=str(v.dtype),
                                       nonzero=int(np.count_nonzero(a)), units=v.units, stagger=v.stagger)
    frames.append(rec)
summary = dict(case=case, frames=len(frames), isurban=isurban, urban_columns=int(urban.sum()),
               qgr_t0_nonzero=int(np.count_nonzero(qgr0)), all_match=bool(ok))
json.dump(dict(summary=summary, frames=frames), open(out, 'w'), indent=1)
if extract:
    with netCDF4.Dataset(files[-1]) as ds:
        last = np.asarray(ds['QGR_URB4D'][0], np.float32)
    np.savez_compressed(extract, smois_top_t0=smois0, qgr_t0=qgr0, ivgtyp=ivg.astype(np.int32),
                        qgr_last=last, isurban=np.int32(isurban),
                        source_sha256=np.asarray([frames[0]['sha256'], frames[-1]['sha256']]))
print(json.dumps(summary))
