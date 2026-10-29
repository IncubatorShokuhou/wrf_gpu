"""G01: bounded late-hour census and real WRF column extraction, no model execution.

Read original CPU files only through ALISIOS facade_stage, one file at a time.
Remove only our facade copy after extraction; preserve every original/reference.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import netCDF4

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G01')
GPU = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/runs/rw3_20261027T094334Z_1056334/runs/N72/wrfout')
PAIR = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate/arms0227/W3P/d6_72h/summary.json')
KEYS = ('qv','qc','qr','qi','qs','qg','ni','nr','th','pii','p','w','dz8w')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def inspect_frame(path, label, lead, extra=()):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        qg = np.asarray(ds['QGRAUP'][0], np.float32)
        mx = qg.max(0)
        active = np.argwhere(mx > 0)
        picks = active if lead >= 71 and len(active) <= 12 else np.array(
            np.unravel_index(np.argsort(mx.ravel())[-4:][::-1], mx.shape)).T
        picks = sorted({tuple(map(int, p)) for p in picks if mx[tuple(p)] > 0} | set(extra))
        row = dict(label=label, lead=lead, file=str(path), sha256=sha(path), shape=list(qg.shape),
                   positive_cells=int(np.count_nonzero(qg)), positive_columns=len(active),
                   above_R1=int(np.count_nonzero(qg > 1e-12)), max=float(qg.max()),
                   positive_levels=np.flatnonzero((qg > 0).any((1,2))).tolist(),
                   active_yx=active.tolist() if lead >= 70 else None,
                   dt_attribute=float(getattr(ds, 'DT', 6.0)), columns=[])
        cols = []
        for y,x in picks:
            g = lambda name: np.asarray(ds[name][0,:,y,x], np.float32)
            qv,qc,qr,qi,qs,qgra,ni,nr = (g(k) for k in
                ('QVAPOR','QCLOUD','QRAIN','QICE','QSNOW','QGRAUP','QNICE','QNRAIN'))
            p = g('P') + g('PB')
            th = g('T') + np.float32(300)
            pii = (p / np.float32(1e5)) ** np.float32(2/7)
            dz = np.diff((g('PH') + g('PHB')) / np.float32(9.81))
            vals = (qv,qc,qr,qi,qs,qgra,ni,nr,th,pii,p,g('W')[:-1],dz)
            name = f'{label}_h{lead:02}_y{y}_x{x}'
            cols.append((name,dict(zip(KEYS, vals))))
            row['columns'].append(dict(name=name,y=y,x=x,qg_max=float(qgra.max()),
                                       nonzero_levels=np.flatnonzero(qgra).tolist()))
    return row, cols, picks

def main():
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    OUT.mkdir(parents=True,exist_ok=True)
    spec = importlib.util.spec_from_file_location('graupel_guard', ROOT/'src/gpuwrf/assimilation/letkf/alisios_guard.py')
    ag = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ag; spec.loader.exec_module(ag)
    pairs = json.loads(PAIR.read_text())['domains']['d03']['pairing']['pairs']
    rows, cols = [], []
    for pair in pairs:
        lead = pair['lead_h']
        if not 60 <= lead <= 72: continue
        if Path('/tmp/wrf_gpu2_quiet').exists(): raise SystemExit('QUIET mid-census; retry with fresh stages')
        src = pair['cpu_file']; day = Path(src).name.split('_')[2].replace('-','')
        stage = OUT/f'stage_h{lead}'
        staged, receipt = ag.facade_stage({src: (['20260227'],[day])}, stage, OUT/f'receipt_h{lead}',
            purpose='v45-graupel G01: d03 late graupel trace localization and pristine-column replay',
            sealed_root=OUT/'facade')
        private = Path(staged[str(ag.resolve_path(src))])
        cpu, cc, picks = inspect_frame(private,'cpu',lead)
        cpu['original'] = src
        assert cpu['sha256'] == receipt['files'][str(ag.resolve_path(src))]['sha256']
        gpu, gc, _ = inspect_frame(GPU/Path(src).name,'gpu',lead,picks)
        rows.extend((cpu,gpu)); cols.extend(cc+gc)
        private.unlink()  # Exact, owned, facade copy only.
        for d in sorted(stage.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if d.is_dir(): d.rmdir()
        stage.rmdir()
        print(lead, 'cpu',cpu['positive_cells'],cpu['max'],'gpu',gpu['positive_cells'],gpu['max'],flush=True)
        (OUT/'census.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(OUT/'columns.npz',names=np.array([n for n,c in cols]),dt=np.full(len(cols),6,np.float32),
        **{f'in_{k}':np.stack([c[k] for n,c in cols]) for k in KEYS})
    print('columns',len(cols),'sha256',sha(OUT/'columns.npz'),flush=True)

if __name__ == '__main__': main()
