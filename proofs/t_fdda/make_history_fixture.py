"""Save a real original-WRF call's operands and held HFX_FDDA (18s -> 36s)."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

root, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
ys, xs = slice(10,13), slice(12,16)
vals = {}
with Dataset(root/'wrffdda_d01') as ds:
    for name, var in (('u','U'),('v','V'),('t','T'),('q','Q')):
        y = slice(10,14) if name == 'v' else ys
        x = slice(12,17) if name == 'u' else xs
        for suffix in ('old','new'):
            vals[f'{name}_{suffix}'] = np.asarray(ds[f'{var}_NDG_{suffix.upper()}'][0,:,y,x],np.float32)[None]
for sec in (0,18,36):
    p = root/'repaired'/f'wrfout_d01_2023-01-15_00:00:{sec:02d}'
    with Dataset(p) as ds:
        for n in ('T','THM','P','PB','PH','PHB','QVAPOR','HFX_FDDA'):
            vals[f's{sec}_{n}'] = np.asarray(ds[n][0,:,ys,xs],np.float32)
        for n in ('U','V'):
            y = slice(10,14) if n == 'V' else ys
            x = slice(12,17) if n == 'U' else xs
            vals[f's{sec}_{n}'] = np.asarray(ds[n][0,:,y,x],np.float32)
        for n in ('HGT','PBLH','MU','MUB'):
            vals[f's{sec}_{n}'] = np.asarray(ds[n][0,ys,xs],np.float32)
path = out/'wrf_grid1_hfx.npz'
np.savez_compressed(path,**vals)
manifest = {'source':str(root),'reference':'original CPU-WRF 4.7.1; actual ingested NEW fields byte audited',
            'executable':json.loads((root/'PREPARATION.json').read_text())['private_sha256'],
            'selection':{'j':[10,13],'i':[12,16],'k':'all44'},'dt_s':18,'guv':3e-4,'gt':3e-4,'gq':3e-4,
            'interval_m':60,'end_h':24,'analysis_record':0,'hfx_call_entry_s':18,'hfx_history_s':36,
            'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'regime':{'hfx_nonzero':int(np.count_nonzero(vals['s36_HFX_FDDA'])),
                      'hfx_max_abs':float(np.max(np.abs(vals['s36_HFX_FDDA'])))} }
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
