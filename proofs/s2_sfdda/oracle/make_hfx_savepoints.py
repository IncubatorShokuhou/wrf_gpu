"""Changed HFX output only: replay frozen operands through unmodified WRF."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from make_savepoints import run_oracle,unfort
build,out=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).parents[1]/'savepoints'
manifest=json.loads((root/'MANIFEST.json').read_text())
work=build/'hfx-work';work.mkdir(exist_ok=True)
arrays={};audit={}
for name,meta in manifest['cases'].items():
    with np.load(root/name) as ds:d=dict(ds)
    extract=lambda prefix:{k[len(prefix):]:v for k,v in d.items() if k.startswith(prefix)}
    case=dict(frames=[extract(f'frame{i}_') for i in (1,2)],
              records=[{k:v for k,v in extract(f'rec{i}_').items() if k[0] in ('u','v','t','q') and not k.startswith(('u10','v10','th2','q2'))} for i in (1,2)],
              surface=[extract(f'rec{i}_') for i in (1,2)],**{k:d[k] for k in ('c1h','c2h','ht')})
    calls=[dict(zip(('itimestep','secs','rec','frame'),map(int,c))) for c in d['calls']]
    kind='r4' if name.endswith('r4.npz') else 'r8'
    path=work/'hfx.bin'
    results=run_oracle(build/kind/'fdda_oracle',case,meta['opts'],calls,kind,work,hfx_path=path)
    # New dz8w affects HFX only. All existing independent oracle outputs retained byte for byte.
    for ci,res in enumerate(results):
        for k,v in res.items():assert np.array_equal(v,d[f'call{ci}_{k}']), (name,ci,k)
    shape=d['frame1_th_phy'].shape;size=int(np.prod(shape))
    raw=np.fromfile(path,dtype=np.float32 if kind=='r4' else np.float64)
    assert raw.size==len(calls)*size and np.isfinite(raw).all()
    arrays[name]=np.stack([unfort(raw[i*size:(i+1)*size],shape) for i in range(len(calls))])
    audit[name]={'original_fixture_sha256':meta['sha256'],'all_prior_outputs_byte_identical':True,'active_hfx':int(np.count_nonzero(raw))}
np.savez_compressed(out/'wrf_surface_hfx.npz',**arrays)
receipt={'pristine':(build/'pristine_sha256.txt').read_text(),'cases':audit,'sha256':hashlib.sha256((out/'wrf_surface_hfx.npz').read_bytes()).hexdigest()}
(out/'HFX_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'cases':len(arrays),'all_prior_outputs_byte_identical':True}))
