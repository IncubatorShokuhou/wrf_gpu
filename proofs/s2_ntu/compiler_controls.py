"""Independent pristine-WRF contraction and single-input +/-1 ULP envelope.

The sweep is fixed before observing the candidate. It does not expand to fit
the compiled port. Any excess remains an explicit experimental deviation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from build_oracle import VOLUME, SURFACE, NK


def read_column(folder, case):
    a=np.fromfile(folder/f'case_{case:02}.bin',dtype=np.float32)
    width=len(VOLUME)*NK+len(SURFACE)
    first=a[width+3*NK+NK+1+1:2*width+3*NK+NK+1+1]
    second=a[2*width+3*NK+NK+1+1:]
    assert first.size==second.size==width
    return np.stack((first,second))


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    a.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    original=root/'proofs/s2_ntu/fixtures/columns_r4.npz'
    tables=root/'src/gpuwrf/physics/ntu/data/aerosol_r4.npz'
    data=np.load(original); modes=[];observations=[]; manifests=[]
    # Each independent original input field is moved one representable REAL4
    # number, with other operands and original aerosol tables held fixed.
    variants=[('baseline',None,0)] + [(f'{n}_{direction:+}',n,direction)
        for n in VOLUME+['p','pii','dz','w'] for direction in (-1,1)]
    for fma in (False,True):
        folder=a.out/('FMA' if fma else 'noFMA')
        cmd=[sys.executable,str(Path(__file__).with_name('build_oracle.py')),str(folder),
             '--input-npz',str(original),'--tables-npz',str(tables),'--retain-build']
        if fma:cmd.append('--fma')
        subprocess.run(cmd,check=True,timeout=180,stdout=subprocess.DEVNULL)
        manifests.append(json.loads((folder/'manifest.json').read_text()))
        for label,field,direction in variants:
            for case in range(1,13):
                operands=[]
                for n in VOLUME+SURFACE:
                    v=data[f'c{case:02}_s0_{n}'].copy()
                    if n==field:
                        changed=np.nextafter(v,np.float32(np.inf if direction>0 else -np.inf))
                        v=np.where((v==0)&(direction<0),v,changed)
                    operands.append(v)
                for n in ['p','dz','pii','w','dt']:
                    v=data[f'c{case:02}_input_{n}'].copy()
                    if n==field:v=np.nextafter(v,np.float32(np.inf if direction>0 else -np.inf))
                    operands.append(v)
                np.concatenate(operands).tofile(folder/f'input_{case:02}.bin')
            subprocess.run([str(folder/'oracle'),'1','12','2','1'],cwd=folder,check=True,timeout=60,
                           stdout=subprocess.DEVNULL)
            values=np.stack([read_column(folder,c) for c in range(1,13)])
            assert np.isfinite(values).all(),(fma,label,'nonfinite original control')
            observations.append(values);modes.append(('FMA' if fma else 'noFMA')+':'+label)
            print(modes[-1],flush=True)
        # All consumer data are now detached; exact private builds are disposable.
        for n in ['error.F90','module_mp_ntu.F90','caller.F90','oracle','module_wrf_error.mod','module_mp_ntu.mod','frozen_tables.bin']:
            (folder/n).unlink(missing_ok=True)
        for c in range(1,13):
            for stem in ('case','input'):(folder/f'{stem}_{c:02}.bin').unlink(missing_ok=True)
    values=np.stack(observations)
    np.savez_compressed(a.out/'controls.npz',outputs=values)
    manifest=dict(scope='Independent original WRF, fixed single-input +/-1ULP and O3 native-FMA controls; no candidate values used',
        modes=modes,volume_fields=VOLUME,surface_fields=SURFACE,levels=NK,steps=2,cases=12,
        source_sha256=manifests[0]['source_sha256'],original_compilers=manifests,
        original_input_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        tables_sha256=hashlib.sha256(tables.read_bytes()).hexdigest(),
        outputs_sha256=hashlib.sha256((a.out/'controls.npz').read_bytes()).hexdigest(),
        nonfinite=0,shape=list(values.shape))
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
