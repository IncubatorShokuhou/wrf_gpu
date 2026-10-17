"""Produce independent pristine-Fortran radar data, never port-generated truth."""
from pathlib import Path
import argparse
import json
import subprocess
import hashlib
import numpy as np
from netCDF4 import Dataset


def generate(oracle, reference, output):
    fields = []
    provenance = []
    # Exact input columns from radar-ON CPU-WRF history (last mass axis vertical).
    # ng is not stored: only take columns with radar-inactive graupel. These are
    # ORIGINAL WRF fields passed to the original operator, not claimed call-site
    # savepoints (history and diagnostic sample times differ slightly).
    for path in sorted(reference.glob('wrfout_d01_*'))[1:]:
        with Dataset(path) as ds:
            p = (ds['P'][:]+ds['PB'][:]).astype('f4')
            t = (ds['T'][:]+300)*(p/100000)**np.float32(287/1004.5)
            names = ['QVAPOR','QCLOUD','QRAIN','QNRAIN','QSNOW','QGRAUP']
            a = [np.asarray(ds[k][:]) for k in names]
            a += [np.zeros_like(a[0]), np.zeros_like(a[0]), np.asarray(t), np.asarray(p)]
            values = np.stack([np.moveaxis(v[0],0,-1) for v in a],axis=1) # y,field,x,z
            candidates = np.argwhere((a[4][0].max(axis=0) > 0) & (a[5][0].max(axis=0) <= 1e-6))
            if len(candidates) == 0:
                raise ValueError(f'{path}: no eligible snow/low-graupel columns')
            for y,x in candidates[np.linspace(0,len(candidates)-1,8,dtype=int)]:
                fields.append(values[y,:,x,:])
            provenance.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    nz=fields[0].shape[-1]
    # Adversarial columns cover otherwise absent graupel, melting, dense rain,
    # dry snow, WRF thresholds, number sensitivity and all-clear conditions.
    for regime in range(16):
        a=np.zeros((10,nz),dtype='f4')
        a[0]=.005
        a[8]=np.linspace(285,240,nz)
        a[9]=np.linspace(95000,30000,nz)
        if regime & 1:
            a[2]=np.geomspace(1e-8,.003,nz)
            a[3]=np.geomspace(1e2,1e5,nz)
        if regime & 2:
            a[4]=np.geomspace(1e-6,.001,nz)
        if regime & 4:
            a[5]=np.geomspace(1e-6,.002,nz)
            a[6]=np.geomspace(1e1,1e4,nz)
        if regime & 8:
            a[8]=255 # disable warm layer and wet-snow branch
        fields.append(a)
    a=np.asarray(fields,dtype='f4')
    inp=f'{len(a)} {nz}\n'+'\n'.join(' '.join(format(float(x),'.9e') for x in row) for col in a for row in col)+'\n'
    run=subprocess.run([str(oracle/'oracle')],input=inp,text=True,capture_output=True,check=True,timeout=60)
    z=np.fromstring(run.stdout,sep=' ').reshape(len(a),nz).astype('f4')
    if not np.isfinite(z).all():
        raise ValueError('nonfinite pristine result')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'thompson_oracle.npz',inputs=a,dbz=z)
    manifest=json.loads((oracle/'manifest.json').read_text())
    manifest.update(reference_files=provenance,constructed_columns=16,real_field_columns=len(a)-16,
                    fixture_sha256=hashlib.sha256((output/'thompson_oracle.npz').read_bytes()).hexdigest(),
                    note='Real history fields plus constructed active regimes; reference is pristine operator, not port. Missing history ng is zero only in radar-inactive graupel columns.')
    (output/'oracle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['oracle','reference','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    generate(args.oracle,args.reference,args.output)
