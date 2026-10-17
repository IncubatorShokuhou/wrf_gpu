"""Reproduce every excess against the fixed independent WRF enclosure."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCES=[*(f'src/gpuwrf/physics/ntu/{n}.py' for n in
           ('constants','tables','math','support','_native','column','scalar_generated','aerosol_generated','phases_generated')),
         'src/gpuwrf/kernels/phys_ntu.py','src/gpuwrf/kernels/_ntu_math.py']


def code_hash(path):
    tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def compare(control_folder,observations):
    m=json.loads((control_folder/'manifest.json').read_text())
    control_file=control_folder/'controls.npz'
    assert hashlib.sha256(control_file.read_bytes()).hexdigest()==m['outputs_sha256']
    controls=np.load(control_file)['outputs']; obs=np.load(observations)
    volumes,surfaces=m['volume_fields'],m['surface_fields']
    actual=np.stack([np.stack([np.concatenate([*(obs[f'c{c:02}_s{s}_{n}'].ravel() for n in volumes),
        *(obs[f'c{c:02}_s{s}_{n}'].ravel() for n in surfaces)]) for s in (1,2)]) for c in range(1,13)])
    assert actual.dtype==np.float32 and np.isfinite(actual).all() and np.isfinite(controls).all()
    low,high=controls.min(axis=0),controls.max(axis=0)
    bad=(actual<low)|(actual>high); stats={}; rows=[];offset=0
    for name,width in [(n,m['levels']) for n in volumes]+[(n,1) for n in surfaces]:
        ix=slice(offset,offset+width);a=actual[:,:,ix];b=bad[:,:,ix];ref=controls[0,:,:,ix]
        excess=np.maximum(0,np.maximum(low[:,:,ix]-a,a-high[:,:,ix]))
        stats[name]=dict(outside=int(b.sum()),max_abs_vs_baseline=float(np.abs(a-ref).max()),
                        max_excess_abs=float(excess.max()))
        for c,s,k in np.argwhere(b):
            rows.append(dict(case=int(c+1),stage=int(s+1),field=name,index=int(k),value=float(a[c,s,k]),
                             baseline=float(ref[c,s,k]),low=float(low[c,s,offset+k]),high=float(high[c,s,offset+k])))
        offset+=width
    return dict(CPU=True,GPU=False,classification='Experimental compiled deviations; NOT an original-enclosure or GPU/D6 pass',
                controls=len(controls),values=actual.size,nonfinite_values=0,outside_original_enclosure=int(bad.sum()),
                fields=stats,deviations=rows,observations_sha256=hashlib.sha256(observations.read_bytes()).hexdigest(),
                original_controls_sha256=m['outputs_sha256'],source_code_sha256={p:code_hash(ROOT/p) for p in SOURCES})


def main():
    p=argparse.ArgumentParser();p.add_argument('control_folder',type=Path);p.add_argument('observations',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    result=compare(a.control_folder,a.observations)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('fields','deviations','source_code_sha256')})


if __name__=='__main__':main()
