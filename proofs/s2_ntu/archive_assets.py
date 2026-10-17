"""Runtime assets must load through the real package from a clean git archive."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    a.out.mkdir(parents=True,exist_ok=False)
    checkout=a.out/'checkout';checkout.mkdir()
    tar=a.out/'source.tar'
    with tar.open('wb') as f:
        subprocess.run(['git','archive','HEAD','src','data/fixtures/thompson-tables-v1.npz',
                        'data/fixtures/thompson-cold-collection-v1.npz',
                        'data/fixtures/thompson-mixed-phase-v1.npz','data/fixtures/rrtmg-tables-v1.npz',
                        'proofs/s2_ntu/fixtures','pyproject.toml'],cwd=ROOT,stdout=f,check=True,timeout=60)
    subprocess.run(['tar','-xf',str(tar.resolve()),'-C',str(checkout.resolve())],check=True,timeout=60)
    code='''import hashlib,json,os
from pathlib import Path
import numpy as np
import gpuwrf
from gpuwrf.physics.ntu.tables import selected_tables,host_tables
root=Path.cwd().resolve()
assert Path(gpuwrf.__file__).is_relative_to(root)
out={}
for kind,dtype in [('r4',np.float32),('r8',np.float64)]:
    for ccnty in (1,2,3,4):
        values=selected_tables(dtype,ccnty)
        assert values['cnmod'].dtype==dtype and values['tblrc'].dtype==np.float64
        assert all(not a.flags.writeable for a in host_tables(kind).values())
        assert all(np.isfinite(a).all() for a in values.values())
    for family in ('columns','scalars','activation','initialization'):
        path=root/'proofs/s2_ntu/fixtures'/f'{family}_{kind}.npz'
        meta=json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['npz_sha256']
        with np.load(path,allow_pickle=False) as data: assert data.files
        out[f'{family}_{kind}']=path.stat().st_size
print(json.dumps(dict(assets=out,package=str(Path(gpuwrf.__file__).relative_to(root)),CPU=True,GPU=False)))
'''
    env={**os.environ,'PYTHONPATH':str(checkout.resolve()/'src'),'JAX_PLATFORMS':'cpu','GPUWRF_JAX_CACHE':'0'}
    r=subprocess.run([sys.executable,'-c',code],cwd=checkout,env=env,capture_output=True,text=True,timeout=60)
    if r.returncode:
        print(r.stderr,file=sys.stderr)
        r.check_returncode()
    result=json.loads(r.stdout.splitlines()[-1]);result['source']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['disposable_paths']=[str(checkout.resolve()),str(tar.resolve())]
    (a.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
