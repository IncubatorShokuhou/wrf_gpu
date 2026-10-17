"""Final product boundary vs unchanged native RADTN whole-call savepoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gpuwrf.physics import gfdleta_column as boundary
from gpuwrf.physics import ra_gfdleta as optics
from gpuwrf.physics import ra_gfdleta_support as support
from literal import load,_wrap


def main(root,output):
    with np.load(root/'tests/fixtures/s2_gfdl_cloud_v1.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    with np.load(root/'tests/fixtures/s2_gfdl_radtn_v1.npz',allow_pickle=False) as f:native={k:f[k].copy() for k in f.files}
    generated=root/'src/gpuwrf/physics/_gfdleta_radtn.py'
    assert generated.read_bytes()==(root/'proofs/s2_flggfdl/generated_radtn.py').read_bytes()
    model=load(generated,source_control=True);physics=load(Path(optics.__file__));clock=load(Path(support.__file__))
    port=load(Path(boundary.__file__));port.solar_distance=clock.solar_distance
    provider={k[4:]:v for k,v in native.items() if k.startswith('co2_')}
    domain={k[7:]:v for k,v in native.items() if k.startswith('domain_')}
    nz=data['in_fields'].shape[1]-1
    solver=model.reference_factory(nz,provider,domain,physics.cloud_overlap,
        physics.ferrier_saturation_pressure,physics.ozone_column,clock.zenith_dynamic)
    fields=data['in_fields'];surface=data['in_sfc'];results=[]
    for row in range(len(fields)):
        c={key:_wrap(fields[row,:nz,col]) for key,col in {'t':0,'p':1,'qv':5,'qc':6,'qi':8,'qs':9}.items()}
        c['p8w']=_wrap(fields[row,:,12])
        for key,col in {'tsk':0,'albedo':2,'xland':3,'snow':6,'vegetation':7}.items():c[key]=np.float32(surface[row,col])
        c['latitude']=native['glat'][row].reshape(())[()];c['longitude']=native['glon'][row].reshape(())[()]
        previous={key:native[key][row].reshape(())[()] for key in ('acfrcv','ncfrcv','acfrst','ncfrst')}
        previous.update(top=surface[row,8],bottom=surface[row,9],cuppt=surface[row,10])
        args=tuple(native[k][row] for k in ('idat','ihrst','xtime','julian'))
        raw=port.native_column(solver,c,args,previous,dt=native['dt'][row],
            stepra=native['nrads'][row],nphs=native['nphs'][row])
        checks={key:np.asarray(value).dtype==native['out_'+key][row].dtype and
            np.asarray(value).shape==native['out_'+key][row].shape and
            np.asarray(value).tobytes()==native['out_'+key][row].tobytes() for key,value in raw.items()}
        results.append(dict(row=row,fields=checks));print(row,all(checks.values()),flush=True)
        output.write_text(json.dumps(dict(all40x17_BYTE=all(all(r['fields'].values()) for r in results),
            completed=len(results),results=results,solver_copy_BYTE=True,
            boundary_sha256=hashlib.sha256(Path(boundary.__file__).read_bytes()).hexdigest(),
            scope='product whole-call mapping+SOLARD/preprocessing; source literal REAL4/nativeDOUBLE; no runtime oracle call'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.root,a.output)
