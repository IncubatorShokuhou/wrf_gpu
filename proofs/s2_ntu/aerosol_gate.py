"""Exact original aerosol calls; no fixture-derived expected computation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from literal_numpy import LiteralNumpy
from gpuwrf.physics.ntu.aerosol_generated import activa
from gpuwrf.physics.ntu.math import find_rc0
from gpuwrf.physics.ntu.tables import selected_tables


def main():
    p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    meta=json.loads((a.oracle/'manifest.json').read_text());file=a.oracle/'activation.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['npz_sha256']
    data=np.load(file);xp=LiteralNumpy();report={};different=0
    with np.errstate(all='ignore'):
        for ccnty in range(1,5):
            before=data['before'][ccnty-1].reshape(-1,meta['width']);expected=data['after'][ccnty-1].reshape(before.shape)
            v={k:xp.asarray(before[:,i]) for i,k in enumerate(meta['fields'])}
            v.update(abcd=xp.asarray(before[:,8:12]),zccns=xp.asarray(before[:,12:15]),do_fr=before[:,15].astype(bool),iae=np.int32(1))
            c=selected_tables(before.dtype,ccnty)
            rc=find_rc0(xp.asarray(v['abcd'][:,0]/v['abcd'][:,1],dtype=np.float64),c['cnmod'][:,0],c['cnstd'][:,0],c['wmas'][:,0],c['tblxf'][:,0],c['tblrc'],xp=xp)
            output=activa(v,xp=xp,c=c)
            actual=np.column_stack([output[k] for k in meta['fields']]+[output['abcd'],output['zccns'],output['do_fr'].astype(before.dtype)])
            assert np.isfinite(actual).all() and np.isfinite(rc).all(),(ccnty,'nonfinite')
            itype='uint32' if before.dtype.itemsize==4 else 'uint64'
            bad=actual.view(itype)!=expected.view(itype);rctruth=data['rc'][ccnty-1].reshape(-1)
            rcbad=np.asarray(rc).view(itype)!=rctruth.view(itype)
            different+=int(bad.sum()+rcbad.sum())
            report[str(ccnty)]={'different':int(bad.sum()),'rc_different':int(rcbad.sum()),
                                'bad_indices':np.argwhere(bad).tolist(),'rc_bad_rows':np.flatnonzero(rcbad).tolist(),
                                'max_abs':float(np.max(np.abs(actual-expected))),'rc_max_abs':float(np.max(np.abs(rc-rctruth)))}
            print(ccnty,report[str(ccnty)],flush=True)
    result={'kind':meta['kind'],'calls':meta['calls'],'different':different,'BYTE':different==0,'ccnty':report}
    a.out.write_text(json.dumps(result,indent=2)+'\n');assert result['BYTE'],'NTU aerosol operator differs from original WRF'


if __name__=='__main__':main()
