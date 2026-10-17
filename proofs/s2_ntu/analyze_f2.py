"""Classify every captured NTU event, without changing values or tolerances."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--counts-from',type=Path)
    a=p.parse_args();manifest=json.loads((a.capture/'manifest.json').read_text())
    rows=[r for r in manifest if r['step']==2]
    last=lambda kind:max((r for r in rows if r['kind']==kind),key=lambda r:r['call'])
    def load(row):
        path=a.capture/Path(row['path']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        return np.load(path,allow_pickle=False)
    final=load(last('post_rk'));update=load(last('update'))
    pd_rows=[r for r in rows if r['kind']=='pd_preload']
    pd=load(last('pd_preload')) if pd_rows else None
    out=dict(scope='Actual full-grid CPU FAST0 operands; no GPU or CPU-WRF trajectory claim',
             reference_control='NTU62: zero events, so these are scheme-specific',
             column_relative_decision_limit=1e-12,fields={})
    expected={'qi':93,'Ni':127}
    if a.counts_from is not None:
        census=json.loads(a.counts_from.read_text())['steps'][-1]['f2']['f2_counts']
        expected={'qi':census[1][3][0],'Ni':census[1][6][0]}
    for name,expected_count in expected.items():
        q=final[name+'_result'];bad=q<0;assert np.isfinite(q).all()
        assert int(bad.sum())==expected_count,(name,int(bad.sum()))
        column_max=np.max(np.abs(q),axis=0)
        ratio=np.abs(q[bad])/np.broadcast_to(column_max,q.shape)[bad]
        origin=update[name+'_origin'];tendency=update[name+'_tendency']
        m1,m2,dt=update['mass_old'],update['mass_new'],update['dt']
        # The port's existing unmodified scalar update. This reconstruction is
        # diagnostic, not an independent fidelity reference.
        reconstructed=(m1*origin+dt*tendency)*(1/m2)
        stage=update[name+'_result']
        wide=lambda x:np.asarray(x,dtype=np.longdouble)
        ideal=(wide(m1)*wide(origin)+wide(dt)*wide(tendency))/wide(m2)
        # Fixed independent original REAL4 input-ULP controls. Never enlarge
        # this enclosure or the manager's column-relative limit to fit a port.
        controls=[]
        for direction in (-np.inf,np.inf):
            t=np.nextafter(tendency.astype(np.float32),np.float32(direction)).astype(np.longdouble)
            controls.append((wide(m1)*wide(origin)+wide(dt)*t)/wide(m2))
        low=np.minimum(*controls);high=np.maximum(*controls)
        enclosed=(wide(stage)>=low)&(wide(stage)<=high)
        out['fields'][name]=dict(count=int(bad.sum()),minimum=float(q.min()),
            max_column_relative_negative=float(ratio.max()) if ratio.size else 0.,
            column_relative_gate=bool(np.all(ratio<=1e-12)),
            negative_indices=np.flatnonzero(bad).tolist(),
            column_max_at_negative_min=float(np.broadcast_to(column_max,q.shape)[bad].min()) if ratio.size else None,
            column_max_at_negative_max=float(np.broadcast_to(column_max,q.shape)[bad].max()) if ratio.size else None,
            post_rk_matches_final_update_BYTE=q.tobytes()==stage.tobytes(),
            negative_cells_match_final_update_BYTE=q[bad].tobytes()==stage[bad].tobytes(),
            negative_sets_match_final_update=bool(np.array_equal(bad,stage<0)),
            final_update_reconstruction_BYTE=stage.tobytes()==reconstructed.tobytes(),
            preload_negative=None if pd is None else int((pd[name+'_result']<0).sum()),
            preload_executed=pd is not None,
            origin_negative=int((origin<0).sum()),
            final_update_negative=int((stage<0).sum()),
            ideal_longdouble_negative=int((ideal<0).sum()),
            negative_enclosed_by_original_REAL4_tendency_ULP=bool(enclosed[stage<0].all()),
            mass_weights_positive=bool((m1>0).all() and (m2>0).all()),
            input_dtype=str(origin.dtype),tendency_dtype=str(tendency.dtype),
            capture_SHA256={kind:last(kind)['sha256'] for kind in ('post_rk','update')+
                           (('pd_preload',) if pd is not None else ())})
    out['column_relative_gate']=all(r['column_relative_gate'] for r in out['fields'].values())
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({n:{k:v for k,v in d.items() if k not in ('negative_indices','capture_SHA256')}
                      for n,d in out['fields'].items()},indent=2))


if __name__=='__main__':main()
