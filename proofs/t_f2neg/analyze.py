"""Fail-closed WRF observer reader and full-field E26 qualification."""
import hashlib,json,os,pathlib,sys
import numpy as np
assert os.environ['JAX_PLATFORMS']=='cpu'
from gpuwrf.io.netcdf_lock import Dataset
ROOT=pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/t-f2neg')
FIELDS=('qv','qc','qr','qi','qs','qg','Ni','Nr','Nc','Nn')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read_field(path, target=None):
    matched=None
    counts=[]; values=[]; regions=[]; minimum=None; first=None; nonfinite=0
    with path.open('rb') as f:
        while raw:=f.read(12):
            assert len(raw)==12
            step,n,nf=np.frombuffer(raw,dtype='>i4').tolist()
            assert step==len(counts)+1 and 0<=n<=42*42*44 and nf>=0,(step,n,nf)
            raw=f.read(16*n)
            assert len(raw)==16*n
            data=np.frombuffer(raw,dtype='>i4').reshape(n,4)
            vals=data[:,3].copy().view('>f4')
            assert np.isfinite(vals).all() and (vals<0).all()
            if n:
                assert ((data[:,:3]>=0)&(data[:,:3]<[42,42,44])).all()
                ring=(data[:,0]==0)|(data[:,0]==41)|(data[:,1]==0)|(data[:,1]==41)
                # Producer traverses k,j,i; first argmin matches C-order kji tie rule.
                k=int(np.argmin(vals)); value=float(vals[k])
                event=dict(step=step,index_kji=data[k,[2,1,0]].tolist(),value=value)
                if minimum is None or value<minimum['value']: minimum=event
                if first is None: first=dict(step=step,index_kji=data[0,[2,1,0]].tolist(),value=float(vals[0]))
                values.append(vals.astype('f4'));regions.append(ring)
                r=int(ring.sum())
            else:r=0
            if target is not None and step==target['step']:
                kji=target['index_kji']; xyz=[kji[2],kji[1],kji[0]]
                hits=np.flatnonzero(np.all(data[:,:3]==xyz,axis=1))
                assert len(hits)<=1
                matched=dict(step=step,index_kji=kji,negative_count=n,ring0_count=r,interior_count=n-r,
                    value_at_gpu_minimum=float(vals[hits[0]]) if len(hits) else None,
                    coordinate_status='NEGATIVE' if len(hits) else 'NOT_NEGATIVE',
                    step_minimum=float(vals.min()) if n else None)
            counts.append([step,n,r,n-r,nf]);nonfinite+=nf
    assert len(counts)==4800, len(counts)
    arr=np.concatenate(values) if values else np.empty(0,'f4')
    ring=np.concatenate(regions) if regions else np.empty(0,bool)
    row=dict(present_observations=len(counts),negative_count=len(arr),ring0_count=int(ring.sum()),interior_count=int((~ring).sum()),nonfinite_count=nonfinite,minimum_negative=minimum,first_negative=first,raw_sha256=sha(path))
    probs=[0,.01,.05,.25,.5,.75,.95,.99,1]
    row['matched_GPU_minimum_step']=matched
    row['negative_value_quantiles']={label:{str(p):float(v) for p,v in zip(probs,np.quantile(x,probs))} if len(x) else None for label,x in [('all',arr),('ring0',arr[ring]),('interior',arr[~ring])]}
    return row,counts

def qualify(arm,reference):
    rows=[]
    files=sorted(arm.glob('wrfout_d01*'))
    assert len(files)==25
    for p in files:
        ref=reference/p.name
        assert ref.exists(),ref
        with Dataset(p) as a,Dataset(ref) as b:
            a.set_auto_mask(False);b.set_auto_mask(False)
            assert set(a.variables)==set(b.variables)
            differences=[];nonfinite=[];hashes={}
            for name in a.variables:
                av=np.asarray(a.variables[name][:]);bv=np.asarray(b.variables[name][:])
                assert av.shape==bv.shape and av.dtype==bv.dtype,(p.name,name)
                aa,bb=av.tobytes(),bv.tobytes()
                if aa!=bb:differences.append(name)
                if av.dtype.kind=='f' and (not np.isfinite(av).all() or not np.isfinite(bv).all()):nonfinite.append(name)
                hashes[name]=[hashlib.sha256(aa).hexdigest(),hashlib.sha256(bb).hexdigest()]
            rows.append(dict(file=p.name,field_count=len(hashes),different_fields=differences,nonfinite_fields=nonfinite,sha256=hashes))
    return dict(pass_byte=all(not r['different_fields'] and not r['nonfinite_fields'] for r in rows),paired_frames=len(rows),paired_fields=sum(r['field_count'] for r in rows),rows=rows)

if __name__=='__main__':
    name=sys.argv[1];arm=ROOT/name
    run=json.loads((arm/'RUN.json').read_text());assert run['rc']==0
    reference=pathlib.Path(run['reference'])
    if name=='release_default':
        binding=json.loads((ROOT/'DEFAULT_E26_REFERENCE.json').read_text())
        assert binding['namelist_values_equal']
        reference=pathlib.Path(binding['serial_reference'])
    receipt=qualify(arm,reference)
    receipt['reference']=str(reference)
    (arm/'E26.json').write_text(json.dumps(receipt,indent=2)+'\n')
    rows=[];steps={}
    gpu=json.loads((pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/v-run')/name/'gpu/f2_negative_summary.json').read_text())['domains']['d01']['records']
    for i,field in enumerate(FIELDS,1):
        p=arm/f'f2_field_{i:02}.bin'
        if not p.exists():continue
        g=next(r for r in gpu if r['field']==field and r['phase']=='post_rk')
        row,count=read_field(p,g['minimum_negative']); row['field']=field
        row['GPU_post_rk']=g
        rows.append(row);steps[field]=count
    result=dict(case=name,E26_pass=receipt['pass_byte'],phase='post_rk',records=rows,interpretation='Original-WRF trajectory vs GPU trajectory, same initial/boundary/namelist inputs. Not identical per-step operands. GPU receipt has no quantiles; only counts, regions and minima are compared.')
    (arm/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (arm/'COUNTS_BY_STEP.json').write_text(json.dumps(steps,separators=(',',':'))+'\n')
    print(json.dumps({k:receipt[k] for k in ('pass_byte','paired_frames','paired_fields')}))
    for r in rows:print(r['field'],r['negative_count'],r['interior_count'],r['minimum_negative'],'GPU',r['GPU_post_rk']['minimum_negative'])
    assert receipt['pass_byte'],'Observer E26 failed; census not qualified'
    assert all(r['nonfinite_count']==0 for r in rows),'Nonfinite post-RK observation'
