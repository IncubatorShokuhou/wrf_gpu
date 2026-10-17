"""Compare staged literal level processes to original pre-sedimentation columns."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,traceback
import numpy as np

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(tree/'src'))
from gpuwrf.physics.jensen_ishmael.processes import processes
folder=tree/'proofs/s2_mpmore/jensen55'
spec=importlib.util.spec_from_file_location('compiler_libm',folder/'libm_numpy.py')
libm=importlib.util.module_from_spec(spec);spec.loader.exec_module(libm)
work=out.parent/'level_probe_cache';work.mkdir(exist_ok=False)
backend=libm.aggregation_backend(work)
stages=json.loads((folder/'column_stages/manifest.json').read_text())
tables_manifest=json.loads((folder/'fixtures/manifest.json').read_text())
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F').read_text()


def original_tables(mode):
    R=np.float32 if mode=='fp32' else np.float64;tables={}
    import re
    text=re.search(r'igrdata\s*=\s*&\s*\(/(.*?)/\)',source,re.S|re.I)[1]
    text=' '.join(line.split('!')[0].strip().strip('&') for line in text.splitlines())
    tables['igrdata']=np.asarray([float(a) for a in text.split(',')],R)
    assert len(tables['igrdata'])==60
    for field,file,nchan in [('itab','ishmael-qi-qc.bin',2),('itabr','ishmael-qi-qr.bin',6)]:
        path=Path('<USER_HOME>/src/wrf_pristine/WRF/run')/file
        assert hashlib.sha256(path.read_bytes()).hexdigest()==tables_manifest['tables'][file]
        dtype=np.dtype([('head','>i4'),('data','>f4',(nchan,)),('tail','>i4')])
        raw=np.fromfile(path,dtype)
        assert len(raw)==51*51*51*11 and np.all(raw['head']==4*nchan) and np.all(raw['tail']==4*nchan)
        tables[field]=raw['data'].astype(R).reshape((51,51,51,11,nchan))
    with np.load(folder/'stages'/(mode+'.npz')) as data:
        for field in ('gamma_table','coltab','coltabn'):tables['gamma_tab' if field=='gamma_table' else field]=data[field].copy()
    return tables


rows=[]
checkpoint_mode=len(sys.argv)>3 and sys.argv[3].startswith('checkpoints')
if checkpoint_mode:
    checkpoints=json.loads((folder/'process_checkpoints/manifest.json').read_text())
for mode in ('fp32','fp64'):
    tables=original_tables(mode)
    if checkpoint_mode:
        with np.load(folder/'process_checkpoints'/(mode+'.npz')) as data:
            checkpoint_indices=data['indices'];checkpoint_values=data['values']
    cases=[4,6] if len(sys.argv)>3 and sys.argv[3] in ('smoke','checkpoints-smoke') else range(1,18)
    for case in cases:
        fields=[]
        for stage in (0,1):
            p=folder/'column_stages'/f'{mode}_{case}_{stage}.npz'
            record=next(r for r in stages['rows'] if (r['mode'],r['case'],r['stage'])==(mode,case,stage))
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
            with np.load(p) as data:fields.append({k:data[k].copy() for k in data.files})
        inputs,reference=fields;nz=len(inputs['qv'])
        inputs['qipre1d']=np.broadcast_to(inputs['qipre1d'],(nz,3)).copy()
        try:
            observations=[]
            def observer(stage,values,mask):
                active=np.broadcast_to(mask,(nz,))
                blocks=[np.column_stack([np.broadcast_to(values[n],(nz,)) for n in checkpoints['scalar_names']])]
                blocks.extend(np.broadcast_to(values[n],(nz,3)) for n in checkpoints['category_names'])
                observations.append((stage,np.flatnonzero(active)+1,np.concatenate(blocks,axis=1)[active]))
            with np.errstate(all='ignore'):
                got=processes(inputs,tables,xp=backend,observer=observer if checkpoint_mode else None)
            differences={}
            for name,expected in reference.items():
                if name in ('it','dt','pres_e','dzmic'):continue
                value=got[name]
                if name=='sedi':value=np.int32(np.any(value))
                expected=np.broadcast_to(expected,np.shape(value))
                if not np.isfinite(value).all() or np.asarray(value).tobytes()!=expected.tobytes():
                    differences[name]=dict(not_BYTE=int(np.count_nonzero(value!=expected)),
                                           nonfinite=int(np.count_nonzero(~np.isfinite(value))),
                                           max_abs=float(np.max(np.abs(value-expected))))
            row=dict(mode=mode,case=case,differences=differences)
            if checkpoint_mode:
                names=checkpoints['scalar_names']+[n+str(i+1) for n in checkpoints['category_names'] for i in range(3)]
                diffs=[]
                for stage,levels,values in observations:
                    selected=checkpoint_indices[(checkpoint_indices[:,0]==case)&(checkpoint_indices[:,1]==stage)]
                    expected=checkpoint_values[(checkpoint_indices[:,0]==case)&(checkpoint_indices[:,1]==stage)]
                    assert selected[:,2].tobytes()==levels.astype(np.int32).tobytes(),(stage,'Original active levels differ')
                    bad=values!=expected
                    byfield={}
                    for i,name in enumerate(names):
                        idx=np.flatnonzero(bad[:,i])
                        if len(idx):byfield[name]=dict(count=len(idx),levels=levels[idx][:5].tolist(),
                                                      got=values[idx[:5],i].tolist(),ref=expected[idx[:5],i].tolist())
                    diffs.append(dict(stage=stage,fields=byfield))
                row['checkpoint_differences']=diffs
            rows.append(row)
        except Exception:
            rows.append(dict(mode=mode,case=case,error=traceback.format_exc()))
        print(rows[-1],flush=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
out.write_text(json.dumps(dict(source_commit=commit,scope='Literal original preSEDI boundary;not fullscheme qualification',rows=rows),indent=2)+'\n')
raise SystemExit(int(any(r.get('error') or r.get('differences') for r in rows)))
