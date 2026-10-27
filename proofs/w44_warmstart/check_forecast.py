"""CPU qualification receipts for the native three-nest forecast, once per bytes."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset, chartostring


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def native_inventory(cache):
    return {str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*'))
            if p.is_file() and (p.name.endswith('-cache') or p.suffix in ('.xlaexec','.cubin'))}


def inspect_forecast(run):
    proof=json.loads((run/'proofs/nested_pipeline_run.json').read_text())
    assert proof['max_dom']==3 and proof['hours']==3
    assert proof['all_domains_finite'] and proof['all_outputs_present']
    rows=[];cloud={}
    for path in sorted((run/'wrfout').glob('wrfout_d0*')):
        with Dataset(path) as ds:
            ds.set_auto_maskandscale(False)
            times=chartostring(ds['Times'][:]).tolist();assert len(times)==1
            bad={}
            for name,v in ds.variables.items():
                if np.issubdtype(v.dtype,np.number):
                    count=int(np.count_nonzero(~np.isfinite(v[:])))
                    if count:bad[name]=count
            assert not bad,(path,bad)
            if times[0]=='2026-05-04_00:00:00':
                get=lambda name:np.asarray(ds[name][0],np.float64)
                total=get('MU')+get('MUB')
                weight=-get('DNW')[:,None,None]*(get('C1H')[:,None,None]*total[None]+get('C2H')[:,None,None])/9.81
                assert np.isfinite(weight).all()and(weight>0).all()
                cwp=np.sum(get('QCLOUD')*weight,axis=0)
                assert np.isfinite(cwp).all()and cwp.max()>0,(path,float(cwp.max()))
                cloud[path.name.split('_')[1]]={'max_kg_m2':float(cwp.max()),'positive_columns':int(np.count_nonzero(cwp>0)),
                    'rule':'sum(QCLOUD * dry-layer-mass); -DNW*(C1H*(MUB+MU)+C2H)/g'}
        rows.append({'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'times':times})
    assert len(rows)==12 and set(cloud)=={'d01','d02','d03'}
    census_path=run/'wrfout/census.json'
    assert census_path.exists(),'requested on-device census receipt missing'
    census=json.loads(census_path.read_text())
    assert set(census['domains'])=={'d01','d02','d03'}
    for domain,value in census['domains'].items():
        assert all(events['nonfinite']==0 for events in value['guards'].values()),domain
        assert 'f2'in value,domain
        assert not any(event['event']=='nonfinite' for event in value['f2']['events']),domain
    inherited={}
    for domain in ('d01','d02','d03'):
        with Dataset(Path(proof['input_dir'])/('wrfinput_'+domain))as ds:
            marker=json.loads(ds.GPUWRF_ATMOSPHERE_CYCLE)
            inherited[domain]={'source_sha256':marker['source_sha256'],
                               'extrema_and_counts':marker['consistency']['inherited_source']}
    return {'passed':True,'files':rows,'cloud_at_tau0':cloud,'all_numeric_variables_finite':True,
            'on_device_census':census,'census_sha256':sha(census_path),
            'inherited_source':inherited,
            'f2_negative_summary':json.loads((run/'wrfout/f2_negative_summary.json').read_text()),
            'nonzero_event_policy':'reported in full; inherited source extrema are in input receipts, any further justification remains explicit',
            'scope':'3h cycling qualification; D6 vs private guarded CPU-WRF is separate'}


def main():
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    mode,run,output=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
    if mode=='launch-smoke':
        model=json.loads((run/'PREFLIGHT.json').read_text())
        gate=json.loads(Path(sys.argv[4]).read_text())
        assert model['platform']=='cpu'and model['model_body_reached']
        assert sum(r['columns']for r in model['records'])==4
        assert all(r['finite']and r['background_QKE_QSQ_exact_zero']for r in model['records'])
        assert gate['e41_gate']=='PASS'and gate['n_entries']>0
        report={'passed':True,'mode':mode,'model_mini':model,'E41_gate':gate['e41_gate'],
                'scope':'CPU mini model and real retained CUDA artifact scan; NOT forecast qualification'}
    elif mode=='cold':
        report=inspect_forecast(run)
        report['native_cache_inventory']=native_inventory(Path(sys.argv[4]))
    else:
        assert mode=='warm'
        cold=json.loads(Path(sys.argv[4]).read_text())
        measurement=json.loads((run/'MEASUREMENT.json').read_text())
        assert measurement['platform']=='gpu'and measurement['compile_count']==0
        rows=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted((run/'wrfout').glob('wrfout_d0*'))]
        assert len(rows)==12
        expected={r['file']:(r['sha256'],r['bytes'])for r in cold['files']}
        assert {r['file']:(r['sha256'],r['bytes'])for r in rows}==expected,'cold/warm file bytes changed'
        inventory=native_inventory(Path(sys.argv[5]))
        assert inventory==cold['native_cache_inventory'],'native executable cache changed during warm run'
        report={'passed':True,'files':rows,'compile_count':0,'file_bytes_exact':True,
                'cloud_and_finite':'carried from identical cold files; no duplicate numerical scan',
                'native_executables_exact':True,
                'scope':'GPU-vs-GPU determinism/cache regression, not WRF fidelity'}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'output':str(output)}))


if __name__=='__main__':main()
