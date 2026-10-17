"""Apply predeclared flux-input heating bound to saved compiled SWR93 results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from swr_heating_bound import heating_input_bound


def qualify(source,candidate):
    flux_names=('fswc','ufswc','dfswc','fswl','ufswl','dfswl','gdfvb','gdfnb','gdfvd','gdfnd')
    results={}
    for key in flux_names:
        value=candidate[key];reference=source['out_'+key]
        if value.dtype!=reference.dtype or value.dtype!=np.float32 or value.shape!=reference.shape:
            raise ValueError('Invalid flux dtype/geometry')
        if not np.isfinite(value).all() or not np.isfinite(reference).all():raise ValueError('Non-finite flux')
        err=np.abs(value.astype(np.float64)-reference.astype(np.float64))
        bound=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
        results[key]=dict(pass_=bool(np.all(err<=bound)),outside=int(np.count_nonzero(err>bound)))
    if not all(v['pass_'] for v in results.values()):
        return dict(pass_=False,flux_prerequisite=False,fields=results)
    for heat,flux in (('hswc','fswc'),('hswl','fswl')):
        value=candidate[heat];reference=source['out_'+heat]
        if value.dtype!=np.float32 or value.shape!=reference.shape or not np.isfinite(value).all():
            raise ValueError('Invalid heating')
        bound=heating_input_bound(source['out_'+flux],source['press'],reference,
                                  source['haf'][:,None],source['radcon'][:,None])
        error=np.abs(value.astype(np.float64)-reference.astype(np.float64))
        results[heat]=dict(pass_=bool(np.all(error<=bound)),outside=int(np.count_nonzero(error>bound)),
                           max_error=float(error.max()),max_bound=float(bound.max()),max_ratio=float(np.max(error/bound)))
    return dict(pass_=all(v['pass_'] for v in results.values()),flux_prerequisite=True,fields=results)


def main(fixture,compiled,output):
    with np.load(fixture,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    with np.load(compiled,allow_pickle=False) as f:candidate={k:f[k].copy() for k in f.files}
    assert np.array_equal(candidate['rows'],np.arange(40,dtype=np.int32))
    result=qualify(source,candidate)
    # Genuine numerical mutants must fail the actual gate, not its setup.
    mutants={}
    zero=dict(candidate,hswc=np.zeros_like(candidate['hswc']))
    mutants['zero_heating_killed']=not qualify(source,zero)['pass_']
    offset=dict(candidate,fswc=candidate['fswc']+np.float32(1000.))
    mutants['one_Wm2_flux_offset_killed']=not qualify(source,offset)['pass_']
    for target in ('candidate','source'):
        c={k:v.copy() for k,v in candidate.items()};s={k:v.copy() for k,v in source.items()}
        (c['hswc'] if target=='candidate' else s['out_hswc']).flat[0]=np.nan
        try:qualify(s,c)
        except ValueError:mutants[target+'_NaN_rejected']=True
        else:mutants[target+'_NaN_rejected']=False
    assert all(mutants.values()),mutants
    output.write_text(json.dumps(dict(result=result,mutants=mutants,
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fixture,compiled,Path(__file__).with_name('swr_heating_bound.py'))},
        scope='CPU source-input ULP gate; original flux gate unchanged; strict heating excess retained FG35; no GPU/forecast/D6 claim'),indent=2)+'\n')
    print(json.dumps(dict(result=result,mutants=mutants),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('fixture','compiled','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.fixture,a.compiled,a.output)
