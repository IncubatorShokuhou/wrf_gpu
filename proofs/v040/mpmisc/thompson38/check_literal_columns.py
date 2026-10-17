"""Compare literal production mp38 code objects with pristine REAL/no-FMA."""
import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np
from literal_reference import executor, LiteralState
from check_columns import parse, surface_projection


def main(reference,out,cases,stages=None):
    refs=[parse(reference/f'case{case}.txt') for case in cases]
    fields={'qv':'QV','qc':'QC','qr':'QR','qi':'QI','qs':'QS','qg':'QG','Ni':'NI','Nr':'NR','Nc':'NC','Ng':'NG','qvolg':'QB','nwfa':'NWFA','nifa':'NIFA','T':'T'}
    cols={name:np.asarray([r['columns'][name] for r in refs],np.float32) for name in refs[0]['columns']}
    values={name:cols[code+'_IN'] for name,code in fields.items()}
    rho=np.float32(.622)*cols['P']/(np.float32(287.04)*values['T']*(values['qv']+np.float32(.622)))
    state=LiteralState(**values,p=cols['P'],rho=rho,dz=cols['DELZ'],w=cols.get('W_IN',np.zeros_like(rho)))
    dt=refs[0]['scalars']['DT']
    start=time.monotonic()
    captured={}
    def profile(frame,event,arg):
        if event=='return' and frame.f_code.co_name in ('_wrf_mixed_phase_sources','aerosol_tendencies','sinks','_eff_aero','_saturation_adjustment_aero','_rain_evaporation_aero','freezing','_working_moments','_fall_speeds','_sedimentation','_render','_finish_aero'):
            prefix=frame.f_code.co_name
            if prefix=='_eff_aero': prefix += '_'+str(frame.f_locals['species'])+'_'+str(frame.f_locals['da'])
            if prefix=='sinks': prefix += '_'+str(frame.f_locals['radius'])
            for name,value in frame.f_locals.items():
                if isinstance(value,np.ndarray) and value.size<=240:
                    captured[prefix+'_'+name]=np.asarray(value).copy()
                elif isinstance(value,dict):
                    for key,array in value.items():
                        if isinstance(array,np.ndarray) and array.size<=240:
                            captured[prefix+'_'+name+'_'+key]=np.asarray(array).copy()
                elif isinstance(value,LiteralState):
                    for key in value.__slots__:
                        captured[prefix+'_'+name+'_'+key]=np.asarray(getattr(value,key)).copy()
            if frame.f_code.co_name=='sinks':
                for i,value in enumerate(arg): captured[prefix+'_returned'+str(i)]=np.asarray(value).copy()
            if frame.f_code.co_name=='_working_moments':
                for key,value in arg.items():
                    if isinstance(value,np.ndarray): captured[prefix+'_returned_'+key]=np.asarray(value).copy()
    with executor() as body:
        if stages is not None: sys.setprofile(profile)
        try: actual,precip=body(state,dt)
        finally: sys.setprofile(None)
    if stages is not None: np.savez_compressed(stages,**captured)
    result=dict(scope='Literal equation code objects, host REAL/no-FMA, independent pristine Fortran reference',cases=cases,dt=dt,wall_s=time.monotonic()-start,fields={})
    for name,code in fields.items():
        a,b=np.asarray(getattr(actual,name)),cols[code+'_OUT']
        if name=='T':
            result['inner_temperature']=a.tolist()
            a=(a/cols['PII'])*cols['PII']
        diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
        result['fields'][name]=dict(finite=bool(np.isfinite(a).all()),byte_equal=bool(np.array_equal(a,b)),matching=int(np.count_nonzero(a==b)),total=int(a.size),max_abs=float(diff.max()),max_ulp=float((diff/np.maximum(np.abs(np.spacing(b)),1.e-45)).max()),actual=a.tolist())
    result['precipitation_components']={name:np.asarray(value).tolist() for name,value in precip.items()}
    rain=((np.asarray(precip['rain'],np.float32)+np.asarray(precip['snow'],np.float32))+np.asarray(precip['graupel'],np.float32))+np.asarray(precip['ice'],np.float32)
    expected=np.asarray([r['scalars']['RAINNCV'] for r in refs],np.float32)
    result['rainncv']=dict(actual=rain.tolist(),reference=expected.tolist(),max_abs=float(np.max(np.abs(rain-expected))))
    result['public_surface']=surface_projection(precip,refs)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in r.items() if n!='actual'} for k,r in result['fields'].items()},indent=2))
    print('RAINNCV',result['rainncv'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--cases',default='1,2,3,4,5,6')
    parser.add_argument('--stages',type=Path)
    a=parser.parse_args()
    main(a.reference,a.out,tuple(map(int,a.cases.split(','))),a.stages)
