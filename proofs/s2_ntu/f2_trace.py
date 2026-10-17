"""CPU-only, read-only operands for the unmatched post-RK qi/Ni events.

First samples are not magnitude bounds (E213). Capture full-grid actual
source/update operands so the original WRF scalar equation can be replayed.
No operand, function result, boundary, cadence or mathematical call changes.
"""
import functools
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(a):
    a=np.asarray(a);negative=a<0
    return dict(dtype=str(a.dtype),shape=list(a.shape),negative=int(negative.sum()),
        nonfinite=int((~np.isfinite(a)).sum()),minimum=float(a.min()),
        maximum=float(a.max()),max_abs=float(np.abs(a).max()),
        sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        negative_indices=np.flatnonzero(negative).tolist())


class F2Trace:
    fields=('qi','Ni')

    def __init__(self,module,out):
        self.module=module;self.out=Path(out);self.out.mkdir(exist_ok=True)
        self.step=0;self.calls={};self.rows=[]

    def save(self,kind,arrays,extra):
        number=self.calls.get((self.step,kind),0)+1
        self.calls[self.step,kind]=number
        path=self.out/f'step{self.step}_{kind}{number}.npz'
        np.savez_compressed(path,**arrays)
        self.rows.append(dict(step=self.step,kind=kind,call=number,**extra,
            arrays={n:stats(a) for n,a in arrays.items() if np.asarray(a).ndim==3},
            path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (self.out/'manifest.json').write_text(json.dumps(self.rows,indent=2)+'\n')
        print('F2_OPERANDS',self.step,kind,number,
              {n:stats(arrays[n])['negative'] for n in arrays if n.endswith('_result')},flush=True)

    def __enter__(self):
        module=self.module;self.originals={}
        def patch(name,recorder):
            original=getattr(module,name);self.originals[name]=original
            @functools.wraps(original)
            def observed(*args,**kwargs):
                result=original(*args,**kwargs)
                recorder(args,kwargs,result)
                return result
            setattr(module,name,observed)
        def pd(args,kwargs,result):
            origin,species,sources,dt,metrics=args
            mass=(np.asarray(metrics.c1h)[:,None,None]*np.asarray(origin.mu_total)[None]+
                  np.asarray(metrics.c2h)[:,None,None])
            arrays={'mass_old':mass,'dt':np.asarray(dt)}
            for name in self.fields:
                if name not in species:continue
                arrays[name+'_origin']=np.asarray(getattr(origin,name))
                arrays[name+'_result']=np.asarray(getattr(result,name))
                for i,source in enumerate(sources):
                    if source is not None and name in source:
                        arrays[f'{name}_source{i}']=np.asarray(source[name])
            self.save('pd_preload',arrays,{})
        def update(args,kwargs,result):
            state,origin=args;metrics=kwargs['metrics'];dt=kwargs['dt_rk']
            mass=lambda s:(np.asarray(metrics.c1h)[:,None,None]*np.asarray(s.mu_total)[None]+
                           np.asarray(metrics.c2h)[:,None,None])
            arrays={'mass_old':mass(origin),'mass_new':mass(state),'dt':np.asarray(dt),
                    'c1':np.asarray(metrics.c1h),'c2':np.asarray(metrics.c2h),
                    'mu_old':np.asarray(origin.mu_total),'mu_new':np.asarray(state.mu_total),
                    'msfty':np.asarray(metrics.msfty)}
            for name,tend in zip(kwargs['species'],kwargs['q_tendencies'],strict=True):
                if name not in self.fields:continue
                arrays[name+'_origin']=np.asarray(getattr(origin,name))
                arrays[name+'_tendency']=np.asarray(tend)
                arrays[name+'_result']=np.asarray(getattr(result,name))
            self.save('update',arrays,{})
        def finished(args,kwargs,result):
            self.save('post_rk',{n+'_result':np.asarray(getattr(result.state,n))
                                for n in self.fields},{})
        patch('_rk_update_scalar_pd',pd)
        patch('_apply_moisture_large_step',update)
        patch('_rk_scan_step',finished)
        return self

    def __exit__(self,*args):
        for name,original in self.originals.items():setattr(self.module,name,original)
