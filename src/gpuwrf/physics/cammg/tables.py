"""Lazy original CAM-MG initialization namespaces (host NumPy only).

The raw namespace is cached, never a JAX array/tracer. The selected provider is registered with runtime.aot_baked_tables before
dispatch; replacements invalidate cheap/AOT identities.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def _raw():
    with np.load(Path(__file__).with_name('data')/'initialization.npz',allow_pickle=False) as z:
        result={k:z[k].copy() for k in z.files}
    for value in result.values():value.setflags(write=False)
    return result


def get_globals(nz):
    from gpuwrf.physics.bl_camuw import EPSILO,LATVAP,LATICE,RH2O,CPAIR,TMELT,GRAVIT,RAIR
    modules={};names={'ndrop':'ndrop','aero':'microp_aero','modal':'modal_aero_data'}
    for key,value in _raw().items():
        namespace,name=key.split('.')
        modules.setdefault(names[namespace],{})[name]=value
    modal=modules['modal_aero_data']
    modal={**modal,**{k[:-3]:v for k,v in modal.items() if k.endswith('_mp')}}
    qmin=np.zeros(12,dtype=np.float64);qmin[0]=1.e-12
    plan=tuple((int(modal['numptr_amode'][n])-1,
                tuple(int(modal['lmassptr_amode'][i,n])-1 for i in range(int(modal['nspec_amode'][n]))),
                int(modal['lspectype_amode'][0,n])-1) for n in range(3))
    return {**modal,'pcols':1,'pver':nz,'pverp':nz+1,'pcnst':12,'qmin':qmin,
            'epsqs':np.float64(EPSILO),'epsilo':np.float64(EPSILO),'tmelt':np.float64(TMELT),
            'hlatv':np.float64(LATVAP),'hlatf':np.float64(LATICE),'rgasv':np.float64(RH2O),'cp':np.float64(CPAIR),
            'latvap':np.float64(LATVAP),'latice':np.float64(LATICE),'cpair':np.float64(CPAIR),
            'rair':np.float64(RAIR),'rh2o':np.float64(RH2O),'gravit':np.float64(GRAVIT),
            'r_universal':np.float64(6.02214e26*1.38065e-23),'mwh2o':np.float64(18.016),'rhoh2o':np.float64(1000),
            'tmin':np.float64(173.16),'__modules':modules,'_modal_plan':plan}
