"""Original full CAM-MG modal aerosol stage on actual caller operands."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',default='1,2,3,4,5,16,17,18');args=ap.parse_args()
    sys.path.insert(0,str(args.tree/'src'))
    from gpuwrf.physics.cammg import aerosol
    from gpuwrf.physics.bl_camuw import EPSILO,LATVAP,LATICE,RH2O,CPAIR,TMELT,GRAVIT,RAIR
    from compiler_math import compiler_math as xp
    nz=24;shapes=dict(pcols=1,pver=nz,pverp=nz+1,pcnst=12,ntot_amode=3)
    modules={};names={'ndrop':'ndrop','aero':'microp_aero','modal':'modal_aero_data'}
    with np.load(args.tree/'proofs/s2_cammg/fixtures/aerosol_constants.npz') as z:
        for k in z.files:
            ns,name=k.split('.');modules.setdefault(names[ns],{})[name]=z[k]
    modal=modules['modal_aero_data']
    modal={**modal,**{k[:-3]:v for k,v in modal.items() if k.endswith('_mp')}}
    globals={**shapes,**modal,'epsqs':np.float64(EPSILO),'epsilo':np.float64(EPSILO),'tmelt':np.float64(TMELT),
             'hlatv':np.float64(LATVAP),'hlatf':np.float64(LATICE),'rgasv':np.float64(RH2O),'cp':np.float64(CPAIR),
             'latvap':np.float64(LATVAP),'latice':np.float64(LATICE),'cpair':np.float64(CPAIR),
             'rair':np.float64(RAIR),'rh2o':np.float64(RH2O),'gravit':np.float64(GRAVIT),
             'r_universal':np.float64(6.02214e26*1.38065e-23),'mwh2o':np.float64(18.016),'rhoh2o':np.float64(1000),
             'tmin':np.float64(173.16),'__modules':modules}
    with np.load(args.tree/'proofs/s2_cammg/fixtures/stages.npz') as z:bank={k:z[k] for k in z.files}
    mapping=dict(t='state1_t',qn='qv',p='state1_pmid',pdel='state1_pdel',cldn='ast',
                 liqcldf='alst_mic',icecldf='aist_mic',pint='state1_pint',rpdel='state1_rpdel',
                 zm='state1_zm',omega='state1_omega',qaer='state1_q',qaertend='ptend_loc_q')
    outmap=dict(qaertend='ptend_loc_q')
    receipt={'rows':[],'BYTE':True};start=time.monotonic()
    try:
        for case in map(int,args.cases.split(',')):
            for step in (1,2):
                prefix=f'c{case:02d}_s{step}';data={}
                q=bank[prefix+'_aero_in_state1_q'].reshape((1,nz,12),order='F')
                for name in aerosol.ARGUMENTS['microp_aero_ts']:
                    spec=aerosol.VARIABLES['microp_aero_ts'][name]
                    shape=tuple(eval(d,{},shapes) for d in spec['dims'])
                    source=mapping.get(name,name)
                    if name=='lchnk':value=case
                    elif name=='ncol':value=1
                    elif name=='deltatin':value=np.float64(18)
                    elif name=='ttend' or spec['intent']=='out':value=np.zeros(shape)
                    elif source=='qv':value=q[:,:,0]
                    else:value=bank[prefix+'_aero_in_'+source].reshape(shape,order='F')
                    data[name]=value
                with np.errstate(all='ignore'):
                    outputs=aerosol.microp_aero_ts(**data,_globals=globals,xp=xp)
                row={'case':case,'step':step,'BYTE':True,'fields':{}}
                for name,value in zip(aerosol.OUTPUTS['microp_aero_ts'],outputs,strict=True):
                    if name=='_fatal':assert value==0,('originalfatal',case,step);continue
                    ref=bank[prefix+'_aero_out_'+outmap.get(name,name)]
                    got=np.asarray(value).reshape(-1,order='F')
                    assert got.shape==ref.shape and got.dtype==ref.dtype,(name,got.shape,ref.shape,got.dtype,ref.dtype)
                    assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)),name
                    byte=got.tobytes()==ref.tobytes();row['BYTE']&=byte
                    if not byte:row['fields'][name]={'different':int(np.count_nonzero(got!=ref)),
                        'max_abs':float(np.max(np.abs(got-ref))),'first':np.argwhere(got!=ref)[:8].tolist()}
                receipt['rows'].append(row);receipt['BYTE']&=row['BYTE']
        receipt['seconds']=time.monotonic()-start
    finally:args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));assert receipt['BYTE'],'original aerosol BYTE gate failed'


if __name__=='__main__':main()
