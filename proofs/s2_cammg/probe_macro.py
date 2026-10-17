"""Original integrated macrophysics inputs -> returned state/held cloud fields."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();sys.path.insert(0,str(args.tree/'src'))
    from gpuwrf.physics.cammg import macro
    from gpuwrf.physics.bl_camuw import EPSILO,LATVAP,LATICE,RH2O,CPAIR,TMELT
    from compiler_math import compiler_math as xp
    nz=24;shapevals=dict(pcols=1,kte=nz,pcnst=12)
    qmin=np.zeros(12);qmin[0]=1.e-12
    globals=dict(pcols=1,pver=nz,pverp=nz+1,pcnst=12,ixcldliq=2,ixcldice=3,ixnumliq=4,ixnumice=5,epsqs=np.float64(EPSILO),tmelt=np.float64(TMELT),
                 hlatv=np.float64(LATVAP),hlatf=np.float64(LATICE),rgasv=np.float64(RH2O),cp=np.float64(CPAIR),
                 latvap=np.float64(LATVAP),cpair=np.float64(CPAIR),qmin=qmin,
                 tmin=np.float64(173.16),ptend_top_level=1,ptend_bot_level=nz)
    with np.load(args.tree/'proofs/s2_cammg/fixtures/stages.npz') as z:bank={k:z[k] for k in z.files}
    receipt={'rows':[],'BYTE':True};start=time.monotonic()
    try:
        for case in range(1,19):
            for step in (1,2):
                prefix=f'c{case:02d}_s{step}';data={}
                scalars=dict(is_first_step=step==1,pcnst=12,pcols=1,kte=nz,ixcldliq=2,ixnumliq=4,lchnk=case,dtime=np.float64(18))
                for name in macro.ARGUMENTS['macrophysics_simple']:
                    spec=macro.VARIABLES['macrophysics_simple'][name]
                    shape=tuple(eval(d,{},shapevals) for d in spec['dims'])
                    if name in scalars:value=scalars[name]
                    elif prefix+'_macro_in_'+name in bank:value=bank[prefix+'_macro_in_'+name].reshape(shape,order='F')
                    elif spec['kind']=='str':value=''
                    else:value=np.zeros(shape,dtype=dict(bool=np.bool_,i32=np.int32,f64=np.float64)[spec['kind']])
                    data[name]=value
                with np.errstate(all='ignore'):
                    outputs=macro.macrophysics_simple(**data,_globals=globals,xp=xp)
                row={'case':case,'step':step,'BYTE':True,'fields':{}}
                for name,value in zip(macro.OUTPUTS['macrophysics_simple'],outputs,strict=True):
                    if name=='_fatal':assert value==0,('originalfatal',case,step);continue
                    key=prefix+'_macro_out_'+name
                    if name=='state1_t':key=prefix+'_aero_in_state1_t'
                    if key not in bank:continue # private reset flags/labels not exported by WRF
                    got=np.asarray(value).reshape(-1,order='F');ref=bank[key]
                    assert got.shape==ref.shape and got.dtype==ref.dtype,(name,got.shape,ref.shape)
                    assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)),name
                    byte=got.tobytes()==ref.tobytes();row['BYTE']&=byte
                    if not byte:row['fields'][name]={'different':int(np.count_nonzero(got!=ref)),
                        'max_abs':float(np.max(np.abs(got-ref))),'first':np.argwhere(got!=ref)[:5].tolist()}
                receipt['rows'].append(row);receipt['BYTE']&=row['BYTE']
        receipt['seconds']=time.monotonic()-start
    finally:args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));assert receipt['BYTE'],'original macro BYTE gate failed'


if __name__=='__main__':main()
