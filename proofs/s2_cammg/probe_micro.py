"""Full original MG core stage gate on the actual integrated CAMMGMP operands."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def inputs(bank,prefix,micro,globals):
    nz=globals['pver'];shapevals=dict(pcols=1,pver=nz,pverp=nz+1,ncol=1)
    q=bank[prefix+'_aero_in_state1_q'].reshape((1,nz,12),order='F')
    mapping=dict(sub_column=False,lchnk=1,ncol=1,deltatin=np.float64(18),
                 tn='state1_t',qn=q[:,:,0],p='state1_pmid',pdel='state1_pdel',
                 cldn='ast',liqcldf='alst_mic',icecldf='aist_mic',npccnin='npccn')
    input={}
    for name in micro.ARGUMENTS['mmicro_pcond']:
        spec=micro.VARIABLES['mmicro_pcond'][name]
        shape=tuple(eval(d,{},shapevals) for d in spec['dims'])
        if spec['intent']=='out':value=np.zeros(shape,dtype=np.float64)
        else:
            source=mapping.get(name,name)
            if not isinstance(source,str):value=source
            elif prefix+'_aero_in_'+source in bank:value=bank[prefix+'_aero_in_'+source]
            else:value=bank[prefix+'_aero_out_'+source]
        input[name]=np.asarray(value,dtype=dict(bool=np.bool_,i32=np.int32,f64=np.float64)[spec['kind']]).reshape(shape,order='F')
    return input


OUTPUT_MAP=dict(rate1ord_cw2pr_st='rate1cld',qctend='qcten',qitend='qiten',nctend='ncten',nitend='niten',
                cmeout='cmeice',deffi='dei',pgamrad='mu',lamcrad='lambdac',dsout='des')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',default='1,2,3,4,5,16,17,18');args=ap.parse_args()
    sys.path.insert(0,str(args.tree/'src'))
    from gpuwrf.physics.cammg import micro
    from gpuwrf.physics.cammg.constants import CONSTANTS
    from gpuwrf.physics.bl_camuw import EPSILO,LATVAP,LATICE,RH2O,CPAIR,TMELT
    from compiler_math import compiler_math as xp
    globals={**CONSTANTS,'pcols':1,'pver':24,'pverp':25,'fillvalue':np.float64(1.e36),
             'liu_in':True,'epsqs':np.float64(EPSILO),'tmelt':np.float64(TMELT),
             'hlatv':np.float64(LATVAP),'hlatf':np.float64(LATICE),'rgasv':np.float64(RH2O),'cp':np.float64(CPAIR)}
    with np.load(args.tree/'proofs/s2_cammg/fixtures/stages.npz') as z:bank={k:z[k] for k in z.files}
    receipt={'source':str(args.tree),'rows':[],'BYTE':True};start=time.monotonic()
    try:
        for case in map(int,args.cases.split(',')):
            for step in (1,2):
                prefix=f'c{case:02d}_s{step}';data=inputs(bank,prefix,micro,globals)
                with np.errstate(all='ignore'):
                    got=micro.mmicro_pcond(**data,_globals=globals,xp=xp)
                row={'case':case,'step':step,'fields':{},'BYTE':True}
                for name,value in zip(micro.OUTPUTS['mmicro_pcond'],got,strict=True):
                    ref=bank[prefix+'_micro_out_'+OUTPUT_MAP.get(name,name)]
                    value=np.asarray(value).reshape(-1,order='F')
                    assert value.shape==ref.shape,(name,value.shape,ref.shape)
                    assert np.all(np.isfinite(value)) and np.all(np.isfinite(ref)),name
                    byte=value.tobytes()==ref.tobytes();row['BYTE']&=byte
                    if not byte:row['fields'][name]={'different':int(np.count_nonzero(value!=ref)),
                        'max_abs':float(np.max(np.abs(value-ref))),'first':np.argwhere(value!=ref)[:5].tolist()}
                receipt['rows'].append(row);receipt['BYTE']&=row['BYTE']
        receipt['seconds']=time.monotonic()-start
    finally:
        args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    assert receipt['BYTE'],'original full MG stage BYTE gate failed'


if __name__=='__main__':main()
