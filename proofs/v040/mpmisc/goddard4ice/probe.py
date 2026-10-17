"""Compare all port outputs with the independently compiled original WRF."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/v025'))
import cpu_guard
sys.path.insert(0,str(ROOT))
import argparse
import json
import numpy as np
from types import FunctionType
import jax
from gpuwrf.physics.goddard4ice import support,equations,column
from proofs.v040.mpmisc.literal_reference import _LiteralNumpy,_LiteralLax,_wrap


def copy(fn,env):
    out=FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)
    out.__kwdefaults__=fn.__kwdefaults__
    return out


def literal():
    numpy=_LiteralNumpy()
    class Work:
        def __init__(self,**fields):self.values={k:_wrap(v) for k,v in fields.items()}
        def __getattr__(self,name):return self.values.get(name,_wrap(np.asarray(0,np.float32)))
        def __setattr__(self,name,value):
            if name=='values':object.__setattr__(self,name,value)
            else:self.values[name]=_wrap(value)
        def put(self,mask,**fields):
            for name,value in fields.items():self.values[name]=numpy.where(mask,value,self.values.get(name,np.zeros_like(value)))
    f=lambda x:_wrap(np.asarray(x,np.float32))
    d=lambda x:_wrap(np.asarray(x,np.float64))
    env=dict(vars(support),jnp=numpy,F=f,D=d,_Work=Work)
    for name in ['eff_rad','gamma_toshi','ipow','table_lookup','shift_up','first_density']:env[name]=copy(getattr(support,name),env)
    eqenv=dict(vars(equations),jnp=numpy,F=f,D=d,_Work=Work,**{k:env[k] for k in ['eff_rad','gamma_toshi','ipow','table_lookup','shift_up','first_density']})
    for name in ['sgmap','vqrqi','saticel_s']:eqenv[name]=copy(getattr(equations,name),eqenv)
    colenv=dict(vars(column),jnp=numpy,F=f,lax=_LiteralLax,**{k:eqenv[k] for k in ['sgmap','vqrqi','saticel_s']})
    for name in ['_point','_distance','_prepare_saticel','_fall_species','column']:colenv[name]=copy(getattr(column,name),colenv)
    return colenv['column']

FIELDS={'theta':'TH_OUT','qv':'QV_OUT','qc':'QC_OUT','qr':'QR_OUT','qi':'QI_OUT','qs':'QS_OUT','qg':'QG_OUT','qh':'QH_OUT'}
FIELDS.update({'re_'+k+'_gsfc':'RE_'+k.upper() for k in ['cloud','rain','ice','snow','graupel','hail']})
FIELDS.update({k:k.upper() for k in ['phys'+x for x in 'cedsmf']+['acphys'+x for x in 'cedsmf']+['preci3d','precs3d','precg3d','prech3d','precr3d']})
SURFACE={'rainnc':'RAINNCV','snownc':'SNOWNCV','graupelnc':'GRAUPELNCV','hailnc':'HAILNCV','sr':'SR'}
INPUT=['TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','QH_IN','DEN','PII','P','DELZ','Z','W']

def main():
    p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('output',type=Path);p.add_argument('--literal',action='store_true');p.add_argument('--case',type=int,default=0);args=p.parse_args()
    fn=literal() if args.literal else jax.jit(column.column)
    receipt=[]
    for case in range(1,7) if not args.case else [args.case]:
        data=json.loads((args.oracle/f'case{case}.json').read_text());cols=data['columns'];sc=data['scalars']
        inp=[np.asarray(cols[name],np.float32) for name in INPUT]
        with np.errstate(all='ignore'):
            caller=json.loads((args.oracle/'provenance.json').read_text()).get('caller',{'steps':1,'dx':3000.,'land':1})
            diagnostics=None
            for step in range(1,caller['steps']+1):
                result=fn(*inp,np.float32(sc['DT']),np.float32(caller['dx']),np.float32(caller['land']),np.float32(0),step,diagnostics)
                inp[:8]=[result[key] for key in ['theta','qv','qc','qr','qi','qs','qg','qh']]
                diagnostics={key:result[key] for key in result if key.startswith('acphys')}
        item={'case':case,'fields':{},'finite':True,'all_byte':True}
        for key,refname in {**FIELDS,**SURFACE}.items():
            got=np.asarray(result[key],np.float32);ref=np.asarray(cols[refname] if refname in cols else sc[refname],np.float32)
            finite=bool(np.isfinite(got).all() and np.isfinite(ref).all())
            byte=got.shape==ref.shape and got.tobytes()==ref.tobytes()
            item['fields'][key]={'finite':finite,'byte':byte,'maxabs':float(np.max(np.abs(got.astype(float)-ref.astype(float)))),'different':int(np.sum(got!=ref))}
            item['finite']&=finite;item['all_byte']&=byte
        receipt.append(item)
        print(case,'finite',item['finite'],'byte',item['all_byte'],{k:v['maxabs'] for k,v in item['fields'].items() if not v['byte']},flush=True)
    args.output.write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
