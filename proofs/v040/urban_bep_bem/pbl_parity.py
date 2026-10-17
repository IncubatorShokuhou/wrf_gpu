"""Qualify the actual YSU/BouLac urban consumers against pristine wrapper outputs."""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.pbl_ysu import ysu_columns
from gpuwrf.physics.pbl_boulac import boulac_columns
from gpuwrf.physics.urban_bep_blend import UrbanPblSources
from conditioned_gate import load_budget,finite_reference,comparison


def run(root,out,schemes=('ysu','boulac'),precisions=('fp64','fp32'),budget=None):
    assert os.environ['JAX_PLATFORMS']=='cpu';assert jax.devices()[0].platform=='cpu'
    results={};budgets=load_budget(budget)
    for scheme in schemes:
        results[scheme]={}
        for precision in precisions:
            # A production parent retains WRF DOUBLE islands; REAL PBL entries
            # must establish their own precision rather than rely on this flag.
            jax.config.update('jax_enable_x64',True)
            dtype=np.float64 if precision=='fp64' else np.float32
            data=[json.loads((root/f'{scheme}_{precision}_{case}.json').read_text()) for case in range(1,7)]
            for item in data:finite_reference(item)
            data64=[json.loads((root/f'{scheme}_fp64_{case}.json').read_text()) for case in range(1,7)] if precision=='fp32' else None
            if data64 is not None:
                for item in data64:finite_reference(item,'REAL8 reference')
            column=lambda name:jnp.asarray([d['columns'][name] for d in data],dtype=dtype)
            scalar=lambda name:jnp.asarray([d['scalars'][name] for d in data],dtype=dtype)
            fields=[column('URBAN_'+name) for name in ['A_U','A_V','A_T','A_Q','A_E','B_U','B_V','B_T','B_Q','B_E','DLG_BEP','DL_U_BEP']]
            sf=column('URBAN_SF_BEP');sf=jnp.concatenate((sf,jnp.ones_like(sf[:,:1])),axis=-1)
            sources=UrbanPblSources(*fields,sf,column('URBAN_VL_BEP'),scalar('FRC_URB'))
            if scheme=='boulac':
                fn=jax.jit(lambda u,v,th,qv,qc,rho,dz,tke,hfx,qfx,ust,dt,urban:boulac_columns(
                    u,v,th,qv,qc,rho,dz,tke,hfx=hfx,qfx=qfx,ust=ust,dt=dt,urban=urban,sf_urban_physics=2))
                values=fn(column('U'),column('V'),column('TH'),column('QV'),column('QC'),column('RHO'),column('DZ'),column('TKE_IN'),
                          scalar('HFX'),scalar('QFX'),scalar('UST'),scalar('DT'),sources)
                names={'u':'RUBLTEN','v':'RVBLTEN','theta':'RTHBLTEN','qv':'RQVBLTEN','qc':'RQCBLTEN','tke':'TKE','pblh':'PBLH',
                       'dlk':'DLK','exch_h':'EXCH_H','exch_m':'EXCH_M','wu':'WU','wv':'WV','wt':'WT','wq':'WQ'}
            else:
                fn=jax.jit(lambda u,v,t,qv,p,pdi,pi,dz,ps,znt,ust,hfx,qfx,wspd,br,psim,psih,dt,xland,u10,v10,urban:ysu_columns(
                    u,v,t,qv,p,pdi,pi,dz,psfc=ps,znt=znt,ust=ust,hfx=hfx,qfx=qfx,wspd=wspd,br=br,psim=psim,psih=psih,
                    dt=dt,xland=xland,u10=u10,v10=v10,urban=urban,sf_urban_physics=2,
                    ysu_topdown_pblmix=0))  # oracle driver: topdown = 0 (v060 base)
                values=fn(column('U'),column('V'),column('T'),column('QV'),column('P'),column('PDI'),column('PI'),column('DZ'),
                    *(scalar(n) for n in ['PSFC','ZNT','UST','HFX','QFX','WSPD','BR','PSIM','PSIH','DT','XLAND','U10','V10']),sources)
                names={'u':'RUBLTEN','v':'RVBLTEN','theta':'RTHBLTEN','qv':'RQVBLTEN','exch_h':'EXCH_H','exch_m':'EXCH_M',
                       'pblh':'PBLH','kpbl':'KPBL','wstar':'WSTAR','delta':'DELTA'}
            rows=[]
            for key,name in names.items():
                bucket='scalars' if name in data[0]['scalars'] else 'columns'
                actual=np.asarray(values[key]);reference=np.asarray(scalar(name) if bucket=='scalars' else column(name))
                assert actual.dtype==dtype or (key=='kpbl' and actual.dtype.kind=='i'),(key,actual.dtype)
                assert np.isfinite(actual).all(),key
                if precision=='fp64':limit=2e-11+2e-12*np.abs(reference)
                else:
                    r64=np.asarray([item[bucket][name] for item in data64])
                    limit=2.*np.abs(reference-r64)+8.*np.finfo(np.float32).eps*np.maximum(np.abs(reference),1.e-8)
                if key=='kpbl':limit=np.zeros_like(reference)
                extra=0.
                if precision=='fp32' and scheme+'__'+key in budgets:
                    extra=budgets[scheme+'__'+key];assert extra.shape==actual.shape
                rows.append(dict(field=key,**comparison(actual,reference,limit,extra,exact=key=='kpbl')))
            results[scheme][precision]={'fields':rows,'violations':sum(x['violations'] for x in rows)}
    out.write_text(json.dumps(results,indent=2)+'\n');print({s:{p:r['violations'] for p,r in d.items()} for s,d in results.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--schemes',nargs='+',choices=['ysu','boulac'],default=['ysu','boulac'])
    parser.add_argument('--precisions',nargs='+',choices=['fp64','fp32'],default=['fp64','fp32'])
    parser.add_argument('--budget',type=Path)
    args=parser.parse_args();run(args.root,args.out,args.schemes,args.precisions,args.budget)
