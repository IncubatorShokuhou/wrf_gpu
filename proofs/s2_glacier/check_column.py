"""Censused Swiss columns and active glacier regimes versus unchanged WRF."""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.physics.noahmp.glacier import glacier_column

CPU=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_00:00:00')


class Literal:
    """NumPy where converts Python scalar branches to R8; WRF assignment is R4."""
    def __getattr__(self,name):
        function=getattr(np,name)
        if name in ('log','exp','sqrt','arctan'):
            # gfortran's scalar REAL intrinsics use libm *f, while NumPy may
            # select a SIMD approximation differing by one binary32 ULP.
            symbol={'log':'logf','exp':'expf','sqrt':'sqrtf','arctan':'atanf'}[name]
            scalar=getattr(ctypes.CDLL('libm.so.6'),symbol)
            scalar.argtypes=[ctypes.c_float];scalar.restype=ctypes.c_float
            function=np.vectorize(lambda x: scalar(float(x)),otypes=[np.float32])
        def call(*args,**kwargs):
            value=function(*args,**kwargs)
            if np.asarray(value).dtype.kind=='f':
                value=np.asarray(value,np.float32)
            return value
        return call


def columns():
    with Dataset(CPU) as d:
        d.set_auto_mask(False)
        mask=np.asarray(d['IVGTYP'][0])==15
        rows=[]
        for y,x in np.argwhere(mask):
            def a(name):return np.asarray(d[name][0,...,y,x],np.float32)
            p=a('P')[0]+a('PB')[0];q=a('QVAPOR')[0]
            z=(a('PH')+a('PHB'))/np.float32(9.81)
            ice=a('SNICE');liq=a('SNLIQ');total=ice+liq
            row=dict(iloc=np.float32(x+1),jloc=np.float32(y+1),nsnow=np.float32(3),nsoil=np.float32(4),
                cosz=np.float32(0),dt=np.float32(18),sfcprs=p,
                sfctmp=(a('T')[0]+np.float32(300))*(p/np.float32(1.e5))**np.float32(287/1004.5),
                uu=a('U')[0],vv=a('V')[0],q2=q/(1.+q),soldn=np.float32(0),lwdn=np.float32(240),prcp=np.float32(0),
                tbot=np.minimum(a('TMN'),np.float32(263.15)),zlvl=(z[0]+z[1])*.5-a('HGT'),
                ficeold=[np.where(total[k]>0.,ice[k]/np.where(total[k]>0.,total[k],1.),0.) for k in range(3)],
                zsoil=list(np.asarray([-.1,-.4,-1.,-2.],np.float32)),
                qsnow=a('QSNOWXY'),sneqvo=np.float32(0),albold=a('ALBOLD'),cm=a('CM'),ch=a('CH'),
                isnow=a('ISNOW').astype(np.int32),sneqv=a('SNOW'),smc=list(a('SMOIS')),zsnso=list(a('ZSNSO')),
                snowh=a('SNOWH'),snice=list(ice),snliq=list(liq),tg=a('TG'),
                stc=list(np.concatenate([a('TSNO'),a('TSLB')])),sh2o=list(a('SH2O')),tauss=a('TAUSS'),qsfc=np.float32(0))
            rows.append((f'Swiss_y{y}_x{x}',row))
    # Active regimes perturb measured glacier columns; their full state is retained.
    origin=rows[0][1]
    def copy():return {k:list(v) if isinstance(v,list) else v.copy() for k,v in origin.items()}
    for name in ('accumulation','melt','refreeze','bare','thin','deep','drain'):
        row=copy()
        row.update(cosz=np.float32(.6),soldn=np.float32(500),sfctmp=np.float32(275),lwdn=np.float32(320))
        if name=='accumulation':
            row.update(sfctmp=np.float32(263),prcp=np.float32(.002),qsnow=np.float32(.001))
        if name in ('melt','refreeze','drain'):
            for k in range(3):
                if k>=row['isnow']+3:
                    transfer=np.float32(.2 if name!='drain' else .5)*row['snice'][k]
                    row['snice'][k]-=transfer;row['snliq'][k]+=transfer
                    row['stc'][k]=np.float32(273.17 if name=='melt' else 270 if name=='refreeze' else 273.16)
            row['ficeold']=[i/np.where(i+l>0.,i+l,1.) for i,l in zip(row['snice'],row['snliq'])]
        if name in ('bare','thin'):
            row.update(isnow=np.int32(0),sneqv=np.float32(0 if name=='bare' else 2),
                       snowh=np.float32(0 if name=='bare' else .02),snice=[np.float32(0)]*3,snliq=[np.float32(0)]*3)
            row['stc'][:3]=[np.float32(0)]*3;row['zsnso']=[np.float32(0)]*3+row['zsoil']
            row['ficeold']=[np.float32(0)]*3
        if name=='deep':
            factor=np.float32(6000)/row['sneqv']
            row['snice']=[a*factor for a in row['snice']];row['snliq']=[a*factor for a in row['snliq']]
            row['sneqv']=np.float32(6000);row['snowh']*=factor
            depth=np.float32(0);z=[]
            for k in range(3):
                if k>=row['isnow']+3:
                    old=origin['zsnso'][k]-(np.float32(0) if k==row['isnow']+3 else origin['zsnso'][k-1])
                    depth=depth+old*factor;z.append(depth)
                else:z.append(np.float32(0))
            row['zsnso']=z+[depth+a for a in row['zsoil']]
        rows.append((name,row))
    return rows


def branch_columns():
    rows=columns();base=rows[0][1]
    def copy():return {k:list(v) if isinstance(v,list) else v.copy() for k,v in base.items()}
    def pack(depths,ices,liquids,temperatures):
        r=copy();count=len(depths);pad=3-count
        r.update(isnow=np.int32(-count),snice=[np.float32(0)]*pad+list(np.asarray(ices,np.float32)),
                 snliq=[np.float32(0)]*pad+list(np.asarray(liquids,np.float32)))
        r['stc'][:3]=[np.float32(0)]*pad+list(np.asarray(temperatures,np.float32))
        depth=np.float32(0);z=[np.float32(0)]*pad
        for d in depths:depth=depth-np.float32(d);z.append(depth)
        r['zsnso']=z+[depth+a for a in r['zsoil']]
        r['snowh']=-depth;r['sneqv']=np.sum(np.asarray(ices,np.float32)+np.asarray(liquids,np.float32),dtype=np.float32)
        r['ficeold']=[i/np.where(i+l>0.,i+l,1.) for i,l in zip(r['snice'],r['snliq'])]
        return r
    cases=[('one_small',[.03],[3.],[0.],[265.]),('one_split',[.15],[15.],[.5],[270.]),
           ('two_thin',[.02,.04],[2.,4.],[.01,.02],[264.,270.]),
           ('two_split',[.12,.4],[12.,40.],[.1,.3],[260.,270.]),
           ('three_merge_top',[.02,.1,.4],[2.,10.,40.],[.01,.02,.03],[260.,265.,270.]),
           ('three_merge_mid',[.05,.01,.4],[5.,1.,40.],[.01,.02,.03],[260.,265.,270.]),
           ('three_merge_bottom',[.05,.2,.05],[5.,20.,5.],[.01,.02,.03],[260.,265.,270.]),
           ('three_remove_top',[.02,.1,.4],[.08,10.,40.],[.02,.1,.1],[270.,270.,270.]),
           ('three_remove_mid',[.05,.01,.4],[5.,.08,40.],[.01,.02,.1],[265.,270.,270.]),
           ('three_remove_bottom',[.05,.2,.01],[5.,20.,.08],[.01,.02,.03],[265.,270.,270.]),
           ('one_remove',[.001],[.08],[.02],[270.])]
    for name,*args in cases:rows.append((name,pack(*args)))
    for name,temperatures,liquid in [('ice_heat_exchange',[274.,260.,275.,265.],.2),
                                    ('ice_freeze_exchange',[260.,275.,265.,274.],.8),
                                    ('ice_melt_all',[274.,274.,274.,274.],.1)]:
        r=copy();r['stc'][3:]=list(np.asarray(temperatures,np.float32));r['sh2o']=list(np.full(4,liquid,np.float32))
        rows.append((name,r))
    for delta in (.49,.51,1.99,2.01,2.19,2.21,2.49,2.51):
        r=copy();r.update(sfctmp=np.float32(273.16)+np.float32(delta),prcp=np.float32(.002),
                         cosz=np.float32(.5),soldn=np.float32(300))
        rows.append((f'rainsnow_{delta}',r))
    return rows


def main(build,output,matrix=False):
    manifest=json.loads((build/'manifest.json').read_text());fields=manifest['fields'];width=manifest['width']
    libs=[]
    for variant in ('zero','alternate'):
        lib=ctypes.CDLL(str(build/f'glacier_{variant}.so'))
        lib.glacier_eval.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')]*2+[np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')]
        libs.append(lib)
    results=[];fixture=[]
    import itertools
    choices=itertools.product((1,2),(1,2),(1,2,3,4),(1,2),(1,2,3)) if matrix else [(g,2,1,2,1) for g in (1,2)]
    for gla,alb,snf,tbot,stc in choices:
      for name,row in (branch_columns() if matrix else columns()):
        packed=np.zeros(width,np.float32)
        for f in fields:
            if f['name'] in row:packed[f['offset']:f['offset']+f['extent']]=row[f['name']]
        options=np.asarray([alb,snf,tbot,stc,gla],np.int32)
        refs=[]
        for lib in libs:
            out=np.zeros(width,np.float32);lib.glacier_eval(packed,out,options);refs.append(out)
        actual=glacier_column(row,opt_gla=gla,opt_alb=alb,opt_snf=snf,opt_tbot=tbot,opt_stc=stc,xp=Literal())
        errors={};sensitive={};nonfinite=[];bad=0
        for f in fields:
            if f['intent']=='IN':continue
            k=f['name'];sl=slice(f['offset'],f['offset']+f['extent']);expected=refs[0][sl]
            val=np.asarray(actual[k],np.float32).reshape(-1)
            defined=np.ones(f['extent'],bool)
            if k=='hcpct':defined[:int(row['isnow'])+3]=False
            if not np.isfinite(val[defined]).all() or not np.isfinite(expected[defined]).all():nonfinite.append(k)
            unequal=(val.view(np.uint32)!=expected.view(np.uint32))&defined
            count=int(unequal.sum());bad+=count
            if count:errors[k]=dict(count=count,max_abs=float(np.max(np.abs(val[defined]-expected[defined]))),actual=val.tolist(),reference=expected.tolist())
            changed=(refs[0][sl].view(np.uint32)!=refs[1][sl].view(np.uint32))&defined
            if changed.any():sensitive[k]=int(changed.sum())
        results.append(dict(name=name,opt_gla=gla,options=options.tolist(),errors=errors,byte_mismatches=bad,nonfinite=nonfinite,sensitive=sensitive))
        fixture.append(dict(name=name,options=options.tolist(),input=packed.tolist(),reference=refs[0].tolist()))
    payload=dict(source_sha256=manifest['source_sha256'],cpu_input_sha256=hashlib.sha256(CPU.read_bytes()).hexdigest(),
                 results=results,literal_backend='NumPy binary32; no compiler contraction',fixtures=fixture)
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(results),byte_mismatches=sum(r['byte_mismatches'] for r in results),
                         failed=[(r['name'],r['opt_gla'],list(r['errors'])) for r in results if r['errors']],
                         nonfinite=[r['name'] for r in results if r['nonfinite']],sensitive=[r['name'] for r in results if r['sensitive']]),indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),'--matrix' in sys.argv[3:])
