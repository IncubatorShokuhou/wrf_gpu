"""Freeze actual Swiss CPU-WRF forecast profiles for cuplus caller oracles.

This is input assembly, not a captured physics savepoint or a forecast parity
claim. P_HYD and the independent total-pressure PI are retained. Density is
reconstructed with the original hypsometric-2 phy_prep equation. Missing KPBL
and ZOL history diagnostics are explicit caller controls in the manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'src'))
    from gpuwrf.io.netcdf_lock import Dataset
    result={};rows=[];sources={}
    for hour in (13,19):
        path=args.case/f'wrfout_d01_2023-01-15_{hour:02d}:00:00'
        with path.open('rb') as f:sources[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
        with Dataset(path) as ds:
            read=lambda key:np.asarray(ds[key][0],np.float32)
            hfx=read('HFX');pblh=read('PBLH');land=read('LANDMASK')
            select=(land>.5)&(pblh>0.)
            select[:2]=False;select[-2:]=False;select[:,:2]=False;select[:,-2:]=False
            ranked=np.argsort(np.where(select,hfx,-np.inf).ravel())
            points=[]
            for linear in ranked[::-1]:
                y,x=map(int,np.unravel_index(linear,hfx.shape))
                if all(abs(y-a)+abs(x-b)>8 for a,b in points):points.append((y,x))
                if len(points)==2:break
            for y,x in points:
                r4=np.float32
                get=lambda key:np.asarray(ds[key][0,:,y,x],np.float32)
                theta=get('T')+r4(300.)
                pressure=get('P')+get('PB')
                pi=(pressure/r4(1e5))**r4(r4(287.)/r4(1004.5))
                temperature=theta*pi
                ph=get('PH')+get('PHB')
                zw=ph/r4(9.81)
                dz=zw[1:]-zw[:-1]
                z=r4(.5)*(zw[:-1]+zw[1:])
                qv=get('QVAPOR')
                u=r4(.5)*(np.asarray(ds['U'][0,:,y,x],np.float32)+np.asarray(ds['U'][0,:,y,x+1],np.float32))
                v=r4(.5)*(np.asarray(ds['V'][0,:,y,x],np.float32)+np.asarray(ds['V'][0,:,y+1,x],np.float32))
                w=get('W')
                mut=r4(ds['MU'][0,y,x])+r4(ds['MUB'][0,y,x])
                c3h=read('C3H');c4h=read('C4H');c3f=read('C3F');c4f=read('C4F')
                ptop=r4(np.asarray(ds['P_TOP'][0]).item())
                pm=c3h*mut+c4h+ptop
                pf=c3f*mut+c4f+ptop
                alt=(ph[1:]-ph[:-1])/pm/np.log(pf[:-1]/pf[1:])
                rho=(r4(1.)+qv)/alt
                terrain=r4(ds['HGT'][0,y,x])
                kpbl=max(1,int(np.count_nonzero(z-terrain<pblh[y,x])))
                condensate={key:get(name) for key,name in
                    (('QC','QCLOUD'),('QI','QICE'),('QR','QRAIN'),('QS','QSNOW'),('QG','QGRAUP'))}
                qtot=qv.copy()
                for key in ('QC','QR','QI','QS','QG'):qtot=qtot+condensate[key]
                faces=np.empty(len(qv)+1,np.float32);faces[-1]=ptop
                c1h=read('C1H');c2h=read('C2H');dnw=read('DNW')
                for k in range(len(qv)-1,-1,-1):
                    mass=c1h[k]*mut+c2h[k]
                    faces[k]=faces[k+1]-(r4(1.)+qtot[k])*mass*dnw[k]
                profile=dict(T=temperature,QV=qv,P=get('P_HYD'),DZ=dz,RHO=rho,U=u,V=v,W=w,
                             W0AVG=np.zeros_like(qv),PI=pi,TH=theta,Z=z,ZW=zw,
                             P8W=faces,TKE=r4(.5)*get('QKE'),CLDFRA=get('CLDFRA'),
                             **condensate,
                             CTRL=np.asarray([float(ds.DX),18.,hfx[y,x],ds['QFX'][0,y,x],pblh[y,x],
                                 kpbl,ds['UST'][0,y,x],-.1,1.,terrain,ds['PSFC'][0,y,x],ds['TSK'][0,y,x]],np.float32))
                number=len(rows)
                for key,value in profile.items():
                    assert value.dtype==np.float32 and np.isfinite(value).all(),key
                    result[f'{number}_{key}']=value
                assert np.all(dz>0.) and np.all(rho>0.) and np.all(pressure>0.)
                rows.append(dict(index=number,hour=hour,y=y,x=x,source=str(path),variant='actual forecast',
                                 hfx=float(hfx[y,x]),pblh=float(pblh[y,x]),kpbl=kpbl))
    # Controlled variants retain the actual thermodynamic/geometry profile.
    base=0
    for variant in ('weak-nonzero-wind','large-grid-spacing'):
        number=len(rows)
        profile={key.split('_',1)[1]:value.copy() for key,value in result.items() if key.startswith(f'{base}_')}
        if variant=='weak-nonzero-wind':
            magnitude=np.sqrt(profile['U'][0]**2+profile['V'][0]**2)
            assert magnitude>0.
            ratio=np.float32(.02)/magnitude
            profile['U']*=ratio;profile['V']*=ratio
        else:profile['CTRL'][0]=np.float32(27000.)
        for key,value in profile.items():result[f'{number}_{key}']=value
        rows.append(dict(rows[base],index=number,variant=variant,base=base))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,**result)
    manifest=dict(scope=__doc__,source_files_sha256=sources,rows=rows,
        fixture_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        controls=['DX','DT','HFX','QFX','PBLH','KPBL','UST','ZOL','XLAND','HT','PSFC','TSK'],
        declared_assembly={'PI':'independent total pressure (P+PB)/1e5 to WRF REAL R/CP',
            'P':'stored original P_HYD','RHO':'original hypsometric-2 ALT reconstruction from PH/PHB, C3/C4, MU/MUB',
            'KPBL':'explicit caller control from mass heights below stored PBLH; not a captured PBL index',
            'ZOL':'explicit controlled -0.1; unavailable in original history',
            'W0AVG':'fresh-cloud zero running-mean carry; original CPS performs its W update',
            'P8W':'original phy_prep hydrostatic top-down REAL recurrence, C1H/C2H/DNW and all six stored moisture species',
            'QC/QI/CLDFRA':'stored original QCLOUD/QICE/CLDFRA profiles',
            'TKE':'half stored QKE from original MYNN forecast; input-profile control, not a CAM-UW or MYJ savepoint',
            'DT':'18s original Swiss namelist; DX from original global attribute',
            'sampling':'two separated interior land columns with largest HFX per hour; no output parity claim'})
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(columns=len(rows),levels=len(result['0_T']),rows=rows)),flush=True)


if __name__=='__main__':main()
