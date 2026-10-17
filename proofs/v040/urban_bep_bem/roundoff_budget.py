"""Proposed WRF-operand roundoff terms. Registered gates remain unchanged.

Reads only pristine outputs, WRITE-only calibrated operands and URBPARM.TBL.
No candidate result, residual or fitted scale is used. Eight REAL epsilons are
the already requested arithmetic budget; state and REAL8 gates are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flux_conditioning import budget as wall_budget,wall_value

EPS=8*np.finfo(np.float32).eps
CP=1004.;XLV=2.45e6


def flat_budget(row):
    """Forward sensitivity of literal WRF flux_flat: Louis + saturation.

    The subtraction's own rounding is included alongside the state operands'
    sensitivity. All derivatives are analytic, in the active WRF branch.
    """
    pt,tg,p,rho,q,u,v,aa,wind,fh,rr=(float(row[k]) for k in
        ('pt','ptg','pr','da','qv','ua','va','aa','utot','fh','rr'))
    zz=float(row['dz'])/2.;z0=float(row['z0']);delta=pt-tg
    ri=2*9.81*zz*delta/((pt+tg)*wind**2)
    if ri>0:df=-9.4/(1+.5*9.4*ri)**3
    else:
        c=9.4*5.3*aa*aa*np.sqrt(zz/z0);root=np.sqrt(-ri)
        df=-9.4*(1+.5*c*root)/(1+c*root)**2
    dri_t=4*9.81*zz*tg/((pt+tg)**2*wind**2)
    dri_g=-4*9.81*zz*pt/((pt+tg)**2*wind**2)
    dri_w=-2*ri/wind
    coeff=aa*aa*wind*fh/rr
    dc_t=aa*aa*wind/rr*df*dri_t;dc_g=aa*aa*wind/rr*df*dri_g
    dc_w=aa*aa/rr*(fh+wind*df*dri_w)
    wind_sigma=EPS*np.sqrt(u*u+v*v) if np.sqrt(u*u+v*v)>.01 else 0.
    heat=rho*CP*(abs(coeff+delta*dc_t)*EPS*abs(pt)
        +abs(-coeff+delta*dc_g)*EPS*abs(tg)+abs(delta*dc_w)*wind_sigma
        +abs(coeff*delta)*EPS)
    heat+=EPS*rho*CP*abs(coeff)*(abs(pt)+abs(tg))
    # WRF tmp=ptg*(pr/1e5)**(287/1004)-273.15; ES is in hPa.
    temperature=tg*(p/1.e5)**(287./1004.);tmp=temperature-273.15
    es=6.11*10**(tmp*7.5/(237.7+tmp));qs=float(row['qvsg'])
    correction=1+.378*es/(.01*p-.378*es)
    dqs_T=qs*correction*np.log(10.)*7.5*237.7/(237.7+tmp)**2
    dqs_g=dqs_T*temperature/tg
    dqs_p=dqs_T*(287./1004.)*temperature/p-qs*correction/p
    difference=q-qs
    latent=rho*XLV*(abs(difference*dc_t)*EPS*abs(pt)
        +abs(difference*dc_g-coeff*dqs_g)*EPS*abs(tg)
        +abs(coeff*dqs_p)*EPS*abs(p)+abs(coeff)*EPS*abs(q)
        +abs(difference*dc_w)*wind_sigma+abs(coeff*difference)*EPS)
    latent+=EPS*rho*XLV*abs(coeff)*(abs(q)+abs(qs))
    assert np.isfinite([heat,latent]).all() and heat>=0 and latent>=0
    return heat,latent


def heat_projection(z,urban_z,pb,ss,bs,ws,volume,horizontal,wall):
    """Positive weights of pristine urban_meso, independently evaluated in NumPy."""
    dz=np.diff(z);n=len(urban_z)-1;result=np.zeros(len(dz))
    for d in range(2):
        result[0]+=horizontal[d,0]/volume[0]/dz[0]*ws[d]/(ws[d]+bs[d])/2
        for k in range(1,n):
            hit=(z[:-1]<=urban_z[k])&(z[1:]>urban_z[k])
            result+=np.where(hit,ss[k]*horizontal[d,k],0)*bs[d]/(ws[d]+bs[d])/volume/dz/2
        for k in range(n):
            overlap=np.maximum(np.minimum(urban_z[k+1],z[1:])-np.maximum(urban_z[k],z[:-1]),0)
            result+=pb[k+1]*(wall[2*d,k]+wall[2*d+1,k])*overlap/(ws[d]+bs[d])/volume/dz/2
    return result


def bem_bounds(staged,save,geo,output):
    summary={};floors=tuple(k for k in range(1,geo.n) if geo.ss[k]>0)
    n=geo.n;pw=float(geo.raw['PWIN'][1]);bs=geo.building.astype(float);ws=geo.width.astype(float)
    for arm in ('urban3_ac_on','urban3_ac_off'):
        receipt=staged['arms'][arm];assert receipt['original_arrays_BYTE']==87
        data=np.load(save/(arm.replace('_ac','_fp32_ac')+'.npz'))
        names=('SFR_URB3D','LFR_URB3D','SFG_URB3D','LFG_URB3D','B_T_BEP','HFX','SH_URB')
        bounds={name:np.zeros_like(data[name],dtype=float) for name in names}
        walls=iter(receipt['operands']['BEMWALL']);flats=iter(receipt['operands']['BEMFLAT']);wc=fc=0
        for r in range(6):
          dz=data['IN_DZ'][r,0,:8].astype(np.float32);z=np.r_[np.float32(0),np.cumsum(dz,dtype=np.float32)].astype(float)
          rho=data['IN_RHO'][r,0,:8].astype(float);fraction=float(data['IN_FRACTION'][r,0])
          for step in range(1,7):
            horizontal=np.zeros((2,n));vertical=np.zeros((4,n))
            for d in range(2):
              for k in (0,)+floors:
                row=next(flats);fc+=1;assert int(row['call'])==fc
                heat,latent=flat_budget(row);horizontal[d,k]=heat/(row['da']*CP)
                key='SFG_URB3D' if k==0 else 'SFR_URB3D';index=d if k==0 else k*2+d
                assert np.float32(row['sf'])==data[key][r,step,index],(arm,key,r,step,index)
                bounds[key][r,step,index]=heat
                wetkey='DG_URB3D' if k==0 else 'DGR_URB3D'
                water=float(data[wetkey][r,step,index]);latentkey='LFG_URB3D' if k==0 else 'LFR_URB3D'
                F=np.float32;mass=F(F(row['da'])*F(XLV));raw=F(row['lf'])
                expected=F(F(-mass)*F(-F(F(water)*raw)/mass)) if water>0 else F(0)
                assert expected==data[latentkey][r,step,index],(arm,latentkey,r,step,index)
                bounds[latentkey][r,step,index]=water*latent+abs(row['lf'])*EPS*max(abs(water),1.e-3)
              for b,levels in enumerate(floors):
                for k in range(levels):
                  for side in range(2):
                    row=next(walls);wc+=1;assert int(row['call'])==wc
                    index=b*36+k*2+d
                    for window in (False,True):
                        key=('SFWIN' if window else 'SFW')+str(side+1)+'_URB3D'
                        assert wall_value(row,True,window=window)==data[key][r,step,index]
                    weight=float(geo.ss[levels])/float(geo.pb[k+1])
                    vertical[2*d+side,k]+=weight*((1-pw)*wall_budget(row,True)+pw*wall_budget(row,True,window=True))/(row['da']*CP)
            # The saved volume is WRF's already blended VL_BEP; remove the known
            # rural contribution to recover the geometry used by urban_meso.
            volume=(data['VL_BEP'][r,step,:8].astype(float)-(1-fraction))/fraction
            assert np.all(volume>0)
            projected=heat_projection(z,geo.z.astype(float),geo.pb.astype(float),geo.ss.astype(float),bs,ws,volume,horizontal,vertical)*fraction
            bounds['B_T_BEP'][r,step,:8]=projected
            hfx=float(np.sum(projected*rho*1004.5*dz*volume))
            bounds['HFX'][r,step-1]=hfx
            bounds['SH_URB'][r,step-1]=hfx/fraction
        assert next(walls,None) is None and next(flats,None) is None
        summary[arm]={name:float(a.max()) for name,a in bounds.items()}
        for name,a in bounds.items():output[arm+'__'+name]=a
    return summary


def inverse_update(old,tendency,dt):
    """Bound the original REAL4 state from its recorded update tendency.

    WRF records RN32(RN32(new-old)/dt). Inverting each rounding interval
    with one full ULP (rather than half) gives a conservative interval for new.
    This uses no candidate data and does not pretend to capture an unrecorded
    exact state. The interval is also checked against WRF's gradient fluxes.
    """
    t=np.asarray(tendency,np.float32).astype(float);old=np.asarray(old,float)
    estimate=old+dt*t
    div_error=abs(dt)*np.abs(np.spacing(np.asarray(t,np.float32))).astype(float)
    subtract_error=np.abs(np.spacing(np.asarray(abs(dt*t)+div_error,np.float32))).astype(float)
    radius=div_error+subtract_error
    return estimate,abs(estimate)+radius,radius


def pbl_bounds(root,output):
    summary={};gradient_checks=[]
    for scheme in ('ysu','boulac'):
        data=[json.loads((root/f'{scheme}_fp32_{case}.json').read_text()) for case in range(1,7)]
        bounds={};updates={}
        mapping={'u':('U','RUBLTEN'),'v':('V','RVBLTEN'),'theta':('TH','RTHBLTEN'),
                 'qv':('QV','RQVBLTEN')}
        if scheme=='boulac':mapping['qc']=('QC','RQCBLTEN')
        for field,(old_name,rate_name) in mapping.items():
            records=[];states=[]
            for row in data:
                c=row['columns'];dt=float(row['scalars']['DT'])*(2 if scheme=='ysu' else 1)
                if scheme=='ysu' and field=='theta':
                    old=(np.asarray(c['T'],np.float32)/np.asarray(c['PI'],np.float32)).astype(float)
                else:old=np.asarray(c[old_name],float)
                estimate,upper,radius=inverse_update(old,c[rate_name],dt)
                records.append(EPS*(abs(old)+upper)/dt);states.append((estimate,upper,radius))
            bounds[field]=np.asarray(records);updates[field]=states
        if scheme=='boulac':
            # Eleven positive arithmetic operations bound construction of the
            # recorded RHOZ/CD/SF/DZ gradient weight. gamma_n is the standard
            # floating-point product/sum bound, not a fitted gate multiplier.
            u=np.finfo(np.float32).eps/2;gamma=11*u/(1-11*u)
            for field,key in [('u','WU'),('v','WV'),('theta','WT'),('qv','WQ')]:
                records=[]
                for case,row in enumerate(data):
                    c=row['columns'];rho=np.asarray(c['RHO'],float);dz=np.asarray(c['DZ'],float)
                    sf=np.r_[np.asarray(c['URBAN_SF_BEP'],float),0.];sf[-1]=sf[0]
                    cd=np.r_[np.asarray(c['EXCH_H' if field in ('theta','qv') else 'EXCH_M'],float),0.]
                    rhoz=(rho[1:]*dz[:-1]+rho[:-1]*dz[1:])/(dz[1:]+dz[:-1])
                    weight=2*sf[1:-1]*rhoz*cd[1:-1]/(dz[1:]+dz[:-1])/rho[1:]
                    estimate,upper,radius=updates[field][case]
                    predicted=-weight*(estimate[1:]-estimate[:-1]);reference=np.asarray(c[key],float)[1:]
                    uncertainty=weight*(radius[1:]+radius[:-1])+gamma*abs(predicted)
                    error=abs(predicted-reference)
                    assert np.all(error<=uncertainty+np.finfo(float).tiny),(scheme,key,case,float(error.max()),float(uncertainty.max()))
                    gradient_checks.append({'field':key,'case':case+1,
                        'inverse_state_join_max_abs':float(error.max()),'allowed_rounding_max':float(uncertainty.max())})
                    records.append(np.r_[0.,EPS*weight*(1+gamma)*(upper[1:]+upper[:-1])])
                bounds[key.lower()]=np.asarray(records)
        summary[scheme]={name:float(a.max()) for name,a in bounds.items()}
        for name,a in bounds.items():output[scheme+'__'+name]=a
    return summary,gradient_checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--operands',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True);p.add_argument('--wall-bounds',type=Path,required=True)
    p.add_argument('--table',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    # Only host parameters are read from this already qualified table parser;
    # this script never calls a candidate physics step or loads its outputs.
    from gpuwrf.physics.urban_bep_static import read_bep_classes
    _,classes=read_bep_classes(a.table,np.float32);geo=classes[1]
    staged=json.loads(a.operands.read_text())
    output={name:value for name,value in np.load(a.wall_bounds).items()}
    canopy=bem_bounds(staged,a.save,geo,output)
    pbl,checks=pbl_bounds(a.save/'pbl',output)
    np.savez_compressed(a.out.with_suffix('.npz'),**output)
    record={'status':'PROPOSED_REFERENCE_BUDGET_ONLY_REGISTERED_GATES_UNCHANGED',
        'operands_sha256':hashlib.sha256(a.operands.read_bytes()).hexdigest(),
        'table_sha256':hashlib.sha256(a.table.read_bytes()).hexdigest(),
        'epsilon_multiple':8,'candidate_results_read':False,
        'canopy_max_additional_budget':canopy,'pbl_max_additional_budget':pbl,
        'original_gradient_inverse_interval_checks':checks,
        'state_and_REAL8_gates':'UNCHANGED',
        'operators':'wall subtraction; analytic Louis/saturation sensitivity; positive urban_meso and Noah-MP weights; update/gradient subtraction'}
    a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
