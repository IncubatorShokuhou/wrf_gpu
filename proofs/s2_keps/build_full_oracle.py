"""Full WRF 4.7.1 EEPS/KEPS oracles with manager-authorized zero initialization.

Pristine bodies unchanged. Alternative compiler initialization is a sensitivity
arm, never truth. CPU-only; private lane build; artifacts require explicit paths.
Full gate frozen before generation: 16 REAL32 ulps of each state/diagnostic field
scale; tendencies get 16 ulps of the upstream input scale divided by dt to account
for WRF's differencing of rounded states. Integers exact; both arms finite.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

FIELDS = ('u','v','theta','temperature','pressure','qv','qc','qi','qr','qs','qg','rho','dz','raten',
          'exner','p8w','tke','epsilon','tpe','prandtl','tke_adv','epsilon_adv','tpe_adv')


def cases():
    result = []
    for scheme in (16, 17):
        for regime in ('stable','unstable','neutral','stratocumulus','mixed_ice','restart'):
            n = 44 if regime in ('stratocumulus','restart') else 16
            m = n+4
            k = np.arange(m, dtype=np.float32)
            z = k*100.+50.
            theta = 290.+.006*z
            if regime == 'neutral': theta = np.full(m, 300., np.float32)
            if regime == 'unstable': theta = 300.-.003*np.minimum(z,1000.)+.004*np.maximum(z-1000.,0.)
            if regime == 'stratocumulus': theta = 290.+.001*z+np.where(z>1000.,12.,0.)
            if regime == 'mixed_ice': theta = 275.+.003*z
            p = 100000.*np.exp(-z/8000.)
            exner = (p/100000.)**np.float32(np.float32(287.)/np.float32(1004.5))
            temperature = theta*exner
            qv = np.full(m, .004 if regime == 'mixed_ice' else .008, np.float32)
            cloud = ((z>=400.)&(z<=900.)).astype(np.float32)
            qc = cloud*.0005 if regime in ('stratocumulus','mixed_ice') else np.zeros(m,np.float32)
            qi = np.full(m,.0001,np.float32) if regime=='mixed_ice' else np.zeros(m,np.float32)
            rad = -cloud*.0001 if regime=='stratocumulus' else np.zeros(m,np.float32)
            hfx = 50. if regime in ('unstable','stratocumulus') else (-20. if regime=='stable' else 0.)
            br = -.03 if hfx>0 else .02
            fields = dict(u=8.+.08*k,v=2.-.015*k,theta=theta,temperature=temperature,pressure=p,
                          qv=qv,qc=qc,qi=qi,qr=cloud*.0001,qs=cloud*.00005,qg=cloud*.00002,
                          rho=p/(287.*temperature*(1.+np.float32(461.6/287.-1.)*qv)),
                          dz=np.full(m,100.),raten=rad,exner=exner,p8w=100000.*np.exp(-k*100./8000.),
                          tke=np.full(m,.2 if regime=='restart' else .0001),
                          epsilon=np.full(m,.001 if regime=='restart' else .0000001),
                          tpe=np.full(m,.01 if regime=='restart' else .0000001),prandtl=np.ones(m),
                          tke_adv=np.full(m,.2 if regime=='restart' else .0001),
                          epsilon_adv=np.full(m,.001 if regime=='restart' else .0000001),
                          tpe_adv=np.full(m,.01 if regime=='restart' else .0000001))
            result.append(dict(name=f'{scheme}_{regime}',scheme=scheme,n=n,steps=5,
                               first=7 if regime=='restart' else 1,dt=18.,dx=3000.,
                               surface=dict(ust=.3,rmol=-.02 if hfx>0 else .01,wspd=float(np.sqrt(68.)),
                                            hfx=hfx,qfx=.00005,tsk=292.,qsfc=.01,psfc=100000.,
                                            br=br,znt=.1,fm=4.,fhh=4.),
                               input={key:np.asarray(value,np.float32).tolist() for key,value in fields.items()}))
    return result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wrf-root',type=Path,default=Path('<USER_HOME>/src/wrf_pristine/WRF'))
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--init',choices=('zero','snan','inf'),default='zero')
    ap.add_argument('--compiler',default='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    args=ap.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    args.build.mkdir(parents=True,exist_ok=True)
    flags=['-ffree-form','-ffree-line-length-none','-O0','-ffp-contract=off','-fcheck=all',
           f'-finit-real={args.init}','-finit-integer=0','-finit-logical=false']
    sources={};objects=[]
    for rel in ('share/module_model_constants.F','phys/module_bl_keps.F','phys/module_bl_eepsilon.F'):
        src=args.wrf_root/rel;dest=args.build/src.name;shutil.copyfile(src,dest)
        assert sha(src)==sha(dest)
        sources[rel]=sha(src);obj=dest.stem+'.o'
        subprocess.run([args.compiler,*flags,'-c',str(dest),'-o',obj],cwd=args.build,check=True,timeout=40)
        objects.append(obj)
    driver=Path(__file__).with_name('full_oracle.f90')
    subprocess.run([args.compiler,*flags,str(driver),*objects,'-o','full_oracle'],cwd=args.build,check=True,timeout=40)
    scenarios=cases();text=io.StringIO();text.write(f'{len(scenarios)}\n')
    for c in scenarios:
        text.write(' '.join(str(c[k]) for k in ('scheme','n','steps','first','dt','dx'))+'\n')
        text.write(' '.join(f'{v:.9e}' for v in c['surface'].values())+'\n')
        for field in FIELDS: text.write(' '.join(f'{v:.9e}' for v in c['input'][field])+'\n')
    run=subprocess.run([str(args.build/'full_oracle')],input=text.getvalue(),cwd=args.build,
                       capture_output=True,text=True,timeout=60)
    (args.build/'stdout.txt').write_text(run.stdout);(args.build/'stderr.txt').write_text(run.stderr)
    assert run.returncode==0,run.stderr[-3000:]
    lines=iter(run.stdout.splitlines());records=[]
    for c in scenarios:
        for s in range(c['first'],c['first']+c['steps']):
            header=next(lines).split();assert header[0]=='RECORD' and int(header[2])==s,header
            rec=dict(name=c['name'],scheme=c['scheme'],step=s,n=c['n'],dt=c['dt'],dx=c['dx'],surface=c['surface'],input={},output={})
            for _ in range(44):
                field=next(lines).strip();values=[float(v) for v in next(lines).split()]
                rec['input' if field.startswith('I_') else 'output'][field[2:]]=values
            records.append(rec)
    assert next(lines,None) is None
    nonfinite=sum(int((~np.isfinite(v)).sum()) for r in records for v in map(np.asarray,r['output'].values()))
    if args.init=='zero': assert nonfinite==0,nonfinite
    packet=dict(scope='full unchanged WRF EEPS/KEPS drivers; compiler-defined initialization',
                init=args.init,flags=flags,fp_traps='off',source_sha256=sources,driver_sha256=sha(driver),
                input_sha256=hashlib.sha256(text.getvalue().encode()).hexdigest(),
                tolerance=dict(state_field_scale_ulps=16,tendency_upstream_input_scale_ulps_per_dt=16,
                               zero_fields='exact',integer_diagnostics='exact',nonfinite='reject candidate and reference'),
                nonfinite_output_values=nonfinite,records=records)
    args.output.write_text(json.dumps(packet,indent=1)+'\n')
    print(json.dumps(dict(init=args.init,records=len(records),nonfinite=nonfinite,sha256=sha(args.output))))


if __name__=='__main__': main()
