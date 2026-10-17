"""Pristine errob fixtures on original CPU-WRF Swiss h1 fields; three calls.

Tolerance registered here: 32 eps times operand scale (theta 600 K, pressure
100 cb, winds 40 m/s, moisture .05, RKO 2*nz), finite values mandatory. Fixed
before candidate comparison; includes FMA/pow/log/exp and pressure cancellation.
"""

import hashlib,json,subprocess,sys
from pathlib import Path
import netCDF4
import numpy as np

SOURCE=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_01:00:00')


def main(build,save):
    save.mkdir(parents=True,exist_ok=True)
    with netCDF4.Dataset(SOURCE) as ds:
        def read(name):return np.asarray(ds[name][0])
        full={k:read(k) for k in ('U','V','T','QVAPOR','PB','P','PH','PHB','HGT')}
    nz=full['T'].shape[0];ny=7;nx=8
    base={k:v[:,17:24,15:23] if v.ndim==3 else v[17:24,15:23] for k,v in full.items()}
    base['U']=full['U'][:,17:24,15:24]
    base['V']=full['V'][:,17:25,15:23]
    manifest={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'oracle':json.loads((build/'build.json').read_text()),'cases':{},
              'tolerance':'32*eps*operand-scale; theta600K,press100cb,wind40m/s,moisture.05,RKO2*nz; finite mandatory'}
    for kind in ('r4','r8'):
        dtype=np.float32 if kind=='r4' else np.float64
        a=lambda x:np.asarray(x,dtype)
        qv=a(base['QVAPOR']);rvrd=dtype(dtype(461.6)/dtype(287.))
        tb=(a(base['T'])+dtype(300))/(dtype(1)+rvrd*qv)-dtype(300)
        zw=(a(base['PH'])+a(base['PHB']))/dtype(9.81)
        z=dtype(.5)*(zw[1:]+zw[:-1])
        ii,jj=np.meshgrid(np.arange(nx),np.arange(ny))
        ur=a(1.1+ii*.03);vr=a(1.2+jj*.02);tr=a(1.+ii*.00005)
        ur[1,1]=0;vr[1,1]=0;tr[1,1]=0  # missing similarity ratios ->1 fallback
        kpbl=np.asarray(3+(ii+jj)%6,np.int32)
        p=a(base['PB'])+a(base['P'])
        # Surface, pressure upper air, height-only, gross-T check, missing-T,
        # and fractional rko at surface interpolation boundary.
        rio=a([3.,4.3,5.4,3.2,6.,3.7]);rjo=a([3.,3.8,4.2,5.1,4.3,3.1])
        rko=a([1.,-99.,-99.,1.,-99.,1.05])
        var=np.zeros((6,6),dtype)
        var[:,0]=a([3.,8.,-2.,3.,5.,1.]);var[:,1]=a([-2.,1.,2.,-1.,0.,3.])
        # Observed T consistent with model height/pressure except intentional gross check.
        var[:,4]=a([p[0,2,2],p[12,3,3],-888888.,p[0,4,2],p[20,3,4],p[0,2,2]])*dtype(.001)
        var[2,4]=dtype(-888888.)
        var[:,5]=a([99999.,99999.,z[8,3,4],99999.,99999.,99999.])
        theta0=np.mean(tb,axis=(1,2))+dtype(300)
        for n,k in enumerate((0,12,8,0,20,0)):
            press=dtype(np.mean(p[k]))*dtype(.001) if n==2 else var[n,4]
            var[n,2]=(theta0[k]+dtype(.5))*(press/dtype(100))**dtype(dtype(287)/dtype(1004.5))
        var[3,2]+=dtype(60);var[4,2]=dtype(-888888.)
        var[:,3]=a([.004,.002,.003,.004,-888888.,.005])
        for wind,temp,mois in ((1,1,1),(1,0,0),(0,1,0),(0,0,1)):
            name=f'errors_{wind}{temp}{mois}_{kind}.npz'
            data=dict(u=a(base['U']),v=a(base['V']),tb=tb,qv=qv,pbase=a(base['PB']),pp=a(base['P']),z=z,
                      uratx=ur,vratx=vr,tratx=tr,kpbl=kpbl,rio=rio,rjo=rjo,initial_rko=rko,initial_varobs=var)
            inp,out=build/'errors_in.bin',build/'errors_out.bin'
            with inp.open('wb') as f:
                f.write(np.asarray([nx,ny,nz,len(rko),3,wind,temp,mois],np.int32).tobytes())
                for key in ('u','v','tb','qv','pbase','pp','z','uratx','vratx','tratx','kpbl'):
                    arr=data[key]
                    f.write((arr.transpose(1,0,2) if arr.ndim==3 else arr).tobytes())
                for key in ('rio','rjo','initial_rko','initial_varobs'):f.write(data[key].tobytes())
            subprocess.run([str(build/kind/'errors'),str(inp),str(out)],check=True,timeout=20)
            raw=np.fromfile(out,dtype).reshape(3,-1)
            n=len(rko)
            data.update(oracle_varobs=raw[:,:6*n].reshape(3,n,6),oracle_rko=raw[:,6*n:7*n],
                        oracle_errf=raw[:,7*n:].reshape(3,n,10))
            assert all(np.isfinite(data[k]).all() for k in data)
            np.savez_compressed(save/name,**data)
            manifest['cases'][name]={'wind':bool(wind),'temperature':bool(temp),'moisture':bool(mois),
                                     'sha256':hashlib.sha256((save/name).read_bytes()).hexdigest()}
            inp.unlink();out.unlink()
    (save/'ERROR_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
