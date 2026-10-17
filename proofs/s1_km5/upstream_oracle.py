"""Independent pristine upstream SMS stages; explicit surface-regime operands.

Gate pre-registered: full finite outputs, 32eps relative +32eps*field max.
Cases use the tracked real Swiss column structure, prescribed BR/RMOL regimes
and nonzero incoming GAMU/GAMV to exercise WRF's held inactive-column values.
"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from proofs.s1_km5 import oracle
from proofs.o1_smag3d import oracle_io

FIELDS3=('nlflux','dlk','elmin','rdz','rdzw','theta','tke','qv','rho','u','v','p8w','t8w')
FIELDS2=('gamu','gamv','hpbl','hfx','rmol','br','wspd','ust','msftx','msfty','hfx_start','gu_start','gv_start')
CASES={
 'real_mixed':dict(dx=600.,periodic=False,isfflx=1),
 'periodic_convective':dict(dx=1200.,periodic=True,isfflx=2,heat=.06),
 'zero_tke_stable':dict(dx=400.,periodic=False,isfflx=0,zero_tke=True,heat=0.),
}


def generate(exe,out,work):
    saved={}
    with np.load(oracle.SOURCE) as src:
        for name,cfg in CASES.items():
            case=oracle.case_for(src,cfg)
            ny,nx=case['hpbl'].shape
            # Real-structured regime forcing, not fabricated scheme outputs.
            rmol=np.broadcast_to(np.linspace(-.002,.002,nx,dtype=np.float32),(ny,nx)).copy()
            br=np.where(rmol>0,np.float32(.1),np.float32(-.1))
            if name=='periodic_convective':br[:]=-.1;rmol[:]=-.001
            if name=='zero_tke_stable':br[:]=.1;rmol[:]=.002
            u=.5*(case['u'][0,:,:-1]+case['u'][0,:,1:]);v=.5*(case['v'][0,:-1]+case['v'][0,1:])
            wspd=np.maximum(np.sqrt(u*u+v*v),np.float32(.1))
            gu=np.full((ny,nx),.0013,np.float32);gv=np.full((ny,nx),-.0021,np.float32)
            fcfg=dict(km_opt=2,vertical=True,periodic=cfg['periodic'],isfflx=cfg['isfflx'],tke_heat_flux=cfg.get('heat',0.))
            ip,op=work/(name+'.bin'),work/(name+'.out')
            oracle_io.write_input(case,ip,fcfg)
            with ip.open('ab') as f:
                f.write(case['hpbl'].T.tobytes(order='F'));f.write(oracle_io._fo(case['dlk']))
                for x in (rmol,br,wspd,gu,gv):f.write(x.T.tobytes(order='F'))
            subprocess.run([str(exe),str(ip),str(op)],check=True,timeout=30)
            raw=op.read_bytes();ims,ime,jms,jme,kms,kme=np.frombuffer(raw[:24],np.int32)
            ni,nj,nk=ime-ims+1,jme-jms+1,kme-kms+1;flat=np.frombuffer(raw[24:],np.float32)
            size=ni*nj*nk
            for i,n in enumerate(FIELDS3):
                a=flat[i*size:(i+1)*size].reshape((ni,nk,nj),order='F').transpose(1,2,0)
                saved[name+'/'+n]=np.array(a[:,2:-2,2:-2])
            off=len(FIELDS3)*size;size2=ni*nj
            for i,n in enumerate(FIELDS2):
                a=flat[off+i*size2:off+(i+1)*size2].reshape((ni,nj),order='F').T
                saved[name+'/'+n]=np.array(a[2:-2,2:-2])
            saved[name+'/dims']=np.array([nk-1,ny,nx],np.int32)
            ip.unlink();op.unlink()
    saved['meta_json']=np.frombuffer(json.dumps(CASES,sort_keys=True).encode(),np.uint8)
    np.savez_compressed(out,**saved)


def execute(z,key):
    from gpuwrf.kernels.phys_sms3dtke import upstream
    cfg=CASES[key];nz,ny,nx=map(int,z[key+'/dims'])
    edge=0 if cfg['periodic'] else 1
    js,is_=slice(3+edge,3+ny-edge),slice(3+edge,3+nx-edge)
    get=lambda n:z[key+'/'+n][:,js,is_] if n in FIELDS3 else z[key+'/'+n][js,is_]
    fields=['rdzw','rdz','theta','p8w','t8w','tke','qv','rho','u','v']
    args=[get(n)[:nz+1] if n in ('rdz','p8w','t8w') else get(n)[:nz] for n in fields]
    args += [get(n) for n in ('ust','hfx_start','br','rmol','wspd','msftx','msfty','gu_start','gv_start')]
    got=upstream(*args,dx=cfg['dx'],dy=cfg['dx'],isfflx=cfg['isfflx'],heat=cfg.get('heat',0.))
    names=('nlflux','gamu','gamv','hpbl','dlk','elmin','hfx')
    expected=[get(n)[:nz] if n=='elmin' else get(n) for n in names]
    return names,got,expected


if __name__=='__main__':generate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
