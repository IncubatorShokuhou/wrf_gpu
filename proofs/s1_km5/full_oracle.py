"""Pristine full SMS first_rk_step_part2 operator, including implicit solves.

Numerical gate: finite coupled tendencies; their integrated field-equivalent
error <=32eps32 relative +32eps32*maxabs(entry field). This pre-registers the
REAL subtraction conditioning of mu*(implicit_solution-entry)/dt. Coefficient
and upstream strict component gates remain separate and unchanged.
"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from proofs.s1_km5 import oracle,upstream_oracle
from proofs.o1_smag3d import oracle_io

CASES=upstream_oracle.CASES
OUT3=('ru_tendf','rv_tendf','rw_tendf','t_tendf','tke_tendf',
      'qv','qc','qr','qi','qs','qg','Ni','Nr',
      'xkmh','xkmv','xkhh','xkhv','xkmv_meso','l_diss','nlflux','dlk','elmin')


def make_case(src,cfg,c1f,c2f):
    c=oracle.case_for(src,cfg);nz,ny,nx=c['tke'].shape
    c['moist'].append(np.array(c['moist'][4]*np.float32(.05),np.float32))
    # Adversarial number profiles preserve real vertical/horizontal structure.
    c['Ni']=np.array(c['moist'][3]*np.float32(1.e7),np.float32)
    c['Nr']=np.array(c['moist'][2]*np.float32(1.e7)+c['moist'][4]*np.float32(1.e6),np.float32)
    c['c1f'],c['c2f']=c1f,c2f
    c['sms_br']=np.broadcast_to(np.linspace(-.1,.1,nx,dtype=np.float32),(ny,nx)).copy()
    c['sms_rmol']=np.broadcast_to(np.linspace(-.002,.002,nx,dtype=np.float32),(ny,nx)).copy()
    if cfg['isfflx']==2:c['sms_br'][:]=-.1;c['sms_rmol'][:]=-.001
    if cfg.get('zero_tke'):c['sms_br'][:]=.1;c['sms_rmol'][:]=.002
    u=.5*(c['u'][0,:,:-1]+c['u'][0,:,1:]);v=.5*(c['v'][0,:-1]+c['v'][0,1:])
    c['sms_wspd']=np.maximum(np.sqrt(u*u+v*v),np.float32(.1))
    c['sms_gamu']=np.full((ny,nx),.0013,np.float32);c['sms_gamv']=np.full((ny,nx),-.0021,np.float32)
    c['sms_ustm']=np.array(c['ust']*np.float32(.7),np.float32)
    c['sms_pblh']=np.array(c['hpbl'])
    return c


def generate(exe,out,work):
    import netCDF4
    wrf=oracle_io.SWISS_DIR/'wrfout_d01_2023-01-15_12:00:00'
    with netCDF4.Dataset(wrf) as ds:
        c1f=np.array(ds.variables['C1F'][0],np.float32);c2f=np.array(ds.variables['C2F'][0],np.float32)
    saved={}
    with np.load(oracle.SOURCE) as src:
        for name,cfg in CASES.items():
            c=make_case(src,cfg,c1f,c2f)
            for n,a in c.items():
                if n=='moist':
                    for i,q in enumerate(a):saved[name+'/in/moist'+str(i)]=q
                else:saved[name+'/in/'+n]=np.asarray(a,np.float32)
            ip,op=work/(name+'.bin'),work/(name+'.out')
            oracle_io.write_input(c,ip,dict(km_opt=2,vertical=True,periodic=cfg['periodic'],isfflx=cfg['isfflx'],tke_heat_flux=cfg.get('heat',0.)))
            with ip.open('ab') as f:
                f.write(c['hpbl'].T.tobytes(order='F'));f.write(oracle_io._fo(c['dlk']))
                for n in ('sms_rmol','sms_br','sms_wspd','sms_gamu','sms_gamv'):f.write(c[n].T.tobytes(order='F'))
                f.write(c1f.tobytes());f.write(c2f.tobytes());f.write(oracle_io._fo(c['Ni']));f.write(oracle_io._fo(c['Nr']))
                f.write(c['sms_ustm'].T.tobytes(order='F'))
            subprocess.run([str(exe),str(ip),str(op)],check=True,timeout=30)
            raw=op.read_bytes();ims,ime,jms,jme,kms,kme=np.frombuffer(raw[:24],np.int32)
            ni,nj,nk=ime-ims+1,jme-jms+1,kme-kms+1;flat=np.frombuffer(raw[24:],np.float32);size=ni*nj*nk
            assert flat.size==len(OUT3)*size+4*ni*nj,(flat.size,len(OUT3)*size+4*ni*nj)
            for i,n in enumerate(OUT3):
                a=flat[i*size:(i+1)*size].reshape((ni,nk,nj),order='F').transpose(1,2,0)
                saved[name+'/out/'+n]=np.array(a[:,2:-2,2:-2])
            off=len(OUT3)*size
            for i,n in enumerate(('sms_gamu','sms_gamv','pblh','hfx')):
                a=flat[off+i*ni*nj:off+(i+1)*ni*nj].reshape((ni,nj),order='F').T
                saved[name+'/out/'+n]=np.array(a[2:-2,2:-2])
            ip.unlink();op.unlink()
    saved['meta_json']=np.frombuffer(json.dumps({'cases':CASES,'scope':'complete pristine SMS operator; no dynamics/physics trajectory yet',
        'bound':'32eps relative +32eps*maxabs(entry field) in uncoupled field-equivalent space'}).encode(),np.uint8)
    np.savez_compressed(out,**saved)


def inputs(z,key):
    from gpuwrf.dynamics.les3d_smagorinsky import Les3dConfig,Les3dInputs
    prefix=key+'/in/';c={n[len(prefix):]:z[n] for n in z.files if n.startswith(prefix)}
    c['moist']=tuple(c.pop('moist'+str(i)) for i in range(6))+(c['Ni'],c['Nr'])
    inp=Les3dInputs(**{n:c[n] for n in Les3dInputs._fields})
    cfg=Les3dConfig(km_opt=5,vertical=True,boundary='periodic' if CASES[key]['periodic'] else 'specified',
        isfflx=CASES[key]['isfflx'],tke_heat_flux=CASES[key].get('heat',0.),iqv=0,iqc=1,iqi=3,
        sms_precip_scalar_indices=(7,))
    return inp,cfg


if __name__=='__main__':generate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
