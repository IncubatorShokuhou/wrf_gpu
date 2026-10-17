"""Three own steps of SMS source + pristine TKE_advance (nine RK stages).

Real Swiss structure at dx=600m, frozen winds/mass/moisture/surface forcing;
TKE and held gamma evolve. This is a caller oracle, not a forecast comparison.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,json,subprocess
import jax
import jax.numpy as jnp
import numpy as np
from proofs.s1_km5 import full_oracle as full
from proofs.o2_km2 import stage_oracle as scalar
from proofs.o1_smag3d import oracle_io
from gpuwrf.dynamics.les3d_smagorinsky import _Mem,les3d_smagorinsky_memory
from gpuwrf.runtime.les3d_km3 import les_tke_stage_update
from gpuwrf.runtime import operational_mode as _runtime  # Warm import-time tables before eval_shape.

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'data/fixtures/sms3dtke-full-wrf-v1.npz'
KEY='real_mixed'


def scalar_inputs(z):
    # Reuse the qualified scalar caller's explicit REAL transport operands.
    shim={'in/real/'+n:z[KEY+'/in/'+n] for n in
          ('tke','mut','u','v','w','msftx','msfty','msfux','msfuy','msfvx','msfvy',
           'c1h','c2h','fnm','fnp','dnw','dx','dy','dt')}
    nz,ny,nx=shim['in/real/tke'].shape
    source=np.zeros((nz+1,ny+11,nx+11),np.float32)
    source[:nz,5:5+ny,5:5+nx]=z[KEY+'/out/tke_tendf'][:nz,3:3+ny,3:3+nx]
    shim['out/real/km2_spec_vert_isfflx1/tke_tendf']=source
    state,src,vel,nl=scalar.inputs(shim,False,0)
    nl.km_opt=5
    nl.mp_physics=8  # Static transport dispatch field added by the numbers lane.
    return state,src,vel,nl


def generate(full_exe,stage_exe,out,work):
    saved={}
    with np.load(FIXTURE) as z:
        state,_,vel,nl=scalar_inputs(z)
        p=KEY+'/in/';c={n[len(p):]:np.array(z[n]) for n in z.files if n.startswith(p)}
    c['moist']=[c.pop('moist'+str(i)) for i in range(6)]
    nz,ny,nx=state.tke.shape; mem=_Mem(nx,ny,nz,jnp.float32)
    for step in range(3):
        c['tke']=np.asarray(state.tke)
        ip,op=work/'sms.bin',work/'sms.out'
        oracle_io.write_input(c,ip,dict(km_opt=2,vertical=True,periodic=False,isfflx=1))
        with ip.open('ab') as f:
            f.write(c['hpbl'].T.tobytes(order='F'));f.write(oracle_io._fo(c['dlk']))
            for n in ('sms_rmol','sms_br','sms_wspd','sms_gamu','sms_gamv'):f.write(c[n].T.tobytes(order='F'))
            f.write(c['c1f'].tobytes());f.write(c['c2f'].tobytes())
            for n in ('Ni','Nr'):f.write(oracle_io._fo(c[n]))
            f.write(c['sms_ustm'].T.tobytes(order='F'))
        subprocess.run([str(full_exe),str(ip),str(op)],check=True,timeout=30)
        raw=op.read_bytes();ims,ime,jms,jme,kms,kme=np.frombuffer(raw[:24],np.int32)
        ni,nj,nk=ime-ims+1,jme-jms+1,kme-kms+1;size=ni*nj*nk
        flat=np.frombuffer(raw[24:],np.float32)
        src=flat[4*size:5*size].reshape(ni,nk,nj,order='F').transpose(1,2,0)[:nz,5:5+ny,5:5+nx].copy()
        saved[f'{step}/source']=src
        off=len(full.OUT3)*size
        for i,n in enumerate(('sms_gamu','sms_gamv','pblh','hfx')):
            a=flat[off+i*ni*nj:off+(i+1)*ni*nj].reshape(ni,nj,order='F').T[5:5+ny,5:5+nx].copy()
            saved[f'{step}/'+n]=a;c['hpbl' if n=='pblh' else n]=a
        scalar.write_input(ip,state,src,vel,nl)
        subprocess.run([str(stage_exe),str(ip),str(op)],check=True,timeout=30)
        vals=np.fromfile(op,np.float32).reshape(3,3,ny,nz,nx).transpose(0,1,3,2,4)
        saved[f'{step}/tke']=vals[:,0]
        state=state.replace(tke=jnp.asarray(vals[-1,0]))
        ip.unlink();op.unlink()
    saved['meta_json']=np.frombuffer(json.dumps(dict(reference='unmodified pristine module_diffusion_em + module_em',
        own_steps=3,RK_stages=9,dx_m=600,transport='real horizontal; zero eta; frozen wind/mass/moisture/surface',
        evolving='TKE and GAMU/GAMV',bound='32eps32 relative +32eps32 maxabs reference')).encode(),np.uint8)
    np.savez_compressed(out,**saved)


def advance():
    with np.load(FIXTURE) as z:
        inp,cfg=full.inputs(z,KEY);state,initial_source,vel,nl=scalar_inputs(z)
    for rk in (1,2,3):
        jax.eval_shape(lambda:les_tke_stage_update(state,state,tke_tend=initial_source,
            rk_step=rk,dt_rk=nl.dt_s/(4-rk),namelist=nl,transport_velocities=vel).tke)
    print('CALLER_SHAPE_PREFLIGHT_PASS_3',flush=True)
    with np.load(ROOT/'data/fixtures/sms3dtke-stages-wrf-v1.npz') as z:
        refs=np.stack([z[f'{s}/tke'] for s in range(3)])
    eps=32*np.finfo(np.float32).eps;atol=eps*abs(refs).max()
    nz,ny,nx=state.tke.shape;mem=_Mem(nx,ny,nz,jnp.float32)
    def bundle(x):
        r=les3d_smagorinsky_memory(x,cfg)
        return mem.interior(r.tke_tendf),r.sms_updates
    fn=jax.jit(bundle);actual=[];sources=[]
    for step in range(3):
        origin=state
        source,updates=fn(inp._replace(tke=state.tke))
        sources.append(np.asarray(source))
        print('SMS_SOURCE_READY',step+1,flush=True)
        inp=inp._replace(sms_gamu=updates['sms_gamu'],sms_gamv=updates['sms_gamv'],
                         sms_pblh=updates['pblh'],hfx=updates['hfx'])
        stages=[]
        for rk in (1,2,3):
            state=les_tke_stage_update(state,origin,tke_tend=source,rk_step=rk,
                dt_rk=nl.dt_s/(4-rk),namelist=nl,transport_velocities=vel)
            stages.append(np.asarray(state.tke))
            actual_stage=stages[-1];reference_stage=refs[step,rk-1]
            assert np.isfinite(actual_stage).all() and np.isfinite(reference_stage).all()
            print('TKE_STAGE',step+1,rk,'max_error',float(abs(actual_stage-reference_stage).max()),flush=True)
            np.testing.assert_allclose(actual_stage,reference_stage,rtol=eps,atol=atol,
                                       err_msg=f'ownstep{step+1}/RK{rk}')
        actual.append(stages)
    return np.asarray(actual),sources


if __name__=='__main__':generate(*(Path(x) for x in sys.argv[1:]))
