"""Pristine solve_em TKE_advance on real Swiss crop inputs, all three RK stages.

Generate a tracked compact fixture with --generate EXE OUT.npz. The driver calls
unmodified module_em/module_bc in libwrflib.a. Normal pytest needs no compiler.
Transport inputs are fixed per stage to isolate the scalar caller; this is an
operator/caller oracle, not a full CPU-WRF forecast comparison.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.dynamics.flux_advection import CoupledVelocities, couple_velocities_periodic, couple_uv_specified
from gpuwrf.runtime.les3d_km3 import les_tke_sixth_order_sc, les_tke_stage_update

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'data/fixtures/les3d-smag3d-oracle-v1.npz'


@dataclass(frozen=True)
class ScalarState:
    tke: object
    mu_total: object

    def replace(self, **kw):
        return replace(self, **kw)


def inputs(z, periodic, diff6):
    a = lambda n: jnp.asarray(z['in/real/' + n], jnp.float32)
    nz, ny, nx = a('tke').shape
    metrics = SimpleNamespace(**{n: a(n) for n in ('msftx','msfty','msfux','msfuy','msfvx','msfvy')},
        **{n: a(n)[:nz] for n in ('fnm','fnp','c1h','c2h')}, rdnw=1/a('dnw')[:nz])
    dx, dy = float(a('dx')), float(a('dy'))
    # Registry TKE and real Swiss shear/terrain structure; source from the
    # independently executed pristine diffusion oracle, not this adapter.
    state = ScalarState(a('tke'), a('mut'))
    key = 'real/km2_iso_periodic_vert_isfflx0' if periodic else 'real/km2_spec_vert_isfflx1'
    source = jnp.asarray(z['out/' + key + '/tke_tendf'][:-1, 5:5+ny, 5:5+nx], jnp.float32)
    vel = couple_velocities_periodic(a('u'), a('v'), a('mut'), c1h=metrics.c1h, c2h=metrics.c2h,
        dnw=a('dnw')[:nz], rdx=1/dx, rdy=1/dy, msfuy=metrics.msfuy, msfvx=metrics.msfvx,
        msftx=metrics.msftx, msfux=metrics.msfux, msfvy=metrics.msfvy)
    ru, rv = couple_uv_specified(a('u'), a('v'), a('mut'), c1h=metrics.c1h, c2h=metrics.c2h,
        msfuy=metrics.msfuy, msfvx=metrics.msfvx)
    # Zero eta transport is a deliberate independent operand; horizontal real
    # transport, RK source, final PD limiter, map factors and boundary signs remain active.
    vel = replace(vel, specified=not periodic, ru_full=ru, rv_full=rv, rom=jnp.zeros_like(a('w')))
    nl = SimpleNamespace(metrics=metrics, grid=SimpleNamespace(projection=SimpleNamespace(dx_m=dx,dy_m=dy),
        bc=SimpleNamespace(source='ideal' if periodic else 'wrf')), run_boundary=not periodic,
        # Match OperationalNamelist's Thompson default at the shared scalar
        # transport seam; this oracle does not select a repaired MP family.
        boundary_config=BoundaryConfig(), mp_physics=8, rk_order=3, dt_s=float(a('dt')), diff_6th_opt=diff6,
        diff_6th_factor=.12, tke_upper_bound=1000.)
    return state, source, vel, nl


def write_input(path, state, source, vel, nl):
    nz,ny,nx=state.tke.shape
    p=nl.grid.projection
    periodic=not nl.run_boundary
    m=nl.metrics
    def f3(x):
        return np.asarray(x,np.float32).transpose(2,0,1).tobytes(order='F')
    def f2(x):
        return np.asarray(x,np.float32).T.tobytes(order='F')
    if periodic:
        ru = jnp.concatenate((vel.ru,vel.ru[...,:1]),axis=-1)
        rv = jnp.concatenate((vel.rv,vel.rv[:,:1,:]),axis=-2)
    else:
        ru,rv=vel.ru_full,vel.rv_full
    with Path(path).open('wb') as f:
        f.write(np.asarray([nx,ny,nz,int(periodic),nl.diff_6th_opt],np.int32).tobytes())
        f.write(np.asarray([p.dx_m,p.dy_m,nl.dt_s,nl.tke_upper_bound,nl.diff_6th_factor],np.float32).tobytes())
        for x in (state.tke,source): f.write(f3(x))
        for x in (state.mu_total,state.mu_total): f.write(f2(x))
        for x in (ru,rv,vel.rom): f.write(f3(x))
        for n in ('msftx','msfty','msfux','msfuy','msfvx','msfvy'): f.write(f2(getattr(m,n)))
        for n in ('c1h','c2h','fnm','fnp','rdn','rdnw'):
            x = m.rdnw if n=='rdn' else getattr(m,n)
            f.write(np.append(np.asarray(x,np.float32),np.float32(0)).tobytes())


def advance(z, periodic, diff6, *, jit=False):
    state,source,vel,nl=inputs(z,periodic,diff6)
    origin=state
    if diff6: source=source+les_tke_sixth_order_sc(origin,origin.mu_total,nl)
    out=[]
    for rk in (1,2,3):
        def step(tke, old, sc):
            return les_tke_stage_update(state.replace(tke=tke),origin.replace(tke=old),tke_tend=sc,
                rk_step=rk,dt_rk=nl.dt_s/(4-rk),namelist=nl,transport_velocities=vel).tke
        state=state.replace(tke=(jax.jit(step) if jit else step)(state.tke,origin.tke,source))
        out.append(np.asarray(state.tke))
    return out


def generate(exe,outpath):
    z=dict(np.load(FIXTURE))
    saved={}
    work=Path(outpath).parent
    for periodic in (False,True):
        for diff6 in (0,2):
            tag=f'{int(periodic)}_{diff6}'
            s,sc,vel,nl=inputs(z,periodic,diff6)
            inp,op=work/(tag+'.bin'),work/(tag+'.out')
            write_input(inp,s,sc,vel,nl)
            subprocess.run([str(exe),str(inp),str(op)],check=True,timeout=30)
            nz,ny,nx=s.tke.shape
            raw=np.fromfile(op,np.float32).reshape(3,3,ny,nz,nx).transpose(0,1,3,2,4)
            for k,n in enumerate(('tke','advection','source')): saved[tag+'/'+n]=raw[:,k]
            inp.unlink(); op.unlink()
    saved['meta_json']=np.frombuffer(json.dumps({'reference':'pristine WRF module_em TKE_advance caller',
        'source_fixture':str(FIXTURE.name),'vertical_transport':'zero; horizontal Swiss winds active'}).encode(),np.uint8)
    np.savez_compressed(outpath,**saved)


if __name__=='__main__':
    assert sys.argv[1]=='--generate'
    generate(Path(sys.argv[2]),Path(sys.argv[3]))
