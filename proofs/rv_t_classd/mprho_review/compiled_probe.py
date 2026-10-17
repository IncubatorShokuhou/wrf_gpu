"""Independent CPU compilation of the unchanged helper newly wired by ab554.

Reference is freshly compiled pristine REAL4 statements, not port output.
This is a boundary probe, not a forecast or GPU performance measurement.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'snapshot'
sys.path[:0] = [str(ROOT/'scripts/v025'), str(ROOT/'src'), str(ROOT/'tests')]
import cpu_guard
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_t_wsmdz_wsm_callers import _prep_case
from gpuwrf.coupling.scan_adapters import _wrf_moist_prep_den_delz

assert jax.devices()[0].platform == 'cpu'
ref = json.loads((ROOT.parent/'RV26_fresh_oracle.json').read_text())
phi = np.asarray(ref['inputs']['ph'], np.float32) + np.asarray(ref['inputs']['phb'], np.float32)
z = phi / np.float32(9.81)
dz_ref = np.asarray(ref['dz8w'], np.float64)
rho_ref = np.asarray(ref['rho'], np.float64)
dz_bound = np.spacing(abs(z[1:])) + np.spacing(abs(z[:-1])) + np.spacing(abs(dz_ref).astype(np.float32))
ulp = np.spacing(abs(phi)).astype(np.float64)
rho_bound = (ulp[1:]+ulp[:-1])/abs(np.diff(phi.astype(np.float64))) + 8*np.finfo(np.float32).eps
results=[]
for dtype in (jnp.float32, jnp.float64):
    grid, state = _prep_case(2, 3, dtype, 18)
    assert state.theta.dtype == np.dtype(dtype)
    assert state.ph_total.dtype == state.mu_total.dtype == np.dtype(dtype)
    f = jax.jit(lambda s: _wrf_moist_prep_den_delz(s, grid, dtype))
    executable = f.lower(state).compile()
    rho, dz = jax.block_until_ready(executable(state))
    assert rho.dtype == dz.dtype == np.dtype(dtype)
    rr, dd = np.asarray(rho).astype(np.float64), np.asarray(dz).astype(np.float64)
    assert rr.shape == dd.shape == (3, 2, 3)
    assert np.isfinite(rr).all() and np.isfinite(dd).all()
    rel = abs(rr/rho_ref[:,None,None]-1)
    err = abs(dd-dz_ref[:,None,None])
    assert np.all(rel <= rho_bound[:,None,None]), (rel, rho_bound)
    assert np.all(err <= dz_bound[:,None,None]), (err, dz_bound)
    memory=executable.memory_analysis()
    hlo=executable.as_text()
    arithmetic64=re.findall(r'f64\[[^\]]*\](?:\{[^}]*\})?\s+(?:add|subtract|multiply|divide|log|power|exponential)\(',hlo)
    assert not arithmetic64, arithmetic64
    results.append(dict(dtype=str(rho.dtype), actual_state_dtype=str(state.theta.dtype),
        rho_relative_error_max_by_level=rel.max(axis=(1,2)).tolist(),
        rho_relative_bound=rho_bound.tolist(), dz_absolute_error_max_by_level=err.max(axis=(1,2)).tolist(),
        dz_absolute_bound=dz_bound.tolist(), memory_analysis=str(memory), cost_analysis=executable.cost_analysis(),
        optimized_cpu_hlo_f64_arithmetic_count=len(arithmetic64)))

# Stop at each real adapter's native-kernel call during tracing, then compile
# its returned operand expressions. No host callback, fake physical kernel
# output, or source mutation is needed to observe the production call prefix.
class Boundary(Exception):
    pass
def boundary(rho, dz):
    raise Boundary(rho, dz)
callers=[]
for mp in (17,18,19,21,22,5,15,7):
    for dtype in (jnp.float32,jnp.float64):
        if mp in (5,15,7) and dtype==jnp.float64:
            continue
        grid,state=_prep_case(2,3,dtype,mp)
        assert state.theta.dtype == state.ph_total.dtype == np.dtype(dtype)
        with pytest.MonkeyPatch.context() as patch:
            if mp in (17,18,19,21,22):
                from gpuwrf.physics.nssl2mom import adapter
                patch.setattr(adapter,'nssl2mom_column',lambda fields,*a,**k: boundary(fields['rho'],fields['dz']))
                kwargs={19:dict(variant=19),21:dict(variant=21),22:dict(hail_enabled=False)}.get(mp,{})
                run=lambda s: adapter.nssl2mom_adapter(s,18.,grid,first_step=True,**kwargs)
            elif mp in (5,15):
                from gpuwrf.physics.ferrier import adapter
                patch.setattr(adapter,'native_column',lambda th,qv,qc,qr,qi,p,pii,rho,dz,*a,**k: boundary(rho,dz))
                run=lambda s: adapter.ferrier_adapter(s,18.,grid,option=mp)
            else:
                from gpuwrf.physics.goddard4ice import native
                from gpuwrf.coupling.goddard4ice_adapter import goddard4ice_adapter
                patch.setattr(native,'native_column',lambda th,qv,qc,qr,qi,qs,qg,qh,rho,exner,p,dz,*a,**k: boundary(rho,dz))
                run=lambda s: goddard4ice_adapter(s,18.,grid)
            def prefix(s):
                try:
                    run(s)
                except Boundary as hit:
                    return hit.args
                raise AssertionError('adapter failed to reach native kernel')
            executable=jax.jit(prefix).lower(state).compile()
            rho,dz=jax.block_until_ready(executable(state))
            assert rho.dtype == dz.dtype == np.dtype(dtype)
            def columns(a):
                a=np.asarray(a,np.float64)
                # NSSL ABI is (..., nz); Ferrier/Goddard ABI is (nz, ny, nx).
                if mp not in (17,18,19,21,22):
                    a=np.moveaxis(a,0,-1)
                return a.reshape(-1,3)
            rr,dd=columns(rho),columns(dz)
            assert rr.shape==dd.shape==(6,3)
            rel=abs(rr/rho_ref-1);err=abs(dd-dz_ref)
            assert np.isfinite(rr).all() and np.isfinite(dd).all()
            assert np.all(rel<=rho_bound) and np.all(err<=dz_bound),(mp,dtype,rel,err)
            callers.append(dict(mp=mp,dtype=str(rho.dtype),native_boundary_compiled=True,
                rho_relative_error_max_by_level=rel.max(axis=0).tolist(),dz_absolute_error_max_by_level=err.max(axis=0).tolist()))
out=dict(platform='cpu', gpu_executed=False, reference='fresh pristine REAL4 7 statements; ALB finite synthetic operand',
         requested_and_actual_dtypes_verified=True, tolerance='unchanged input-ULP bounds', results=results,compiled_adapter_prefixes=callers)
(ROOT.parent/'RV27_compiled_probe.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
