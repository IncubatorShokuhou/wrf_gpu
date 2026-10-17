"""F7 oracle: port held RTHFTEN/RQVFTEN after steps 1..N on real c05 (G3, Swiss Jan) vs pristine serial WRF history.
asserted GPU (with_gpu_lock). argv: case_dir wrf_run_dir out_json nsteps [eager]; eager = jax.disable_jit (no whole-step compile)."""
import sys, json, time, os
from pathlib import Path
case, wrfdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); nsteps = int(sys.argv[4]); eager = len(sys.argv) > 5 and sys.argv[5] == 'eager'
import gpuwrf  # release defaults
import jax, jax.numpy as jnp, numpy as np, netCDF4
PLATFORM = jax.devices()[0].platform; assert PLATFORM == 'gpu', PLATFORM
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
t0 = time.time()
cfg = NestedPipelineConfig(case, out.parent / 'port-output', out.parent / 'port-proof', hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
nml, carry = bundles['d01'].namelist, carries['d01']
assert int(nml.cu_physics) == 5 and carry.cu_forcing is not None, (nml.cu_physics, carry.cu_forcing is None)
clock = om.build_clock_base(nml)
cad = int(nml.radiation_cadence_steps)
rec = dict(platform=PLATFORM, device=str(jax.devices()[0]), eager=eager, fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS', '<unset=1>'), case=str(case), load_s=time.time()-t0, dt=float(nml.dt_s), cadence=cad, steps={})
nc = lambda k: netCDF4.Dataset(sorted(wrfdir.glob('wrfout_d01_*'))[k])
for k in range(1, nsteps + 1):
    t = time.time()
    if eager:
        with jax.disable_jit():
            carry = om._advance_chunk(carry, nml, jnp.asarray(k, jnp.int32), clock, n_steps=1, cadence=cad)
    else:
        carry = om._advance_chunk(carry, nml, jnp.asarray(k, jnp.int32), clock, n_steps=1, cadence=cad)
    th, qv = (np.asarray(a, np.float64) for a in carry.cu_forcing)
    d = nc(k)
    wth = np.asarray(d['RTHFTEN'][0], np.float64); wqv = np.asarray(d['RQVFTEN'][0], np.float64)
    theta = np.asarray(carry.state.theta, np.float64); q = np.asarray(carry.state.qv, np.float64)
    wT = np.asarray(d['T'][0], np.float64) + 300.0; wQ = np.asarray(d['QVAPOR'][0], np.float64)
    d.close()
    rvrd = 461.6 / 287.0
    thp = theta / (1 + rvrd * q)
    th_moist = th * theta / thp + rvrd * thp * qv  # mutant M1: no dry conversion (invert phy_prep_part2)
    def stats(a, b, sl):
        a, b = a[sl], b[sl]
        e = a - b
        return dict(rms_err=float(np.sqrt(np.mean(e**2))), rms_ref=float(np.sqrt(np.mean(b**2))),
                    rel=float(np.sqrt(np.mean(e**2)) / max(np.sqrt(np.mean(b**2)), 1e-30)),
                    max_err=float(np.abs(e).max()), max_ref=float(np.abs(b).max()))
    full = (slice(None),) * 3; inner = (slice(None), slice(5, -5), slice(5, -5))
    rec['steps'][k] = dict(
        wall_s=time.time() - t,
        state_theta_rel=stats(theta, wT, inner), state_qv_rel=stats(q, wQ, inner),
        rthften_full=stats(th, wth, full), rthften_inner=stats(th, wth, inner),
        rqvften_full=stats(qv, wqv, full), rqvften_inner=stats(qv, wqv, inner),
        mutant_nodry_rthften_inner=stats(th_moist, wth, inner),
        mutant_zero_rthften_inner=stats(0 * th, wth, inner),
        mutant_zero_rqvften_inner=stats(0 * qv, wqv, inner))
    print(k, json.dumps({n: v['rel'] for n, v in rec['steps'][k].items() if isinstance(v, dict)}), flush=True)
out.write_text(json.dumps(rec, indent=1))
