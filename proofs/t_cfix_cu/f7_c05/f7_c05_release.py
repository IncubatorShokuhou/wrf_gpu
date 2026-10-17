"""F7 c05 end-to-end pristine oracle on the RELEASE path (RTX 5090, asserted GPU): steps 1..3 metrics exactly as
f7_c05_compare.py (pre-registered gate on step 1) + step-1 fields npz for attribution.
argv: case_dir wrf_run_dir out_json out_npz nsteps"""
import sys, json, time, os
from pathlib import Path
case, wrfdir, out, out_npz = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); nsteps = int(sys.argv[5])
import gpuwrf  # release defaults
import jax, jax.numpy as jnp, numpy as np, netCDF4
PLATFORM = jax.devices()[0].platform; assert PLATFORM == 'gpu', PLATFORM
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
t0 = time.time()
cfg = NestedPipelineConfig(case, out.parent / 'port-output', out.parent / 'port-proof', hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
nml, carry = bundles['d01'].namelist, carries['d01']
assert int(nml.cu_physics) == 5 and carry.cu_forcing is not None
clock = om.build_clock_base(nml); cad = int(nml.radiation_cadence_steps)
rec = dict(platform=PLATFORM, device=str(jax.devices()[0]), fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS', '<unset=1>'),
           case=str(case), load_s=time.time() - t0, dt=float(nml.dt_s), cadence=cad, steps={})
rvrd = 461.6 / 287.0
for k in range(1, nsteps + 1):
    t = time.time()
    carry = om._advance_chunk(carry, nml, jnp.asarray(k, jnp.int32), clock, n_steps=1, cadence=cad)
    th, qv = (np.asarray(a, np.float64) for a in carry.cu_forcing)
    d = netCDF4.Dataset(sorted(wrfdir.glob('wrfout_d01_*'))[k])
    wth = np.asarray(d['RTHFTEN'][0], np.float64); wqv = np.asarray(d['RQVFTEN'][0], np.float64)
    theta = np.asarray(carry.state.theta, np.float64); q = np.asarray(carry.state.qv, np.float64)
    if k == 1:
        np.savez_compressed(out_npz, port_rthften=th, port_rqvften=qv, port_theta=theta, port_qv=q,
                            port_w=np.asarray(carry.state.w, np.float64), port_mu=np.asarray(carry.state.mu_total, np.float64),
                            wrf_RTHFTEN=wth, wrf_RQVFTEN=wqv, wrf_T=np.asarray(d['T'][0], np.float64),
                            wrf_QVAPOR=np.asarray(d['QVAPOR'][0], np.float64), wrf_W=np.asarray(d['W'][0], np.float64),
                            wrf_MU=np.asarray(d['MU'][0], np.float64), wrf_MUB=np.asarray(d['MUB'][0], np.float64),
                            wrf_RTHCUTEN=np.asarray(d['RTHCUTEN'][0], np.float64), msfty=np.asarray(nml.metrics.msfty, np.float64))
    d.close()
    thp = theta / (1 + rvrd * q)
    th_moist = th * theta / thp + rvrd * thp * qv  # mutant: no dry conversion (inverts phy_prep_part2)
    def stats(a, b, sl):
        a, b = a[sl], b[sl]; e = a - b
        return dict(rms_err=float(np.sqrt(np.mean(e**2))), rms_ref=float(np.sqrt(np.mean(b**2))),
                    rel=float(np.sqrt(np.mean(e**2)) / max(np.sqrt(np.mean(b**2)), 1e-30)),
                    max_err=float(np.abs(e).max()), max_ref=float(np.abs(b).max()))
    full = (slice(None),) * 3; inner = (slice(None), slice(5, -5), slice(5, -5))
    rec['steps'][k] = dict(wall_s=time.time() - t,
        rthften_full=stats(th, wth, full), rthften_inner=stats(th, wth, inner),
        rqvften_full=stats(qv, wqv, full), rqvften_inner=stats(qv, wqv, inner),
        mutant_nodry_rthften_inner=stats(th_moist, wth, inner),
        mutant_zero_rthften_inner=stats(0 * th, wth, inner), mutant_zero_rqvften_inner=stats(0 * qv, wqv, inner))
    print(k, json.dumps({n: v['rel'] for n, v in rec['steps'][k].items() if isinstance(v, dict)}), flush=True)
s1 = rec['steps'][1]
ratio = s1['mutant_nodry_rthften_inner']['rel'] / s1['rthften_inner']['rel']
rec['gate_step1'] = dict(rthften_rel=s1['rthften_inner']['rel'], rqvften_rel=s1['rqvften_inner']['rel'],
    accuracy_pass=s1['rthften_inner']['rel'] <= 0.10 and s1['rqvften_inner']['rel'] <= 0.10,
    no_dry_ratio=ratio, zero_ratio_th=1.0 / s1['rthften_inner']['rel'], zero_ratio_qv=1.0 / s1['rqvften_inner']['rel'],
    discrimination_pass=ratio >= 3.0 and 1.0 / s1['rthften_inner']['rel'] >= 3.0 and 1.0 / s1['rqvften_inner']['rel'] >= 3.0)
out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec['gate_step1']))
