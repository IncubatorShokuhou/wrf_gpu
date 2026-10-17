"""F7 c05 diagnosis: save step-1 port vs pristine-WRF fields (npz) to attribute the RTHFTEN/RQVFTEN residual.
asserted GPU (wz1080 legacy+shim). argv: case_dir wrf_run_dir out_npz"""
import sys, json, time, os
from pathlib import Path
case, wrfdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
import gpuwrf
import jax, jax.numpy as jnp, numpy as np, netCDF4
PLATFORM = jax.devices()[0].platform; assert PLATFORM == 'gpu', PLATFORM
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
cfg = NestedPipelineConfig(case, out.parent / 'port-output', out.parent / 'port-proof', hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
nml, carry = bundles['d01'].namelist, carries['d01']
c0 = carry
clock = om.build_clock_base(nml)
carry = om._advance_chunk(carry, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1, cadence=int(nml.radiation_cadence_steps))
f = lambda a: np.asarray(a, np.float64)
st = carry.state
d = netCDF4.Dataset(sorted(wrfdir.glob('wrfout_d01_*'))[1]); d0 = netCDF4.Dataset(sorted(wrfdir.glob('wrfout_d01_*'))[0])
w = {k: np.asarray(d[k][0], np.float64) for k in ('RTHFTEN', 'RQVFTEN', 'RTHCUTEN', 'RQVCUTEN', 'RTHBLTEN', 'RQVBLTEN', 'RTHRATEN',
                                                  'T', 'QVAPOR', 'U', 'V', 'W', 'MU', 'MUB', 'P', 'PB', 'PH', 'PHB',
                                                  'MAPFAC_M', 'MAPFAC_MX', 'MAPFAC_MY', 'MAPFAC_UY', 'MAPFAC_VX') if k in d.variables}
w0 = {k: np.asarray(d0[k][0], np.float64) for k in ('T', 'QVAPOR', 'U', 'V', 'W', 'MU') if k in d0.variables}
attrs = {k: str(d.getncattr(k)) for k in ('USE_THETA_M', 'DX', 'DT', 'MAP_PROJ', 'CU_PHYSICS', 'BL_PBL_PHYSICS')
         if k in d.ncattrs()}
d.close(); d0.close()
np.savez_compressed(out, port_rthften=f(carry.cu_forcing[0]), port_rqvften=f(carry.cu_forcing[1]),
                    port_theta=f(st.theta), port_qv=f(st.qv), port_u=f(st.u), port_v=f(st.v), port_w=f(st.w),
                    port_mu=f(st.mu_total), port_p=f(st.p), port0_theta=f(c0.state.theta), port0_qv=f(c0.state.qv),
                    port0_mu=f(c0.state.mu_total),
                    msfty=f(nml.metrics.msfty), msftx=f(getattr(nml.metrics, 'msftx', nml.metrics.msfty)),
                    **{'wrf_' + k: v for k, v in w.items()}, **{'wrf0_' + k: v for k, v in w0.items()})
print(json.dumps(dict(platform=PLATFORM, attrs=attrs, out=str(out))))
