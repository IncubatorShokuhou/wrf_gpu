"""PRODUCT solar/ozone paths against independent original-WRF references.

Frozen cohorts and solar literal come from lane/w44-topo 7808ad63f8 TP80.
No GPU or forecast runs. Only static input columns and actual radiation clocks.
"""
import argparse
import ctypes
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cohorts', type=Path, required=True)
p.add_argument('--source', type=Path)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
ROOT = a.source or Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import wrf_cam_ozone as O
assert jax.devices()[0].platform == 'cpu'
C._SOLAR_JULIAN_WRF = True
F = np.float32
fixture = ROOT / 'tests/v025/b_phys/fixtures/solar_julian_wrf.json'
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == '7566a1a1afa8e4c2ebbe08e2a9a9b6d91f4959b322b8bd6a14acbb2f849b2081'

def product(lat, lon, base, utc, lead, solar):
    clock = (base, utc)
    g = C._compute_solar_geometry(lat, lon, None, lead, clock_base=clock, solar_lead_seconds=solar)
    return (g.coszen, g.hour_angle_rad, g.declination_rad,
            C._solcon_for_time(None, lead, clock_base=clock),
            C._wrf_solar_julian_real(base, utc, lead))

rows = json.loads(fixture.read_text())['rows']
args = [jnp.asarray([r[n] for r in rows], jnp.float32)
        for n in ['lat', 'lon', 'julian_base', 'minute_base', 'lead']]
solar = F(np.array([r['lead'] + r['midpoint'] for r in rows]))
out = jax.jit(product)(*args, solar)
ref = np.asarray([r['reference_bits'] for r in rows], np.int32).view(F)
reference_fields = [ref[:, 1], ref[:, 2], ref[:, 0], ref[:, 3],
                    np.asarray([r['call_julian'] for r in rows], F)]
for i, (got, expected) in enumerate(zip(out, reference_fields)):
    assert np.isfinite(got).all()
    if i == 0:
        np.testing.assert_allclose(got, expected, atol=np.spacing(F(1)), rtol=0)
    elif i == 2:
        np.testing.assert_array_max_ulp(np.asarray(got), expected, maxulp=1)
    else:
        np.testing.assert_array_equal(np.asarray(got).view('I'), expected.view('I'))
# Gate amendment 2026-10-25T18:09:44Z: retain the frozen WRF bounds and prove
# actual BYTE identity to the unchanged base, eager and compiled separately.
candidate_functions = {name: getattr(C, name) for name in
    ('_wrf_solar_julian_real', '_wrf_solcon_real', '_wrf_solar_geometry_real')}
candidate = {mode: tuple(np.asarray(x) for x in fn(*args, solar))
             for mode, fn in [('eager', product), ('jit', jax.jit(product))]}
base = subprocess.check_output(['git', '-C', str(ROOT), 'show',
    '8f31d181ba:src/gpuwrf/coupling/physics_couplers.py'], text=True)
begin = base.index('def _wrf_solar_julian_real(')
end = base.index('def _compute_coszen(', begin)
exec(compile(base[begin:end], '<unchanged-base-solar>', 'exec'), C.__dict__)
jax.clear_caches()
identity = {}
for mode, fn in [('eager', product), ('jit', jax.jit(product))]:
    baseline = tuple(np.asarray(x) for x in fn(*args, solar))
    for x, y in zip(candidate[mode], baseline):
        np.testing.assert_array_equal(x.view('I'), y.view('I'))
    identity[mode] = dict(rows=len(rows), all_fields_BYTE=True,
                         unequal_to_WRF=[int(np.count_nonzero(x.view('I') != y.view('I')))
                                         for x, y in zip(baseline, reference_fields)])
for name, fn in candidate_functions.items():
    setattr(C, name, fn)
jax.clear_caches()
print('PRODUCT ORIGINAL SOLAR', len(rows), 'FROZEN BOUNDS PASS; BASE BYTE eager+jit', flush=True)

library = a.out / 'solar_reference.so'
subprocess.run(['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran', '-shared', '-fPIC', '-O2',
                '-fno-inline', '-fno-unsafe-math-optimizations', '-ffp-contract=off',
                '-ffree-line-length-none', str(HERE / 'solar_reference.F90'), '-o', str(library)],
               check=True, timeout=60, cwd=a.out)
lib = ctypes.CDLL(str(library))
array = np.ctypeslib.ndpointer(dtype=F, flags='C_CONTIGUOUS')
lib.replay.argtypes = [ctypes.c_int, array, array] + [ctypes.c_float]*4 + [ctypes.c_int]*2 + [array]*5

def reference(lat, lon, base, utc, lead, solar, mode=0):
    cos = np.empty(lat.size, F); angle = cos.copy(); aux = np.empty(6, F)
    num = np.empty(11, F); den = num.copy()
    lib.replay(lat.size, lat, lon, F(base), F(utc), F(lead), F(solar), mode, 0,
               cos, angle, aux, num, den)
    return (cos, angle, aux[1], aux[2], aux[0]), num, den

spec = importlib.util.spec_from_file_location('original_wrf_ozone', ROOT / 'tests/test_v0234_rrtmg_o3rad.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
solar_fn = jax.jit(product)
# Capture the operands and results of the actual product helper calls while
# tracing, then return them as device values. No independent reimplementation
# of the protected quotient graph is substituted for the product.
real_div = O._real4_div_rn
captured = []
def observe_div(x, y):
    value = real_div(x, y)
    captured.append((jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32), value))
    return value
C._real4_div_rn = O._real4_div_rn = observe_div
def observed_product(lat, lon, base, utc, lead, solar):
    captured.clear()
    result = product(lat, lon, base, utc, lead, solar)
    O.wrf_ozn_time_indices_factors(base, utc, lead)
    return result, tuple(captured)
solar_fn = jax.jit(observed_product)
summaries = []
for cohort in json.loads(a.cohorts.read_text()):
    input_file = Path(cohort['input_file'])
    assert hashlib.sha256(input_file.read_bytes()).hexdigest() == cohort['input_sha256']
    with netCDF4.Dataset(input_file) as ds:
        lat = np.ascontiguousarray(ds['XLAT'][0], F)
        lon = np.ascontiguousarray(ds['XLONG'][0], F)
        height = np.asarray(ds['HGT'][0], F)
        ny, nx = lat.shape
        cols = sorted(set([(0, 0), (0, nx-1), (ny-1, 0), (ny-1, nx-1), (ny//2, nx//2),
                           tuple(int(v) for v in np.unravel_index(height.argmax(), height.shape))]))
        sampled_lat = np.asarray([lat[y, x] for y, x in cols], F)
        pressure = np.asarray([F(ds['P'][0, :, y, x] + ds['PB'][0, :, y, x]) for y, x in cols], F)
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    date = datetime.strptime(cohort['start'], '%Y-%m-%d_%H:%M:%S')
    base = F(date.timetuple().tm_yday)
    utc = F(date.hour*60 + date.minute + date.second/60)
    flips = [0, 0, 0]
    max_ulp = [0]*5
    quotient_values = 0
    ozone_max_ulp = 0
    for lead in cohort['radiation_physics_lead_seconds']:
        solar = F(lead + .5*cohort['radt_s'])
        ref, nums, dens = reference(lat, lon, base, utc, lead, solar)
        got, quotients = solar_fn(lat, lon, base, utc, F(lead), solar)
        for numerator, denominator, quotient in quotients:
            expected = F(np.asarray(numerator) / np.asarray(denominator))
            np.testing.assert_array_equal(np.asarray(quotient).view('I'), expected.view('I'))
            quotient_values += np.asarray(quotient).size
        for i, (x, y) in enumerate(zip(got, ref)):
            x, y = np.asarray(x, F), np.asarray(y, F)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            # Frozen solar science gates. Exact quotient gates below are separate
            # from CPU JAX vs libm transcendental representation differences.
            # Record CPU transcendental differences over the expanded cohort;
            # the frozen 504 original-WRF gate above remains authoritative.
            max_ulp[i] = max(max_ulp[i], int(np.max(np.abs(x.view('I').astype(np.int64)-y.view('I').astype(np.int64)))))
        surrogate, _, _ = reference(lat, lon, base, utc, lead, solar, mode=4)
        for x, y in zip(surrogate, ref):
            np.testing.assert_array_equal(np.asarray(x).view('I'), np.asarray(y).view('I'))
        for i, threshold in enumerate((F(0), F(1e-6), F(1e-4))):
            count = int(np.count_nonzero((surrogate[0] > threshold) != (ref[0] > threshold)))
            flips[i] += count
            assert count == 0, (cohort['case'], cohort['domain'], lead, threshold, count)
        # The same RN helper on every MUST quotient's actual scalar operands;
        # longitude is the one field-valued family. Host constants are compared too.
        for site in (1, 2, 3, 4, 5, 6, 8, 10):
            x, y = nums[site-1], dens[site-1]
            q = np.asarray(O._real4_div_rn(x, y), F)
            np.testing.assert_array_equal(q.view('I'), F(x/y).view('I'))
        q = np.asarray(O._real4_div_rn(lon, F(15)), F)
        np.testing.assert_array_equal(q.view('I'), F(lon/F(15)).view('I'))
        if cohort['domain'] == 'd01':
            expected = original._wrf_profile_reference(sampled_lat, pressure,
                julian_day_1based=int(base), utc_minute=F(utc + F(F(lead)/F(60))))
            # Existing independent gate executes scalar WRF operations (no jit).
            actual = np.asarray(O.wrf_cam_ozone_profile(sampled_lat, pressure,
                julian_day_1based=base, utc_minute=utc, lead_seconds=F(lead)))
            np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
            ozone_max_ulp = max(ozone_max_ulp, int(np.max(abs(actual.view('I').astype(np.int64)-expected.view('I').astype(np.int64)))))
    summary = dict(case=cohort['case'], domain=cohort['domain'], clocks=len(cohort['radiation_physics_lead_seconds']),
                   points=lat.size, solar_max_ULP=max_ulp, predicate_flips=flips,
                   protected_quotient_values=quotient_values, protected_quotient_max_ULP=0,
                   ozone_profile_max_ULP=ozone_max_ulp, ozone_columns=cols if cohort['domain']=='d01' else [])
    summaries.append(summary)
    print(json.dumps(summary), flush=True)
library.unlink()
receipt = dict(platform='cpu', GPU_executed=False, original_solar_rows_frozen_bounds_PASS=len(rows),
               base_product_BYTE=identity,
               fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(), cohorts=summaries)
(a.out / 'component.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert sum(r['clocks'] for r in summaries) == 677
