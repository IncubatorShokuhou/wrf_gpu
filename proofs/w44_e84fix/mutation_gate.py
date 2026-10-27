"""Fourteen per-family deletion kills plus two numeric semantic mutants.

Source mutation is in memory; every by-name product alias is patched (E228).
Only genuine AssertionError counts as a kill; setup errors propagate (E202).
Deletion gates inspect CUDA-target raw-HLO provenance, not XLA:CPU division.
Semantic gates execute the product and compare values to the unchanged arm.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import wrf_cam_ozone as O
from lowering_audit import audit, C as CFILE, O as OFILE, PATTERNS, HOST

assert jax.devices()[0].platform == 'cpu'
C._SOLAR_JULIAN_WRF = True
originals = {module: {name: getattr(module, name) for name in names} for module, names in [
    (C, ('_wrf_solar_julian_real', '_wrf_solcon_real', '_wrf_solar_geometry_real')),
    (O, ('wrf_ozn_time_indices_factors',))]}
sources = {CFILE: (a.source / CFILE).read_text(), OFILE: (a.source / OFILE).read_text()}


def restore():
    for module, functions in originals.items():
        for name, fn in functions.items():
            setattr(module, name, fn)
    jax.clear_caches()


def install(file, source):
    module = C if file == CFILE else O
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in originals[module]:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(a.source / file), 'exec'), module.__dict__)
    jax.clear_caches()


def product(lat, lon, base, utc, lead, midpoint):
    geometry = C._compute_solar_geometry(lat, lon, None, lead,
        clock_base=(base, utc), solar_lead_seconds=midpoint)
    return (geometry, C._solcon_for_time(None, lead, clock_base=(base, utc)),
            O.wrf_ozn_time_indices_factors(base, utc, lead))


def lower_gate():
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    coords = jax.ShapeDtypeStruct((16,), jnp.float32)
    lowered = jax.jit(product).trace(coords, coords, scalar, scalar, scalar, scalar).lower(lowering_platforms=('cuda',))
    module = lowered.compiler_ir('hlo')
    return audit(module.as_serialized_hlo_module_proto(), module.as_hlo_text(), a.source)


baseline = lower_gate()
print('BASELINE', baseline['family_counts'], flush=True)
mutations = {
    'solar_lead60': ('_real4_div_rn(F(lead_seconds), F(60))', 'F(lead_seconds) / jax.lax.optimization_barrier(F(60))'),
    'solar_day1440': ('_real4_div_rn(absolute, F(1440))', 'absolute / jax.lax.optimization_barrier(F(1440))'),
    'solar_degrad': (PATTERNS['solar_degrad'][1], 'div(F(3.141592653589793), F(180))'),
    'solar_dpd': (PATTERNS['solar_dpd'][1], 'div(F(360), F(365))'),
    'solar_eot365': (PATTERNS['solar_eot365'][1], 'div(mul(F(6.2831853071795862), sub(call_julian, F(1))), F(365))'),
    'solar_midpoint60': (PATTERNS['solar_midpoint60'][1], 'div(F(solar_lead_seconds), F(60))'),
    'solar_local60': (PATTERNS['solar_local60'][1], 'div(xt24, F(60))'),
    'solar_lon15': (PATTERNS['solar_lon15'][1], 'div(jnp.asarray(lon, F), F(15))'),
    'solar_solcon365': ('def div(a, b): return barrier(_real4_div_rn(a, b))', 'def div(a, b): return barrier(a / barrier(b))'),
    'ozone_lead60': (PATTERNS['ozone_lead60'][1], 'jnp.asarray(lead_seconds, dtype=dtype) / jnp.asarray(60.0, dtype=dtype)'),
    'ozone_cyclic1early': (PATTERNS['ozone_cyclic1early'][1], '(cday_plus - day) / delta_cyclic'),
    'ozone_cyclic2early': (PATTERNS['ozone_cyclic2early'][1], '(day + jnp.asarray(365.0, dtype=dtype) - cday_minus) / delta_cyclic'),
    'ozone_linear1': (PATTERNS['ozone_linear1'][1], '(cday_plus - day) / delta'),
    'ozone_linear2': (PATTERNS['ozone_linear2'][1], '(day - cday_minus) / delta'),
}
results = []
for family, (old, new) in mutations.items():
    restore()
    file = PATTERNS[family][0]
    assert old in sources[file], ('mutant setup missing expression', family)
    install(file, sources[file].replace(old, new))
    try:
        lower_gate()
    except AssertionError as exc:
        results.append(dict(family=family, killed=True, exception='AssertionError', reason=str(exc)[:600]))
        print('DELETION KILLED', family, str(exc)[:200], flush=True)
    else:
        raise AssertionError(('deletion survived', family))

# Numerical semantic mutants preserve the f64 helper and graph shapes, so
# provenance alone cannot kill them. Both execute the real product functions.
values = [jnp.asarray(v, jnp.float32) for v in
          (np.linspace(25, 40, 16), np.linspace(-18, 5, 16), 58, 0, 21384, 22284)]
restore()
expected = tuple(np.asarray(x) for x in jax.tree.leaves(product(*values)))
for label, file, old, new in [
    ('solar_denominator60_to61', CFILE, '_real4_div_rn(F(lead_seconds), F(60))', '_real4_div_rn(F(lead_seconds), F(61))'),
    ('ozone_weight_sign', OFILE, '_real4_div_rn(cday_plus - day, delta)', '_real4_div_rn(day - cday_plus, delta)'),
]:
    restore()
    install(file, sources[file].replace(old, new))
    actual = tuple(np.asarray(x) for x in jax.tree.leaves(product(*values)))
    try:
        for x, y in zip(actual, expected):
            np.testing.assert_array_equal(x.view('I') if x.dtype == np.float32 else x,
                                          y.view('I') if y.dtype == np.float32 else y)
    except AssertionError as exc:
        results.append(dict(family=label, killed=True, exception='AssertionError', reason=str(exc)[:300]))
        print('SEMANTIC KILLED', label, flush=True)
    else:
        raise AssertionError(('semantic mutant survived', label))
restore()
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(dict(platform='cpu', GPU_executed=False,
    baseline={k:v for k,v in baseline.items() if k != 'records'}, results=results,
    genuine_AssertionError_kills=len(results), setup_errors=0), indent=2)+'\n')
