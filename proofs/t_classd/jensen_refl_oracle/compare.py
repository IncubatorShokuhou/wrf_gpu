import json, sys, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / 'src'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
F = ROOT / 'proofs/t_classd/jensen_refl_oracle/fixtures'
m = json.loads((F / 'manifest.json').read_text()); d = np.load(F / 'oracle.npz')
nz = m['nz']; names = m['input_names']; onames = m['output_names']
fields = {n: d['inputs'][:, i*nz:(i+1)*nz] for i, n in enumerate(names)}
ref = {n: d['outputs'][:, i*nz:(i+1)*nz] for i, n in enumerate(onames)}
from gpuwrf.physics.jensen_ishmael.column import step, mp_jensen_ishmael
res = {}
got, _ = jax.jit(lambda f: step(f, jnp.float32(m['dt']), m['it']))({k: jnp.asarray(v) for k, v in fields.items()})
xla = np.asarray(got['dbz'])
spec = importlib.util.spec_from_file_location('libm', ROOT / 'proofs/s2_mpmore/jensen55/libm_numpy.py'); libm = importlib.util.module_from_spec(spec); spec.loader.exec_module(libm)
spec = importlib.util.spec_from_file_location('reft', ROOT / 'proofs/s2_mpmore/jensen55/reference_tables.py'); reft = importlib.util.module_from_spec(spec); spec.loader.exec_module(reft)
backend = libm.aggregation_backend(out.parent / 'libmvec')
with np.errstate(all='ignore'):
    lit, _ = mp_jensen_ishmael(fields, reft.load(ROOT, 'fp32'), np.float32(m['dt']), m['it'], xp=backend)
lit = np.asarray(lit['dbz'])
cpu = d['cpu_refl']; R = ref['dbz']
for name, arr in (('xla', xla), ('literal_libm', lit)):
    diff = np.abs(arr.astype(np.float64) - R)
    res[name] = dict(byte_equal=int((arr.view(np.uint32) == R.view(np.uint32)).sum()), n=int(R.size),
                     max_abs_dbz=float(diff.max()), live_mismatch=int(((arr > -35) != (R > -35)).sum()))
live = (R > -35) | (cpu > -35)
dc = np.abs(R.astype(np.float64) - cpu)[live]
res['pristine_vs_cpu_history_attribution'] = dict(values=int(live.sum()), median=float(np.median(dc)), p90=float(np.quantile(dc, .9)),
    max=float(dc.max()), live_agree=float(((R > -35) == (cpu > -35)).mean()), corr=float(np.corrcoef(R[live], cpu[live])[0, 1]))
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res))
