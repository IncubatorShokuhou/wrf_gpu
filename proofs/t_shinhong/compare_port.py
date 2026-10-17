"""Compare the JAX Shin-Hong column kernel with the pristine-WRF oracle (lane t-shinhong).

usage: JAX_PLATFORMS=cpu python compare_port.py <repo> <lane_dir> <tag> [--wrf-args]
Default arm reproduces the CURRENT adapter call (no qc/qi, corf default); with
--wrf-args the kernel receives WRF's QC3D/QI3D and CORF=f (module_pbl_driver.F:1301).
REAL4 ABI as in the operational caller (jax.enable_x64(False), f32 operands).
"""
from pathlib import Path
import inspect, json, sys, time
import numpy as np

repo, lane, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
wrf_args = '--wrf-args' in sys.argv
sys.path.insert(0, str(repo / 'src'))
import jax, jax.numpy as jnp
from gpuwrf.physics import bl_shinhong
assert jax.devices()[0].platform == 'cpu'
assert Path(bl_shinhong.__file__).resolve().is_relative_to(repo.resolve()), bl_shinhong.__file__

d = np.load(lane / 'oracle.npz')
manifest = json.loads((lane / 'oracle_manifest.json').read_text())
rows = manifest['rows']
FIELDS = [('tke', 0), ('el_pbl', 1), ('exch_h', 2), ('u', 3), ('v', 4), ('theta', 5), ('qv', 6), ('qc', 7), ('qi', 8)]
SURF = ('psfc', 'znt', 'ust', 'hfx', 'qfx', 'wspd', 'br', 'psim', 'psih', 'dt', 'xland', 'u10', 'v10', 'dx', 'dy',
        'corf', 'ctopo')
params = inspect.signature(bl_shinhong.shinhong_columns).parameters
report = dict(tag=tag, wrf_args=wrf_args, eager='--eager' in sys.argv, xla_flags=__import__('os').environ.get('XLA_FLAGS', ''), kernel=str(bl_shinhong.__file__), rows=[])
t0 = time.monotonic()
groups = {}
port_rows = {}
for idx, r in enumerate(rows):
    groups.setdefault(r['nz'], []).append(idx)
for n, idxs in groups.items():
    a = d['profiles'][idxs][:, :n + 1]
    s = d['surface'][idxs]
    with jax.enable_x64(False):
        real = lambda x: jnp.asarray(x, jnp.float32)
        kw = {name: real(s[:, i]) for i, name in enumerate(SURF) if name not in ('dt', 'dx', 'dy', 'corf', 'ctopo')}
        kw.update(dt=real(float(s[0, 9])), dx=real(float(s[0, 13])), dy=real(float(s[0, 14])))
        if wrf_args:
            kw.update(qc=real(a[:, :n, 9]), qi=real(a[:, :n, 10]), corf=real(s[:, 15]))
        kw['shinhong_tke_diag'] = jnp.asarray(d['mode'][idxs], jnp.int32)
        args = [real(a[:, :, j] if j == 5 else a[:, :n, j]) for j in range(9)]
        if '--eager' in sys.argv:
            with jax.disable_jit():
                out = bl_shinhong.shinhong_columns(*args, **kw)
        else:
            out = jax.jit(bl_shinhong.shinhong_columns)(*args, **kw)
        out = {k: np.asarray(v) for k, v in out.items()}
    for pos, idx in enumerate(idxs):
        port_rows[idx] = {k: v[pos] for k, v in out.items()}
        ref = d['out_zero'][idx]; fma = d['out_fma'][idx]
        row = dict(rows[idx])
        for name, j in FIELDS:
            exp = ref[:n, j]; spread = np.abs(fma[:n, j] - exp)
            if name not in out:
                act = np.zeros(n, np.float32); row[name + '_missing'] = True
            else:
                act = out[name][pos]
            err = np.abs(act.astype(np.float64) - exp)
            scale = max(float(np.max(np.abs(exp))), 1e-30)
            row[name] = dict(max_abs=float(err.max()), max_rel_to_colmax=float(err.max() / scale),
                             byte=bool(act.astype(np.float32).tobytes() == exp.astype(np.float32).tobytes()),
                             fma_spread_abs=float(spread.max()), nonfinite=int(np.count_nonzero(~np.isfinite(act))),
                             colmax=scale)
        sc = ref[:, 9]
        row['hpbl'] = [float(out['pblh'][pos]), float(sc[0])]
        row['kpbl'] = [int(out['kpbl'][pos]), int(round(float(sc[1])))]
        row['wstar'] = [float(out['wstar'][pos]), float(sc[2])]
        report['rows'].append(row)
report['elapsed_s'] = time.monotonic() - t0
summary = {}
for name, _ in FIELDS:
    for mode in ('mode0', 'mode1_cold', 'mode1_warm'):
        sel = [r for r in report['rows'] if r['label'] == mode]
        summary[f'{name}:{mode}'] = dict(
            worst_rel=max(r[name]['max_rel_to_colmax'] for r in sel),
            byte_rows=sum(r[name]['byte'] for r in sel), rows=len(sel),
            beyond_fma_spread=sum(r[name]['max_abs'] > max(4 * r[name]['fma_spread_abs'], 1e-30) for r in sel))
summary['kpbl_mismatch'] = sum(r['kpbl'][0] != r['kpbl'][1] for r in report['rows'])
report['summary'] = summary
np.save(lane / f'port_{tag}.npy', np.array([port_rows[i] for i in range(len(rows))], dtype=object), allow_pickle=True)
(lane / f'compare_{tag}.json').write_text(json.dumps(report, indent=1) + '\n')
for k, v in summary.items():
    print(k, v)
