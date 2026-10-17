"""Compare the JAX YSU port with the pristine WRF oracle (lane t-ysuqc).

usage: JAX_PLATFORMS=cpu python compare_port.py <repo> <lane> <tag> [--eager] [--build zero|fma]
--eager runs op-by-op (jax.disable_jit): the literal-REAL4 path. Without it the
whole batch is jitted (XLA:CPU contraction/algsimp). --glibc (proof only) routes
the kernel's exp/tanh through glibc expf/tanhf via pure_callback: XLA's own exp
(tanh) differs from glibc in ~9 % (~58 %) of float32 inputs, pow is bitwise glibc. Writes <lane>/compare_<tag>.json
and <lane>/port_<tag>.npy (same layout as the oracle result).
"""
from pathlib import Path
import json, sys
import numpy as np

REPO, LANE, TAG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
EAGER = '--eager' in sys.argv
BUILD = sys.argv[sys.argv.index('--build') + 1] if '--build' in sys.argv else 'zero'
sys.path.insert(0, str(REPO / 'src'))
import jax, jax.numpy as jnp  # noqa: E402
from gpuwrf.physics.pbl_ysu import ysu_columns  # noqa: E402
import gpuwrf.physics.pbl_ysu as pbl_ysu  # noqa: E402

if '--glibc' in sys.argv:
    import ctypes
    _libm = ctypes.CDLL('libm.so.6')
    for _name in ('expf', 'tanhf'):
        getattr(_libm, _name).restype = ctypes.c_float
        getattr(_libm, _name).argtypes = [ctypes.c_float]

    def _host(fn):
        vec = np.vectorize(lambda x: getattr(_libm, fn)(float(x)), otypes=[np.float32])

        def call(x):
            assert x.dtype == jnp.float32
            return jax.pure_callback(lambda a: vec(np.asarray(a)).astype(np.float32),
                                     jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method='broadcast_all')
        return call

    class _GlibcJnp:
        exp = staticmethod(_host('expf'))
        tanh = staticmethod(_host('tanhf'))

        def __getattr__(self, name):
            return getattr(jnp, name)

    pbl_ysu.jnp = _GlibcJnp()

KEYS = ('u', 'v', 'theta', 'qv', 'qc', 'qi', 'exch_h', 'exch_m')
SCALARS = ('pblh', 'kpbl', 'wstar', 'delta', 'u10', 'v10')


def run_port(prof, surf, topdown, flag_qi):
    n = prof.shape[1] - 1
    f = lambda a: jnp.asarray(a, jnp.float32)
    c = lambda j: f(prof[:, :n, j])
    s = lambda j: f(surf[:, j])
    kwargs = dict(psfc=s(0), znt=s(1), ust=s(2), psim=s(3), psih=s(4), xland=s(5), hfx=s(6), qfx=s(7),
                  wspd=s(8), br=s(9), u10=s(10), v10=s(11), uoce=s(12), voce=s(13), ctopo=float(surf[0, 14]),
                  ctopo2=float(surf[0, 15]), dt=float(surf[0, 16]), qc=c(4), qi=c(5) if flag_qi else None,
                  rthraten=c(10), ysu_topdown_pblmix=topdown)
    args = (c(0), c(1), c(2), c(3), c(6), f(prof[:, :, 7]), c(8), c(9))

    def call(*a):
        return ysu_columns(*a, **kwargs)

    with jax.enable_x64(False):
        if EAGER:
            with jax.disable_jit():
                out = call(*args)
        else:
            out = jax.jit(call)(*args)
    res = np.zeros((prof.shape[0], n + 1, 9), np.float32)
    for j, key in enumerate(KEYS):
        if key in out:
            res[:, :n, j] = np.asarray(out[key])
    for j, key in enumerate(SCALARS):
        res[:, j, 8] = np.asarray(out[key])
    return res


def main():
    d = np.load(LANE / 'oracle.npz')
    ref = d['out_' + BUILD]
    port = np.zeros_like(ref)
    dts = d['surface'][:, 16]
    for topdown in (0, 1):
        for fqi in (False, True):
            for dt in np.unique(dts):  # dt is a scalar kernel argument
                idx = np.nonzero((d['topdown'] == topdown) & (d['flag_qi'] == fqi) & (dts == dt))[0]
                if idx.size:
                    port[idx] = run_port(d['profiles'][idx], d['surface'][idx], topdown, fqi)
    np.save(LANE / f'port_{TAG}.npy', port)
    report = {}
    for j, key in enumerate(KEYS):
        rows = np.arange(ref.shape[0]) if key != 'qi' else np.nonzero(d['flag_qi'])[0]
        r, p = ref[rows, :-1, j], port[rows, :-1, j]
        colmax = np.maximum(np.max(np.abs(r), axis=1, keepdims=True), 1e-30)
        diff = np.abs(p - r)
        normal = np.abs(r) >= np.finfo(np.float32).tiny
        report[key] = dict(max_abs=float(diff.max()), max_rel_colmax=float(np.max(diff / colmax)),
                           cells_differ=int(np.count_nonzero(p != r)), cells=int(r.size),
                           cells_differ_normal=int(np.count_nonzero((p != r) & normal)),
                           worst_row=int(rows[np.argmax(np.max(diff / colmax, axis=1))]))
    for j, key in enumerate(SCALARS):
        r, p = ref[:, j, 8], port[:, j, 8]
        report[key] = dict(max_abs=float(np.max(np.abs(p - r))), rows_differ=int(np.count_nonzero(p != r)))
    report['_meta'] = dict(eager=EAGER, glibc_exp_tanh='--glibc' in sys.argv, build=BUILD, rows=int(ref.shape[0]),
                           nonfinite_port=int(np.count_nonzero(~np.isfinite(port))))
    (LANE / f'compare_{TAG}.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
