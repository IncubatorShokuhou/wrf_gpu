"""First-divergence observer for the Shin-Hong momentum solve (lane t-shinhong).

Injects WRITE-only statements into a lane copy of the pristine module (body
otherwise byte-identical; restoring the edits must give the original bytes),
dumps the momentum tridiagonal (al, ad, au, f1, f2) and the PBL scalars, and
compares with the JAX kernel's own _thomas_scan operands (eager spy).
usage: JAX_PLATFORMS=cpu python observe_momentum.py <repo> <lane_dir> <row> [<row> ...]
"""
from pathlib import Path
import ctypes, json, os, subprocess, sys
import numpy as np

repo, lane = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
rows_wanted = [int(x) for x in sys.argv[3:]]
here = Path(__file__).resolve().parent
src = (lane / 'oracle/module_bl_shinhong.F').read_text()
marker = '   call tridi1n(al,ad,cu,r1,r2,au,f1,f2,its,ite,kts,kte,1)\n'
dump = ('   write(99) al(its,kts:kte),ad(its,kts:kte),au(its,kts:kte),f1(its,kts:kte),f2(its,kts:kte), &\n'
        '     xkzm(its,kts:kte),hgamu(its),hgamv(its),ufxpbl(its),vfxpbl(its),cslen(its),hpbl(its), &\n'
        '     real(kpbl(its)),real(merge(1,0,pblflg(its))),we(its),wspd1(its),rhox(its),zfacent(its,kts:kte)\n')
assert src.count(marker) == 1
observed = src.replace(marker, dump + marker)
assert observed.replace(dump, '', 1) == src
build = lane / 'observe'
build.mkdir(exist_ok=True)
(build / 'observed.F').write_text(observed)
wrapper = (here / 'shinhong_oracle_wrapper.f90').read_text()
wrapper = wrapper.replace('call shinhong(', "open(99,file='observed.bin',access='stream',form='unformatted',status='replace')\ncall shinhong(", 1)
wrapper = wrapper.replace('result=0.\n', 'close(99)\nresult=0.\n', 1)
(build / 'wrapper.f90').write_text(wrapper)
subprocess.run(['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran', '-shared', '-fPIC', '-O0', '-ffp-contract=off',
                '-finit-real=zero', '-ffree-form', '-ffree-line-length-none', 'observed.F', 'wrapper.f90', '-o',
                'observed.so'], cwd=build, check=True, timeout=120, capture_output=True)
os.chdir(build)
lib = ctypes.CDLL(str(build / 'observed.so'))
arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
lib.probe.argtypes = [ctypes.c_int, ctypes.c_int, arr, arr, arr]

sys.path.insert(0, str(repo / 'src'))
import jax, jax.numpy as jnp
from gpuwrf.physics import bl_shinhong as m
d = np.load(lane / 'oracle.npz')
names = ('psfc', 'znt', 'ust', 'hfx', 'qfx', 'wspd', 'br', 'psim', 'psih', 'dt', 'xland', 'u10', 'v10', 'dx', 'dy',
         'corf', 'ctopo')
captured = []
orig = m._thomas_scan
m._thomas_scan = lambda *a: (captured.append([np.asarray(x) for x in a]), orig(*a))[1]
report = {}
for row in rows_wanted:
    n = int(d['nz'][row]); p = np.array(d['profiles'][row][:n + 1], order='F'); s = np.array(d['surface'][row], order='F')
    out = np.zeros((n + 1, 10), np.float32, order='F')
    lib.probe(n, int(d['mode'][row]), p, s.copy(), out)
    assert out.tobytes() == d['out_zero'][row][:n + 1].tobytes(), 'observer must not change public bytes'
    raw = np.fromfile(build / 'observed.bin', np.float32)
    k = 0
    def take(c):
        global k
        v = raw[k:k + c]; k += c; return v
    w = {nm: take(n) for nm in ('al', 'ad', 'au', 'f1', 'f2', 'xkzm')}
    for nm in ('hgamu', 'hgamv', 'ufxpbl', 'vfxpbl', 'cslen', 'hpbl', 'kpbl', 'pblflg', 'we', 'wspd1', 'rhox'):
        w[nm] = float(take(1)[0])
    w['zfacent'] = take(n)
    assert k == raw.size
    captured.clear()
    with jax.enable_x64(False), jax.disable_jit():
        f = lambda x: jnp.asarray(x, jnp.float32)
        kw = {nm: f(s[i]) for i, nm in enumerate(names)}
        m._shinhong_column(*[f(p[:, j] if j == 5 else p[:n, j]) for j in range(9)], f(p[:n, 9]), f(p[:n, 10]),
                           **kw, shinhong_tke_diag=int(d['mode'][row]))
    lower, diag, upper, rhs_uv = captured[2]; rhs_u, rhs_v = rhs_uv[0], rhs_uv[1]
    rel = lambda a, b: float(np.max(np.abs(np.asarray(a, np.float64) - b)) / max(np.max(np.abs(b)), 1e-30))
    report[row] = dict(
        wrf_scalars={kk: w[kk] for kk in ('hgamu', 'hgamv', 'ufxpbl', 'vfxpbl', 'cslen', 'hpbl', 'kpbl', 'pblflg', 'we', 'wspd1', 'rhox')},
        upper=rel(upper[:n - 1], w['au'][:n - 1]), lower=rel(lower[1:], w['al'][:n - 1]), diag=rel(diag, w['ad']),
        rhs_u=rel(rhs_u, w['f1']), rhs_v=rel(rhs_v, w['f2']),
        rhs_u_minus_u_rel=rel(rhs_u - p[:n, 0], w['f1'] - p[:n, 0]),
        per_level_rhs_u_minus_u=[[float(a), float(b)] for a, b in zip(rhs_u - p[:n, 0], w['f1'] - p[:n, 0])][:12],
        per_level_diag=[[float(a), float(b)] for a, b in zip(diag, w['ad'])][:6])
print(json.dumps(report, indent=1))
