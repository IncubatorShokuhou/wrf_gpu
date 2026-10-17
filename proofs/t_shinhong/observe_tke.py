"""First-divergence observer for the Shin-Hong TKE diagnostic (lane t-shinhong).

WRITE-only edits to a lane copy of the pristine module (restoring the edits must
give the original bytes; the observed build must reproduce the public outputs
byte for byte). Dumps mixlen/prodq2/vdifq operands and results and compares
them with the JAX kernel's own stage operands (eager spies).
usage: JAX_PLATFORMS=cpu python observe_tke.py <repo> <lane_dir> <row> [<row> ...]
"""
from pathlib import Path
import ctypes, json, os, subprocess, sys
import numpy as np

repo, lane = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
rows_wanted = [int(x) for x in sys.argv[3:]]
here = Path(__file__).resolve().parent
src = (lane / 'oracle/module_bl_shinhong.F').read_text()
edits = [
    ('     call mixlen(lmh,',
     '     write(99) uxk,vxk,txk,thxk,thvxk,qxk(kts:kte),qxk(kte+1:2*kte),q2xk,zqk, &\n'
     '       akmk(kts+1:kte),akhk(kts+1:kte),mfk(kts+1:kte),ufxpblk(kts+1:kte),vfxpblk(kts+1:kte),qfxpblk(kts+1:kte), &\n'
     '       hgame,ptke1,ust(i),corf(i),epshol(i),hpbl(i),real(kpbl(i)),real(merge(1,0,pblflg(i))), &\n'
     '       efxpbl(i),delxy,hgamu(i),hgamv(i),hgamq(i)\n'),
    ('     call prodq2(lmh,', '     write(99) s2(kts+1:kte),gh(kts+1:kte),rig(kts+1:kte),el(kts+1:kte)\n'),
    ('     call vdifq(lmh,', '     write(99) q2xk\n'),
    ('!---- save the new tke and mixing length.', '     write(99) q2xk\n'),
]
observed = src
for marker, add in edits:
    assert observed.count(marker) == 1, marker
    observed = observed.replace(marker, add + marker)
restored = observed
for _, add in edits:
    restored = restored.replace(add, '', 1)
assert restored == src
build = lane / 'observe_tke'
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
cap = {}
for nm in ('_mixlen', '_prodq2', '_vdifq'):
    orig = getattr(m, nm)
    setattr(m, nm, (lambda o, k: (lambda *a: (cap.__setitem__(k, ([np.asarray(x) for x in a], o(*a))), cap[k][1])[1]))(orig, nm))


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


report = {}
for row in rows_wanted:
    n = int(d['nz'][row]); p = np.array(d['profiles'][row][:n + 1], order='F'); s = np.array(d['surface'][row], order='F')
    mode = int(d['mode'][row])
    assert mode == 1
    out = np.zeros((n + 1, 10), np.float32, order='F')
    lib.probe(n, mode, p, s.copy(), out)
    assert out.tobytes() == d['out_zero'][row][:n + 1].tobytes(), 'observer must not change public bytes'
    raw = np.fromfile(build / 'observed.bin', np.float32); k = 0
    def take(c):
        global k
        v = raw[k:k + c]; k += c; return v
    w = {nm: take(n) for nm in ('u', 'v', 't', 'th', 'thv', 'qv', 'qc', 'q2')}
    w['z'] = take(n + 1)
    for nm in ('akm', 'akh', 'mf', 'ufx', 'vfx', 'qfx'):
        w[nm] = take(n - 1)
    w['hgame'] = take(n); w['ptke'] = take(n)
    for nm in ('ust', 'corf', 'epshol', 'hpbl', 'kpbl', 'pblflg', 'efxpbl', 'delxy', 'hgamu', 'hgamv', 'hgamq'):
        w[nm] = float(take(1)[0])
    for nm in ('s2', 'gh', 'ri', 'el'):
        w[nm] = take(n - 1)
    w['q2_prod'] = take(n); w['q2_vdif'] = take(n)
    assert k == raw.size, (k, raw.size)
    cap.clear()
    with jax.enable_x64(False), jax.disable_jit():
        f = lambda x: jnp.asarray(x, jnp.float32)
        kw = {nm: f(s[i]) for i, nm in enumerate(names)}
        m._shinhong_column(*[f(p[:, j] if j == 5 else p[:n, j]) for j in range(9)], f(p[:n, 9]), f(p[:n, 10]),
                           **kw, shinhong_tke_diag=1)
    (u, v, t, the, q, cwm, q2, z, ustar, corf, epshol, hpbl, lpbl, pblflg, mf, ufx_fold, vfx_fold, qfx_fold), (s2, gh, ri, el) = cap['_mixlen']
    pargs, q2p = cap['_prodq2']
    vargs, q2v = cap['_vdifq']
    akm, akh = pargs[7], pargs[8]
    ptke1, hgame, efx = vargs[5], vargs[6], vargs[10]
    report[row] = dict(
        inputs={nm: rel(a, w[nm]) for nm, a in (('u', u), ('v', v), ('t', t), ('th', the), ('qv', q), ('qc', cwm),
                                                ('q2', q2), ('z', z), ('akm', akm[1:]), ('akh', akh[1:]), ('mf', mf[1:]),
                                                ('hgame', hgame), ('ptke', ptke1))},
        scalars={nm: [float(np.asarray(a)), w[nm]] for nm, a in (('epshol', epshol), ('hpbl', hpbl), ('kpbl', lpbl),
                                                                 ('efxpbl', efx), ('corf', corf), ('ust', ustar))},
        ufx_fold=rel(ufx_fold[1:], w['hgamu'] / w['hpbl'] + w['ufx']) if w['pblflg'] else None,
        mixlen={nm: rel(a[1:], w[nm]) for nm, a in (('s2', s2), ('gh', gh), ('ri', ri), ('el', el))},
        prodq2_out=rel(q2p, w['q2_prod']), vdifq_out=rel(q2v, w['q2_vdif']))
    report[row]['per_level_q2_prod'] = [[float(a), float(b)] for a, b in zip(q2p, w['q2_prod'])][:8]
    report[row]['per_level_q2_vdif'] = [[float(a), float(b)] for a, b in zip(q2v, w['q2_vdif'])][:8]
print(json.dumps(report, indent=1))
