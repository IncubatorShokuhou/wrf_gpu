"""Shin-Hong (bl_pbl_physics=11) against pristine WRF 4.7.1 module_bl_shinhong.F.

Truth: proofs/t_shinhong (unmodified lane build copy, public call mirroring
module_pbl_driver.F:1301-1329, -O0 no-FMA REAL4, init zero == init inf).
Columns: 4 synthetic regimes + 8 real c07 CPU-WRF columns (stable pblflg=F
kpbl>1, cloud water/ice, convective entrainment), modes 0 / 1 cold / 1 warm.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'proofs/t_shinhong/fixtures/shinhong_wrf_oracle.npz'
META = ROOT / 'proofs/t_shinhong/fixtures/shinhong_wrf_oracle.json'
FIELDS = (('tke', 0), ('el_pbl', 1), ('exch_h', 2), ('u', 3), ('v', 4), ('theta', 5), ('qv', 6), ('qc', 7), ('qi', 8))
SURF = ('psfc', 'znt', 'ust', 'hfx', 'qfx', 'wspd', 'br', 'psim', 'psih', 'dt', 'xland', 'u10', 'v10', 'dx', 'dy',
        'corf', 'ctopo')
# XLA:CPU flushes subnormal operands/results (FINDINGS E117); WRF keeps them.
# Hydrometeor rates below this magnitude are compared with this absolute floor.
Q_SUBNORMAL_FLOOR = 1.0e-30
# (fixture row, field, level) outside the strict compiled-path rule on XLA:CPU:
# one synthetic cell; row 3 is the identical-input mode-0 sibling of rows 4/5.
KNOWN_COMPILED_EXCESS = {(3, 'v', 9), (4, 'v', 9), (5, 'v', 9)}


def _load():
    meta = json.loads(META.read_text())
    assert hashlib.sha256(FIX.read_bytes()).hexdigest() == meta['fixture_sha256']
    assert meta['init_sensitivity_zero_vs_inf_byte'] is True
    assert meta['reference_build']['fma_instructions'] == 0
    d = np.load(FIX)
    assert len(d['mode']) == 36 and set(d['mode'].tolist()) == {0, 1}
    return d


def _run(d, idxs, eager):
    from gpuwrf.physics.bl_shinhong import shinhong_columns
    n = int(d['nz'][idxs[0]])
    a = d['profiles'][idxs][:, :n + 1]
    s = d['surface'][idxs]
    with jax.enable_x64(False):
        real = lambda x: jnp.asarray(x, jnp.float32)
        kw = {name: real(s[:, i]) for i, name in enumerate(SURF) if name not in ('dt', 'dx', 'dy')}
        kw.update(dt=real(s[0, 9]), dx=real(s[0, 13]), dy=real(s[0, 14]), qc=real(a[:, :n, 9]),
                  qi=real(a[:, :n, 10]), shinhong_tke_diag=jnp.asarray(d['mode'][idxs], jnp.int32))
        args = [real(a[:, :, j] if j == 5 else a[:, :n, j]) for j in range(9)]
        if eager:
            with jax.disable_jit():
                out = shinhong_columns(*args, **kw)
        else:
            out = jax.jit(shinhong_columns)(*args, **kw)
        return n, {k: np.asarray(v) for k, v in out.items()}


def _groups(d):
    groups = {}
    for i, n in enumerate(d['nz']):
        groups.setdefault(int(n), []).append(i)
    return list(groups.values())


def test_literal_real4_shinhong_is_byte_identical_to_pristine_wrf():
    """Op-by-op REAL4 evaluation reproduces the original public call bit for bit."""
    d = _load()
    checked = 0
    for idxs in _groups(d):
        n, out = _run(d, idxs, eager=True)
        for pos, i in enumerate(idxs):
            for name, j in FIELDS:
                wrf = d['wrf'][i][:n, j]
                act = out[name][pos].astype(np.float32)
                assert np.isfinite(act).all(), (i, name)
                if name in ('qc', 'qi'):
                    tiny = np.abs(wrf) < Q_SUBNORMAL_FLOOR
                    assert act[~tiny].tobytes() == wrf[~tiny].tobytes(), (i, name)
                    assert np.all(np.abs(act[tiny].astype(np.float64) - wrf[tiny]) <= Q_SUBNORMAL_FLOOR), (i, name)
                else:
                    assert act.tobytes() == wrf.tobytes(), (
                        f'row {i} {name}: max abs {np.max(np.abs(act.astype(np.float64) - wrf)):.3e}')
                checked += 1
            assert int(out['kpbl'][pos]) == int(round(float(d['wrf'][i][1, 9]))), i
            assert out['pblh'][pos].tobytes() == d['wrf'][i][0, 9].tobytes(), i
    assert checked == 36 * len(FIELDS)


DRIVING = ('exch_h', 'u', 'v', 'theta', 'qv', 'qc', 'qi')
TINY = np.finfo(np.float32).tiny


def _bounds_row(d, i):
    """Envelope row for row i. Registry-default mode 0 has no TKE diagnostic, so
    its driving rates use the identical-input mode-1-cold sibling's bounds
    (WRF computes the TKE diagnostic after, and independently of, the rates)."""
    if d['has_envelope'][i]:
        return i
    sib = i + 1
    assert int(d['mode'][i]) == 0 and int(d['mode'][sib]) == 1 and d['has_envelope'][sib], i
    assert d['profiles'][i].tobytes() == d['profiles'][sib].tobytes(), i
    assert d['surface'][i].tobytes() == d['surface'][sib].tobytes(), i
    for name, j in FIELDS:
        if name in DRIVING:
            assert d['wrf'][i][:, j].tobytes() == d['wrf'][sib][:, j].tobytes(), (i, name)
    return sib


def _assert_compiled(d, n, idxs, out):
    """Strict compiled-path rule; raises AssertionError on any violation."""
    for pos, i in enumerate(idxs):
        e = _bounds_row(d, i)
        held = not d['has_envelope'][i]
        for name, j in FIELDS:
            act = np.asarray(out[name][pos], np.float32)
            wrf = d['wrf'][i][:n, j]
            lo, hi = d['lower'][e][:n, j], d['upper'][e][:n, j]
            for arr, what in ((act, 'candidate'), (wrf, 'reference'), (lo, 'lower'), (hi, 'upper')):
                assert np.isfinite(arr).all(), f'nonfinite {what} row {i} {name}'
            if held and name in ('tke', 'el_pbl'):
                # Registry default diag=0: TKE_PBL untouched, EL_PBL stays 0 (:673, :1478).
                expect = d['profiles'][i][:n, 8] if name == 'tke' else wrf
                assert act.tobytes() == expect.tobytes(), f'outside held mode0 row {i} {name}'
                continue
            for _ in range(2):
                lo = np.nextafter(lo, np.float32(-np.inf)); hi = np.nextafter(hi, np.float32(np.inf))
            compare = (np.abs(wrf) >= TINY) | (np.abs(act) >= TINY)
            # Both subnormal/zero: XLA:CPU flushes (E117); WRF keeps the subnormal.
            sub = ~compare
            assert np.all(np.abs(act[sub].astype(np.float64) - wrf[sub]) <= TINY), f'outside subnormal pair row {i} {name}'
            bad = compare & ((act < lo) | (act > hi))
            for k in np.nonzero(bad)[0]:
                # Disclosed XLA:CPU excess (proofs/t_shinhong/README.md): one synthetic
                # cell (cold, warm, and since review its identical-input mode-0 sibling),
                # bounded by one ULP of the solved wind over 2*dt, the quantum of
                # WRF's (f-ux)*rdt tendency; op-by-op is BYTE.
                assert (int(i), name, int(k)) in KNOWN_COMPILED_EXCESS, f'outside envelope row {i} {name} k={int(k)}'
                solved = d['profiles'][i][k, j - 3] + np.float32(36.0) * wrf[k]
                quantum = np.spacing(np.abs(solved)) / np.float32(36.0)
                excess = max(float(lo[k] - act[k]), float(act[k] - hi[k]))
                assert excess <= quantum, f'outside quantum row {i} {name} k={int(k)} {excess} {quantum}'


@pytest.fixture(scope='module')
def compiled():
    d = _load()
    return d, [(idxs, *_run(d, idxs, eager=False)) for idxs in _groups(d)]


def test_compiled_shinhong_inside_wrf_input_ulp_envelope(compiled):
    """XLA-compiled kernel inside the original +-1 ULP input envelope +- 2 output ULPs."""
    d, runs = compiled
    for idxs, n, out in runs:
        _assert_compiled(d, n, idxs, out)


@pytest.mark.parametrize('case', ['erase_driving', 'zero_reference_normal_candidate', 'nan_candidate'])
def test_compiled_gate_rejects_sentinels(compiled, case):
    """Gate sensitivity (review rv-t-shinhong): zeroed compiled driving outputs vs
    normal WRF truth, a normal candidate where WRF is exactly zero, and NaN."""
    d, runs = compiled
    edited = [(idxs, n, {k: np.array(v, copy=True) for k, v in out.items()}) for idxs, n, out in runs]
    hit = False
    for idxs, n, out in edited:
        for pos, i in enumerate(idxs):
            for name, j in FIELDS:
                if name not in DRIVING:
                    continue
                wrf = d['wrf'][i][:n, j]
                if case == 'erase_driving':
                    hit |= bool(np.any(np.abs(wrf) >= TINY))
                    out[name][pos] = 0.0
                elif case == 'zero_reference_normal_candidate':
                    zero = np.nonzero(wrf == 0.0)[0]
                    if zero.size:
                        out[name][pos][zero[0]] = 1.0; hit = True
                elif case == 'nan_candidate' and pos == 0 and name == 'u':
                    out[name][pos][0] = np.nan; hit = True
    assert hit
    match = 'nonfinite' if case == 'nan_candidate' else 'outside'
    with pytest.raises(AssertionError, match=match):
        for idxs, n, out in edited:
            _assert_compiled(d, n, idxs, out)


def test_adapter_applies_wrf_rqcblten_rqiblten(monkeypatch):
    """module_bl_shinhong.F:253-256: the State path adds dt*RQC/RQIBLTEN."""
    import importlib.util
    from gpuwrf.coupling import scan_adapters as scan
    spec = importlib.util.spec_from_file_location('shin_fix', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(nz=8, ny=2, nx=2)
    state = f._state(grid)
    state = state.replace(qc=jnp.full_like(state.qc, 2.0e-4), qi=jnp.full_like(state.qi, 1.0e-5))
    rate_c, rate_i = np.float32(-3.0e-7), np.float32(5.0e-8)

    def producer(*args, **kwargs):
        zero = jnp.zeros((4, 8), jnp.float32)
        return dict(u=zero, v=zero, theta=zero, qv=zero, tke=args[8], exch_h=zero,
                    qc=jnp.full_like(zero, rate_c), qi=jnp.full_like(zero, rate_i))

    monkeypatch.setattr(scan, 'shinhong_columns', producer)
    out = scan.shinhong_pbl_adapter(state, 18.0, grid, shinhong_tke_diag=0)
    np.testing.assert_array_equal(np.asarray(out.qc), np.asarray(state.qc + 18.0 * rate_c))
    np.testing.assert_array_equal(np.asarray(out.qi), np.asarray(state.qi + 18.0 * rate_i))
