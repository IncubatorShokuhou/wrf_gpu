"""Reduction experiments only; not replacements for complete release gates."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_sms_projected_u():
    import jax
    from proofs.s1_km5 import full_oracle as ora
    from gpuwrf.dynamics.les3d_smagorinsky import les3d_smagorinsky_memory, _Mem
    with np.load(ROOT / 'data/fixtures/sms3dtke-full-wrf-v1.npz') as z:
        inp, cfg = ora.inputs(z, 'real_mixed')
        got = jax.jit(lambda x: les3d_smagorinsky_memory(x, cfg).ru_tendf)(inp)
        nz, ny, nx1 = inp.u.shape
        m = _Mem(nx1 - 1, ny, nz, np.float32)
        actual = np.asarray(m.interior(got, istag=1, nk=nz))
        ref = np.asarray(m.interior(z['real_mixed/out/ru_tendf'], istag=1, nk=nz))
        mass = inp.c1h[:nz, None, None] * inp.mut[None] + inp.c2h[:nz, None, None]
        eps = 32 * np.finfo(np.float32).eps
        atol = eps * float(abs(mass).max()) * float(abs(inp.u).max()) / float(inp.dt)
        assert np.isfinite(actual).all() and np.isfinite(ref).all()
        np.testing.assert_allclose(actual, ref, rtol=eps, atol=atol)


@pytest.mark.parametrize('joint', [False, True])
def test_kf_root_only(monkeypatch, joint):
    from gpuwrf.integration import nested_pipeline as nested
    load = nested._load_domains
    monkeypatch.setattr(nested, '_load_domains', lambda config, names: load(config, ('d01',)))
    path = ROOT / 'tests/v025/b_column/test_kf_rk.py'
    spec = importlib.util.spec_from_file_location('kf_reduction', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_actual_prod_rk_dispatch_d1_and_composed_keys(monkeypatch, joint)
