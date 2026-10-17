"""LES TKE transport caller against unmodified WRF, not a JAX self-compare."""
from pathlib import Path
import importlib.util

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tke_stage_oracle', ROOT/'proofs/o2_km2/stage_oracle.py')
stage = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = stage
spec.loader.exec_module(stage)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('diff6', [0, 2])
def test_tke_three_rk_stages_match_pristine_wrf(periodic, diff6):
    with np.load(stage.FIXTURE) as z, np.load(ROOT/'data/fixtures/les-tke-stage-wrf-v2.npz') as ref:
        key = f'{int(periodic)}_{diff6}'
        want = ref[key+'/tke']
        assert np.count_nonzero(ref[key+'/advection']) > 0
        assert np.count_nonzero(ref[key+'/source'][:2]) > 0
        for jit in (False, True):
            got = np.asarray(stage.advance(z, periodic, diff6, jit=jit))
            assert np.isfinite(got).all() and np.isfinite(want).all()
            np.testing.assert_allclose(got, want, rtol=32*np.finfo(np.float32).eps,
                                       atol=32*np.finfo(np.float32).eps*np.abs(want).max())
        jax.clear_caches()


def test_tke_history_is_separate_from_pbl_qke():
    from types import SimpleNamespace
    from gpuwrf.io.wrfout_writer import _build_subset_output_fields, _dimension_sizes
    nz,ny,nx = 3,4,5
    tke = np.arange(nz*ny*nx,dtype=np.float32).reshape(nz,ny,nx)/10
    state = SimpleNamespace(tke=tke, qke=tke*7, tke_pbl=tke*11)
    fields = _build_subset_output_fields(state, grid=SimpleNamespace(nx=nx,ny=ny,nz=nz),
        namelist=None, diagnostics=None, land_state=None, run_start=None,
        dimensions=_dimension_sizes(nx=nx,ny=ny,nz=nz,namelist=None),
        lead_hours=0, requested_names=frozenset({'TKE','QKE'}))
    np.testing.assert_array_equal(fields['TKE'],tke)
    np.testing.assert_array_equal(fields['QKE'],tke*7)


def test_tke_fixtures_load_from_clean_git_archive():
    import io
    import subprocess
    import tarfile

    if not (ROOT/'.git').exists():
        pytest.skip('git archive proof requires the source checkout')
    names = ['data/fixtures/les-tke-stage-wrf-v2.npz',
             'data/fixtures/les3d-smag3d-oracle-v1.npz']
    raw = subprocess.check_output(['git','archive','HEAD',*names],cwd=ROOT,timeout=30)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for name in names:
            with np.load(io.BytesIO(archive.extractfile(name).read())) as z:
                assert 'meta_json' in z and len(z.files)>4
