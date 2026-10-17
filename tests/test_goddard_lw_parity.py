"""Pristine option-5 lwrad solver gate at its original REAL4 output boundary."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from gpuwrf.physics.ra_goddard_lw import solve_lw_column

SAVE=Path(__file__).resolve().parents[1]/'proofs/s2_goddardrad/savepoints'


@pytest.mark.parametrize('case',range(1,7))
def test_native_lwrad_pristine(case):
    manifest=json.loads((SAVE/'lw_manifest.json').read_text())
    path=SAVE/f'lw_case_{case}.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['savepoint_sha256'][path.name]
    data=json.loads(path.read_text()); c=data['columns']; s=data['scalars']
    tau=np.broadcast_to(np.asarray(c['taucl_b1'])[:,None],(s['np'],10))
    zeros=np.zeros_like(tau)
    args=[np.asarray(c[k]) for k in ('pl','ta','wa','oa')]+[s['tb'],s['ts'],np.asarray(c['emiss']),np.asarray(c['fcld']),tau,zeros,zeros,zeros]
    assert all(np.isfinite(x).all() for x in args)
    expected=[np.asarray(c[k],np.float32) for k in ('flx','acflxd','acflxu')]
    actual=jax.jit(solve_lw_column)(*args)
    for a,e in zip(actual,expected):
        assert a.dtype==np.float32
        assert np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a,e,**manifest['tolerance'])
