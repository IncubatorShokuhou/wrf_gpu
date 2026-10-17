"""Pristine full-wrapper oracle: distinguish per-level GOTO200 from column GOTO400."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.microphysics_morrison_aero import morrison_aero_run

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ['TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN',
          'NI_IN','NS_IN','NR_IN','NG_IN','NC_IN','PII','P','DZ','W','KZH_IN',
          'AEROCU_DUST1','AEROCU_DUST2','AEROCU_DUST3','AEROCU_DUST4',
          'AEROCU_SEASALT','AEROCU_SULFATE','AEROCU_BCPHOB','AEROCU_BCPHIL',
          'AEROCU_OCPHOB','AEROCU_OCPHIL']


@pytest.mark.parametrize('mode', ['fp32','fp64'])
@pytest.mark.parametrize('case', range(8,13))
def test_nc_dry_and_active_columns(mode, case):
    jax.config.update('jax_enable_x64', True)
    data = json.loads((ROOT/f'proofs/s1_mpfix/fixtures/dry_{mode}_{case}.json').read_text())
    dtype = jnp.float32 if mode == 'fp32' else jnp.float64
    args = [jnp.asarray([data['columns'][name]], dtype) for name in INPUTS]
    out = morrison_aero_run(*args,jnp.asarray(data['scalars']['DT'],dtype),aercu_opt=0)
    reference = np.array(data['columns']['NC_OUT'])
    actual = np.asarray(out['nc'])[0]
    assert np.isfinite(actual).all()
    if case in (8,9):
        # Wrapper zeroes incoming NC, then the entire column skips finalization.
        np.testing.assert_array_equal(reference, 0.)
        np.testing.assert_array_equal(actual, reference)
    else:
        # Final reset applies to ALL levels once ANY level passes GOTO200.
        assert np.count_nonzero(reference) == 40
        np.testing.assert_allclose(actual, reference, rtol=2e-6 if mode=='fp32' else 1e-12,atol=0.)


def test_batched_columns_do_not_share_ltrue():
    jax.config.update('jax_enable_x64', True)
    cases = [json.loads((ROOT/f'proofs/s1_mpfix/fixtures/dry_fp32_{case}.json').read_text())
             for case in range(8,13)]
    args = [jnp.asarray([d['columns'][name] for d in cases],jnp.float32) for name in INPUTS]
    out = morrison_aero_run(*args,jnp.float32(60.),aercu_opt=0)
    reference = np.array([d['columns']['NC_OUT'] for d in cases])
    np.testing.assert_allclose(np.asarray(out['nc']),reference,rtol=2e-6,atol=0.)
