"""Track/hash/finite checks runnable directly in a clean git archive, without JAX."""
from pathlib import Path
import hashlib,json,sys
import numpy as np


def audit_assets():
    root=Path(__file__).resolve().parents[3]
    points=root/'proofs/s2_sfdda/savepoints'
    manifest=json.loads((points/'MANIFEST.json').read_text())
    assert len(manifest['cases'])==24
    arrays=0
    for name,meta in manifest['cases'].items():
        path=points/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
        with np.load(path) as values:
            assert values['calls'].shape==(9,4)
            for key in values.files:
                a=values[key]
                assert a.dtype.kind in 'ifu' and np.isfinite(a).all(),(name,key)
                arrays+=1
    assert arrays>1000
    hfx_meta=json.loads((points/"HFX_MANIFEST.json").read_text())
    hfx_path=points/"wrf_surface_hfx.npz"
    assert hashlib.sha256(hfx_path.read_bytes()).hexdigest()==hfx_meta["sha256"]
    with np.load(hfx_path) as hfx:
        assert set(hfx.files)==set(manifest["cases"])
        for name in hfx.files:
            assert hfx[name].shape==(9,44,5,6) and np.isfinite(hfx[name]).all()
            assert hfx_meta["cases"][name]["all_prior_outputs_byte_identical"]
    return {'cases':24,'arrays':arrays,'jax_imported':'jax' in sys.modules}


def test_tracked_surface_savepoints():
    assert audit_assets()['cases']==24


if __name__=='__main__':
    assert 'jax' not in sys.modules
    result=audit_assets()
    assert not result['jax_imported']
    print(json.dumps(result))
