"""Tracked SMS savepoints and literal drivers must survive a clean archive."""
from pathlib import Path
import hashlib
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]


def test_sms_full_and_upstream_assets():
    for manifest in ('fixture.yaml','upstream_fixture.yaml','full_fixture.yaml','stage_fixture.yaml'):
        meta=yaml.safe_load((ROOT/'proofs/s1_km5'/manifest).read_text())
        for record in meta['files']:
            p=ROOT/record['path']
            assert p.is_file(),p
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],p
    with np.load(ROOT/'data/fixtures/sms3dtke-full-wrf-v1.npz') as z:
        assert np.any(z['real_mixed/in/sms_ustm']!=z['real_mixed/in/ust'])
        for key in ('real_mixed','periodic_convective','zero_tke_stable'):
            for n in ('qg','Ni','Nr'):
                assert np.count_nonzero(z[key+'/out/'+n])>0,(key,n)
    with np.load(ROOT/'data/fixtures/sms3dtke-stages-wrf-v1.npz') as z:
        assert z['0/tke'].shape[0]==3
        assert not np.array_equal(z['0/tke'],z['2/tke'])
