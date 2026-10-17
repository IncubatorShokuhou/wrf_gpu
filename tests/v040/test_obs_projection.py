"""Host projection and wind rotation against original full WRF module_llxy."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from gpuwrf.io.obs_projection import ObservationProjection

ROOT=Path(__file__).resolve().parents[2]/'proofs/s1_obsnudge/full/savepoints'


@pytest.mark.parametrize('precision',[4,8])
def test_all_admitted_projection_families(precision):
    meta=json.loads((ROOT/'PROJECTION_MANIFEST.json').read_text())
    path=ROOT/'projection.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['fixture_sha256']
    a=np.load(path);dtype=np.float32 if precision==4 else np.float64
    for values,truth in zip(a['inputs'],a[f'output_r{precision}'],strict=True):
        code,*parameters=values[:8]
        proj=ObservationProjection(int(code),*parameters,dtype=dtype)
        lat,lon,u,v=values[8:]
        actual=np.asarray((*proj.station_coordinates(lat,lon),*proj.rotate_wind(lon,u,v)),dtype)
        assert np.isfinite(actual).all()
        bound=4*np.finfo(dtype).eps*np.maximum(np.abs(truth),1)
        assert np.all(np.abs(actual-truth)<=bound),(values,actual,truth,actual-truth)
