from pathlib import Path
import hashlib,json
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.coupling.boundary_feedback import build_feedback_weights,point_feedback_weights,apply_feedback

FIXTURE=Path(__file__).parent/"fixtures/u2_point_feedback_oracle.npz"
META=json.loads(FIXTURE.with_suffix(".json").read_text())

@pytest.mark.parametrize("index,case",list(enumerate(META["cases"])))
def test_point_feedback_verbatim_wrf(index,case):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==META["fixture_sha256"]
    with np.load(FIXTURE) as f:
        p,c,expected=(f[key+str(index)] for key in ("p","c","o"))
    mass=build_feedback_weights(parent_grid_ratio=case["ratio"],i_parent_start=3,j_parent_start=3,
        parent_we=22,parent_sn=20,child_we=c.shape[-1]-(case["stagger"]=="U"),
        child_sn=c.shape[-2]-(case["stagger"]=="V"),spec_zone=case["zone"],stagger=case["stagger"])
    actual=np.asarray(apply_feedback(jnp.asarray(p),jnp.asarray(c),point_feedback_weights(mass)))
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_array_equal(actual,expected)


def test_initial_staggered_coordinate_feedback_uses_pristine_points(tmp_path):
    from types import SimpleNamespace
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.nesting.initial_feedback import _input_point_overrides
    from gpuwrf.coupling.boundary_feedback import StateFeedbackWeights
    expected={}
    with np.load(FIXTURE) as fixture:
        for domain,prefix,ny,nx in (('d01','p',20,22),('d02','c',30,36)):
            with Dataset(tmp_path/f'wrfinput_{domain}','w') as ds:
                for name,size in (('Time',1),('south_north',ny),('west_east',nx),
                                  ('south_north_stag',ny+1),('west_east_stag',nx+1)):
                    ds.createDimension(name,size)
                for index,name,dims in ((1,'XLAT_U',('Time','south_north','west_east_stag')),
                                         (2,'XLAT_V',('Time','south_north_stag','west_east'))):
                    ds.createVariable(name,'f4',dims)[0]=fixture[prefix+str(index)]
                    expected[name]=fixture['o'+str(index)]
    common=dict(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,
        parent_we=22,parent_sn=20,child_we=36,child_sn=30)
    mass=build_feedback_weights(**common)
    weights=StateFeedbackWeights(mass,build_feedback_weights(stagger='U',**common),
        build_feedback_weights(stagger='V',**common),point=point_feedback_weights(mass))
    got=_input_point_overrides(SimpleNamespace(path=tmp_path),'d01','d02',weights)
    for name,value in expected.items():np.testing.assert_array_equal(got[name],value)
