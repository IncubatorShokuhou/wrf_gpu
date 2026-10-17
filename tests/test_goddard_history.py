"""Original public Goddard oracle fields through both history and writer seams."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
from gpuwrf.coupling.goddard_radiation import HISTORY_FIELDS
from test_goddard_operational import case
from test_m7_netcdf_writer import writer_authority


def expected(shape):
    p=Path(__file__).resolve().parents[1]/'proofs/s2_goddardrad/savepoints/wrapper_real4.npz'
    with np.load(p) as f:values=f['2_swextra']
    assert values[0]>0 and values[1]>0 and values[-1]>0
    return {key:np.full(shape,value,np.float32) for key,value in zip(HISTORY_FIELDS,values,strict=True)}


@pytest.mark.parametrize('full',[False,True])
def test_native_held_history_and_public_writer(case,tmp_path,full,monkeypatch):
    from gpuwrf.integration.nested_pipeline import _noahmp_surface_diagnostics_for_output
    from gpuwrf.runtime import operational_mode as om
    grid,nl,carry=case;gold=expected(carry.state.t_skin.shape)
    diag=carry.radiation_diagnostics._replace(**{k.lower():v for k,v in gold.items()})
    def forbidden(*a,**kw):raise AssertionError('God-only history must not recompute M9/radiation')
    monkeypatch.setattr(om,'compute_m9_diagnostics',forbidden)
    calls=[]
    output=_noahmp_surface_diagnostics_for_output(carry.state,nl,datetime(2026,6,3,tzinfo=timezone.utc),
        lead_seconds=3600.,noahmp_land=None,noahmp_rad=None,radiation_diagnostics=diag,
        variable_subset=HISTORY_FIELDS,output_radiation_work=calls)
    assert calls==[0]
    for key in HISTORY_FIELDS:assert output[key].tobytes()==gold[key].tobytes()
    path=tmp_path/'wrfout'
    write_wrfout_netcdf(carry.state,grid,nl,path,domain='d01',domain_authority=writer_authority(grid,'d01'),
        lead_hours=1.,valid_time='2026-06-03_01:00:00',run_start='2026-06-03_00:00:00',
        diagnostics=output,variable_subset=HISTORY_FIELDS,full_variable_set=full)
    with Dataset(path) as ds:
        for key in HISTORY_FIELDS:
            assert np.asarray(ds[key][0]).tobytes()==gold[key].tobytes()
            assert ds[key].dimensions==('Time','south_north','west_east')
        assert ds['CTOP2D_OUT'].units=='mb'
        assert ds['COD2D_OUT'].description=='Column Cloud Optical Depth'


@pytest.mark.parametrize('key',HISTORY_FIELDS)
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_writer_rejects_nonfinite_goddard_field_even_when_unrequested(case,tmp_path,key,bad):
    grid,nl,carry=case;gold=expected(carry.state.t_skin.shape);gold[key][1,1]=bad
    with pytest.raises(ValueError,match='nonfinite|non-finite'):
        write_wrfout_netcdf(carry.state,grid,nl,tmp_path/'bad',domain='d01',domain_authority=writer_authority(grid,'d01'),
            lead_hours=1.,valid_time='2026-06-03_01:00:00',run_start='2026-06-03_00:00:00',
            diagnostics=gold,variable_subset=('T2',))
