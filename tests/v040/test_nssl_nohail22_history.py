"""Original WRF NSSL field names/metadata on actual emitted22 history files.

The original18 samples provide shared Registry field values for transcription
only. They are not used as a22 physical trajectory reference.
"""
from datetime import datetime,timezone
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
_spec=importlib.util.spec_from_file_location('nssl22_writer_fixture',Path(__file__).resolve().parents[1]/'test_m7_netcdf_writer.py')
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
synthetic_case,writer_authority=_fixture.synthetic_case,_fixture.writer_authority

PROOF=Path(__file__).resolve().parents[2]/'proofs/t_mpout'
NAMES={'QNDROP':'Nc','QNCCN':'Nn','QNSNOW':'Ns','QNGRAUPEL':'Ng','QVGRAUPEL':'qvolg'}


@pytest.mark.parametrize('mode',('full','operational','subset'))
def test_original_registry_fields_emitted(tmp_path,mode):
    state,grid,nml=synthetic_case();nml['mp_physics']=22
    with np.load(PROOF/'cpu_history_samples.npz') as refs:
        expected={name:refs['mp18__'+name].copy() for name in NAMES}
    for name,leaf in NAMES.items():setattr(state,leaf,expected[name])
    for leaf in ('qh','Nh','qvolh','hail_acc'):setattr(state,leaf,None)
    authority=writer_authority(grid,'d01');start=datetime(2023,1,15,tzinfo=timezone.utc)
    subset=tuple(NAMES) if mode=='subset' else None
    payload=prepare_wrfout_payload(state,grid,nml,tmp_path/'history22.nc',domain='d01',
        domain_authority=authority,valid_time=start,run_start=start,lead_hours=0,
        full_variable_set=mode!='operational',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    schema=json.loads((PROOF/'cpu_history_schema.json').read_text())['mp18']['missing_schema']
    with Dataset(payload.target) as ds:
        assert set(NAMES) <= set(ds.variables)
        assert not {'QNCLOUD','QHAIL','QNHAIL','QVHAIL'} & set(ds.variables)
        for name,value in expected.items():
            var=ds[name]
            assert var.dtype==np.dtype('float32')
            assert var.dimensions==tuple(schema[name]['dimensions'])
            np.testing.assert_array_equal(var[0],value)
            for key,ref in schema[name]['attrs'].items():assert var.getncattr(key)==ref,(name,key)


@pytest.mark.parametrize('name',tuple(NAMES))
@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_nonfinite_held_species_refused(tmp_path,name,bad):
    state,grid,nml=synthetic_case();nml['mp_physics']=22
    with np.load(PROOF/'cpu_history_samples.npz') as refs:
        value=refs['mp18__'+name].copy()
    value.flat[0]=bad;setattr(state,NAMES[name],value)
    authority=writer_authority(grid,'d01');start=datetime(2023,1,15,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='nonfinite microphysics history field '+name):
        prepare_wrfout_payload(state,grid,nml,tmp_path/'bad22.nc',domain='d01',
            domain_authority=authority,valid_time=start,run_start=start,lead_hours=0,
            full_variable_set=True,variable_subset=(name,))
