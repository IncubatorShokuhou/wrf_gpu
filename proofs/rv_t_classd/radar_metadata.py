import sys,json
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/snapshot')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
from test_m7_netcdf_writer import synthetic_case,writer_authority
records=[]
for case,mp,folder in [('c14',19,'<USER_HOME>/wrf_gpu2_lanes/v-prep/c14/cpu'),
                       ('c15',21,'<USER_HOME>/wrf_gpu2_lanes/v-prep/c15/cpu'),
                       ('c08r',10,'<USER_HOME>/wrf_gpu2_lanes/t-classd/c08r/run')]:
    path=sorted(Path(folder).glob('wrfout_d01_*'))[1]
    with Dataset(path) as ds:
        v=ds['REFL_10CM'];want=dict(dims=list(v.dimensions),dtype=str(v.dtype),attrs={key:str(v.getncattr(key)) for key in v.ncattrs()})
    state,grid,nl=synthetic_case();nl.update(mp_physics=mp,do_radar_ref=1)
    auth=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,ROOT.parent/'radar_writer'/case,domain='d01',domain_authority=auth,
        valid_time='2023-01-15T01:00:00',run_start='2023-01-15T00:00:00',lead_hours=1.,
        full_variable_set=True,variable_subset=('REFL_10CM',),diagnostics={'REFL_10CM':np.zeros(state.qv.shape,np.float32)})
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=auth,variable_subset=('REFL_10CM',))
    with Dataset(payload.target) as ds:
        v=ds['REFL_10CM'];got=dict(dims=list(v.dimensions),dtype=str(v.dtype),attrs={key:str(v.getncattr(key)) for key in v.ncattrs()})
    mismatches={key:[value,got['attrs'].get(key)] for key,value in want['attrs'].items() if value!=got['attrs'].get(key)}
    records.append(dict(case=case,field='REFL_10CM',dimensions_match=want['dims']==got['dims'],dtype_match=want['dtype']==got['dtype'],attribute_mismatches=mismatches))
    assert not mismatches and want['dims']==got['dims'] and want['dtype']==got['dtype']
(ROOT.parent/'RV12_radar_metadata.json').write_text(json.dumps(records,indent=2)+'\n')
print(records)
