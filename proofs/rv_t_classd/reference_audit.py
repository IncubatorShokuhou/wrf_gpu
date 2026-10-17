import sys,json,hashlib,glob,runpy
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/snapshot')
OUT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
from test_m7_netcdf_writer import synthetic_case,writer_authority

groups=[('c57',3,8,('QGR_URB4D','TGR_URB4D','DRAINGR_URB3D','EP_PV_URB3D')),
        ('c19',0,28,('QNBCA','TAOD5502D')),
        ('c65',0,38,('QNBCA','TAOD5502D')),
        ('c20',0,40,('MSKF_REFL_10CM',))]
records=[]
for case,urban,mp,names in groups:
    files=sorted(Path('<USER_HOME>/wrf_gpu2_lanes/v-prep',case,'cpu').glob('wrfout_d01_*'))
    assert len(files)==25
    with Dataset(files[0]) as ds:
        meta={name:dict(dims=list(ds[name].dimensions),dtype=str(ds[name].dtype),attrs={key:str(ds[name].getncattr(key)) for key in ds[name].ncattrs()}) for name in names}
    state,grid,nl=synthetic_case();nl.update(sf_urban_physics=urban,mp_physics=mp)
    qgr=np.arange(20,dtype=np.float32).reshape(4,5)/40
    diag={'QGR_URB4D':qgr} if urban else {}
    auth=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,OUT/'writer'/case,domain='d01',domain_authority=auth,
        valid_time='2023-01-15T01:00:00',run_start='2023-01-15T00:00:00',lead_hours=1.,
        full_variable_set=True,variable_subset=names,diagnostics=diag)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=auth,variable_subset=names)
    with Dataset(payload.target) as ds:
        for name in names:
            v=ds[name]
            got=dict(dims=list(v.dimensions),dtype=str(v.dtype),attrs={key:str(v.getncattr(key)) for key in v.ncattrs()})
            mismatches={key:[value,got['attrs'].get(key)] for key,value in meta[name]['attrs'].items() if value!=got['attrs'].get(key)}
            records.append(dict(case=case,name=name,dims_match=got['dims']==meta[name]['dims'],dtype_match=got['dtype']==meta[name]['dtype'],attribute_mismatches=mismatches))
    # Attest every original CPU frame against the producer's immutable receipt.
    proof=json.loads((ROOT/'proofs/t_classd'/('URBAN_CPU_ALL_FRAMES.json' if urban else 'CONSTANT_CPU_ALL_FRAMES.json')).read_text())
    frames=proof['frames'] if urban else proof['cases'][case]['frames']
    expected={f['file']:f['sha256'] for f in frames}
    for path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[path.name],path
(OUT/'RV05_metadata.json').write_text(json.dumps(records,indent=2)+'\n')
print('METADATA',json.dumps(records),flush=True)

sys.argv=['urban_cpu_all_frames.py','<USER_HOME>/wrf_gpu2_lanes/v-prep/c57',str(OUT/'RV05_urban_cpu.json')]
runpy.run_path(str(ROOT/'proofs/t_classd/urban_cpu_all_frames.py'),run_name='__main__')
sys.argv=['constant_cpu_all_frames.py',str(OUT/'RV05_constants_cpu.json')]+[f'<USER_HOME>/wrf_gpu2_lanes/v-prep/{c}:'+','.join(n) for c,u,m,n in groups if not u]
runpy.run_path(str(ROOT/'proofs/t_classd/constant_cpu_all_frames.py'),run_name='__main__')

# Binary operands for a fresh lane-private pristine oracle build.
cols=np.load(ROOT/'proofs/t_classd/morr_radar_oracle/fixtures/inputs.npz')['columns']
for tag,dt in [('fp32',np.float32),('fp64',np.float64)]:
    with (OUT/f'input_{tag}.bin').open('wb') as f:
        np.asarray([cols.shape[2],cols.shape[0]],np.int32).tofile(f)
        cols.astype(dt).tofile(f)
print('ORACLE_INPUTS',cols.shape,flush=True)
