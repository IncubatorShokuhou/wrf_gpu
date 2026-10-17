"""Deletion-sensitive tests of the all-frame inventory gate itself."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('writer_schema_gate',ROOT/'proofs/t_writer/compare_schema.py')
gate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(tmp_path):
    case=tmp_path/'refs/c00'
    (case/'cpu').mkdir(parents=True)
    (case/'input').mkdir()
    initial=case/'input/namelist.input'
    initial.write_text('&physics\n mp_physics=14,\n/\n')
    (case/'cpu_run.json').write_text(json.dumps(dict(status='COMPLETE',actual_input_sha256={
        initial.name:hashlib.sha256(initial.read_bytes()).hexdigest()})))
    (case.parent/'MATRIX.json').write_text(json.dumps(dict(cases=[dict(id='c00',domains=['d01'],hours=1)])))
    phases={}
    for h in (0,1):
        path=case/'cpu'/f'wrfout_d01_2023-01-15_0{h}:00:00'
        with Dataset(path,'w') as ds:
            ds.createDimension('Time',None)
            ds.createDimension('south_north',2)
            ds.createDimension('west_east',3)
            ds.MP_PHYSICS=np.int32(14)
            ds.createVariable('XTIME','f4',('Time',))[:]=[60*h]
            v=ds.createVariable('T2','f4',('Time','south_north','west_east'))
            v[0]=np.full((2,3),280+h,np.float32)
            v.units='K'
        with Dataset(path) as ds:
            phases[str(h)]={'d01':dict(dimensions={k:len(v) for k,v in ds.dimensions.items()},
                variables={k:dict(dimensions=list(v.dimensions),dtype=str(v.dtype),attrs={a:v.getncattr(a)
                    for a in v.ncattrs()}) for k,v in ds.variables.items()},globals={'MP_PHYSICS':14})}
    inv=tmp_path/'inventories/c00'
    inv.mkdir(parents=True)
    (inv/'inventory.json').write_text(json.dumps(dict(status='COMPLETE',phases=phases)))
    return case.parent,inv.parent,tmp_path/'result.json'


def run(args):
    root,inventories,output=args
    gate.compare(root,inventories,output,ROOT)
    return json.loads(output.read_text())['rows'][0]


def test_healthy_schema_control(tmp_path):
    row=run(fixture(tmp_path))
    assert row['status']=='PASS_CPU_SCHEMA_COVERAGE'
    assert row['frames']==2


@pytest.mark.parametrize('fault',['omit','late_units','late_dimension','late_global'])
def test_missing_and_late_frame_changes_cannot_pass(tmp_path,fault):
    args=fixture(tmp_path)
    if fault=='omit':
        path=args[1]/'c00/inventory.json'
        rec=json.loads(path.read_text())
        del rec['phases']['1']['d01']['variables']['T2']
        path.write_text(json.dumps(rec))
    else:
        path=sorted((args[0]/'c00/cpu').glob('wrfout*'))[-1]
        with Dataset(path,'r+') as ds:
            if fault=='late_units':ds['T2'].units='wrong'
            elif fault=='late_global':ds.MP_PHYSICS=np.int32(16)
            else:ds.createDimension('missing_dimension',7)
    row=run(args)
    assert row['status']=='FAIL'
    assert any('01:00:00' in f for e in row['errors'].values() for f in e['frames'])


def test_incomplete_reference_cannot_pass(tmp_path):
    args=fixture(tmp_path)
    sorted((args[0]/'c00/cpu').glob('wrfout*'))[-1].unlink()
    assert run(args)['status']=='INCOMPLETE'


def test_c53_one_frame_cannot_become_a_24_hour_qualification(tmp_path):
    root,inventories,output=fixture(tmp_path)
    (root/'c00').rename(root/'c53');(inventories/'c00').rename(inventories/'c53')
    matrix=json.loads((root/'MATRIX.json').read_text())
    matrix['cases'][0].update(id='c53',hours=24)
    (root/'MATRIX.json').write_text(json.dumps(matrix))
    receipt=root/'c53/cpu_run.json';data=json.loads(receipt.read_text());data['status']='FAILED'
    receipt.write_text(json.dumps(data))
    sorted((root/'c53/cpu').glob('wrfout*'))[-1].unlink()
    row=run((root,inventories,output))
    assert row['status'].startswith('INCOMPLETE') and row['frames']==1
    assert 'nominal 25' in row['reason'] and 'no 24-hour qualification' in row['reason']


def test_changed_actual_input_cannot_pass(tmp_path):
    args=fixture(tmp_path)
    (args[0]/'c00/input/namelist.input').write_text('wrong inputs')
    assert run(args)['status']=='FAIL'
