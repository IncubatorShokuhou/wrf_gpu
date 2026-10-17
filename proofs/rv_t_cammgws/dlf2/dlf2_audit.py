"""Independent read-only CPU-WRF inventory and a metadata-deletion-sensitive mutant."""
import dataclasses
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
root=Path(sys.argv[1]);out=Path(__file__).parent
sys.path[:0]=[str(root/'src'),str(root/'tests')]
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
proof=json.loads((root/'proofs/t_cammgws/DLF2_CPU_ALL_FRAMES.json').read_text())
audit=json.loads((root/'proofs/t_cammgws/DLF2_SOURCE_AUDIT.json').read_text())
wrf=Path(audit['pristine']);cpu=Path(proof['source'])
source_hashes={name:hashlib.sha256((wrf/name).read_bytes()).hexdigest()==sha for name,sha in audit['files'].items()}
assert all(source_hashes.values())
frames=[];cells=0
paths=sorted(cpu.glob('wrfout_d01_*'))
assert len(paths)==25
for path in paths:
    with Dataset(path) as ds:
        var=ds['DLF2'];data=np.asarray(var[:]);bits=data.view(np.uint32)
        assert data.dtype==np.dtype('float32') and list(data.shape)==[1,44,42,42]
        assert list(var.dimensions)==proof['field']['dimensions']
        attrs={k:str(var.getncattr(k)) for k in var.ncattrs()}
        expected={k:v for k,v in proof['field'].items() if k not in ('dimensions','dtype')}
        assert attrs==expected,(path,attrs,expected)
        for k,v in proof['global_attributes'].items():assert str(ds.getncattr(k))==v
        assert np.all(bits==0),path
        cells+=data.size;frames.append(path.name)
assert frames==[x['file'] for x in proof['frames']]

tests=runpy.run_path(str(root/'tests/v040/test_cammg_dlf2_history.py'))
from gpuwrf.io import wrfout_writer as writer
with tempfile.TemporaryDirectory(dir=out) as tmp:
    tests['test_writer_emits_original_bits_and_metadata'](Path(tmp),False)
original=writer.WRFOUT_VARIABLE_SPECS['DLF2']
writer.WRFOUT_VARIABLE_SPECS['DLF2']=dataclasses.replace(original,units='kg/kg')
with tempfile.TemporaryDirectory(dir=out) as tmp:
    try:tests['test_writer_emits_original_bits_and_metadata'](Path(tmp),False)
    except AssertionError as exc:
        assert str(exc)=='units',str(exc)
        mutant={'result':'KILLED','exception':'AssertionError','message':'units','setup_errors':0}
    else:raise AssertionError('DLF2 unit-loss mutant survived')
writer.WRFOUT_VARIABLE_SPECS['DLF2']=original
result={'candidate':'3eb96246c8','pristine_source_hashes':source_hashes,
        'CPU_reference':str(cpu),'frames':frames,'frame_count':len(frames),'cells_positive_zero':cells,
        'field_metadata':proof['field'],'all_headers_match':True,'all_bits_positive_zero':True,'units_mutant':mutant}
(out/'DLF2_REVIEW_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
