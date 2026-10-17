"""Replay ONLY the corrected host serialization of a genuine GPU initial frame.

The original file is preserved. All its fields/attributes must remain byte exact;
only U10E/V10E from the final WRF physics_init producer are added to a new file.
This is a disclosed serialization replay, never a new GPU or physics run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS, _write_float_variable, _publish_wrfout_noreplace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(case: Path, target: Path):
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    raw=case/'gpu/wrfout_d01_2023-01-15_00:00:00'
    original_sha=sha(raw)
    assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.replay-tmp')
    with Dataset(raw) as ds:
        ny,nx,nz=(len(ds.dimensions[n]) for n in ('south_north','west_east','bottom_top'))
    writer=object.__new__(pipeline._PerDomainWrfoutWriter)
    writer.input_dir=case/'input'
    writer.bundles={'d01':SimpleNamespace(grid=SimpleNamespace(nx=nx,ny=ny,nz=nz),
                                         namelist=SimpleNamespace(sf_sfclay_physics=2))}
    fields=writer._initial_surface_fields('d01')
    shutil.copyfile(raw,temp)
    try:
        with Dataset(temp,'r+') as ds:
            dims={n:None if d.isunlimited() else len(d) for n,d in ds.dimensions.items()}
            for name in ('U10E','V10E'):
                assert name not in ds.variables
                _write_float_variable(ds,WRFOUT_VARIABLE_SPECS[name],fields[name],dims)
        preserved=[]
        with Dataset(raw) as old,Dataset(temp) as new,Dataset(case/'cpu'/raw.name) as ref:
            old.set_auto_mask(False);new.set_auto_mask(False);ref.set_auto_mask(False)
            assert set(new.variables)-set(old.variables)=={'U10E','V10E'}
            assert old.ncattrs()==new.ncattrs()
            for attr in old.ncattrs(): np.testing.assert_array_equal(old.getncattr(attr),new.getncattr(attr))
            for name in old.variables:
                a,b=old[name],new[name]
                assert a.dimensions==b.dimensions and a.dtype==b.dtype and a.ncattrs()==b.ncattrs(),name
                assert np.asarray(a[:]).tobytes()==np.asarray(b[:]).tobytes(),name
                for attr in a.ncattrs(): np.testing.assert_array_equal(a.getncattr(attr),b.getncattr(attr))
                if np.issubdtype(b.dtype,np.number): assert np.isfinite(b[:]).all(),name
                preserved.append(name)
            for name in ('U10E','V10E'):
                assert np.isfinite(new[name][:]).all() and np.isfinite(ref[name][:]).all()
                assert np.asarray(new[name][:]).tobytes()==np.asarray(ref[name][:]).tobytes()
                assert new[name].dimensions==ref[name].dimensions and new[name].units==ref[name].units
        assert sha(raw)==original_sha
        with temp.open('rb') as f: os.fsync(f.fileno())
        _publish_wrfout_noreplace(temp,target)
    finally:
        temp.unlink(missing_ok=True)
    result={'scope':'HOST_SERIALIZATION_REPLAY_ONLY; original GPU physics/output preserved',
            'raw_original':str(raw),'raw_sha256':original_sha,'raw_unchanged':sha(raw)==original_sha,
            'replayed':str(target),'replayed_sha256':sha(target),
            'producer_source':pipeline.__file__,'all_original_fields_BYTE':preserved,
            'all_original_metadata_BYTE':True,'new_fields_original_CPU_BYTE':['U10E','V10E'],
            'scientific_admission':'PENDING_MANAGER_DECISION'}
    (target.parent/'REPLAY_PROOF.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'preserved_fields':len(preserved),'new_fields_CPU_BYTE':2,'raw_unchanged':True}))


if __name__=='__main__':
    replay(Path(sys.argv[1]),Path(sys.argv[2]))
