"""Actual packaged readers against independently observed original WRF tables."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('packaged_original_tables',ROOT/'proofs/s2_mpmore/jensen55/reference_tables.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


@pytest.mark.parametrize('mode,dtype',[('fp32',np.float32),('fp64',np.float64)])
def test_actual_packaged_table_reader_BYTE_to_original(mode,dtype):
    from gpuwrf.physics.jensen_ishmael import tables
    manifest=json.loads(tables._ASSET.with_name('manifest.json').read_text())
    assert hashlib.sha256(tables._ASSET.read_bytes()).hexdigest()==manifest['asset_sha256']
    got=tables.get_tables(dtype);original=reference.load(ROOT,mode)
    assert set(got)==set(original)
    for name,value in got.items():
        expected=original[name]
        assert value.shape==expected.shape and value.dtype==expected.dtype
        assert value.tobytes()==expected.tobytes(),name
        assert np.isfinite(value).all()
