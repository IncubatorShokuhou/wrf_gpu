"""Verify frozen tool/input bytes; never compare persisted device/inode values."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'CUDA_MINI_MANIFEST.json').read_text())
for relative,expected in manifest['files'].items():
    path=root/relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
case=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/cuda_mini_case_v1')
for name,expected in manifest['mini_inputs'].items():
    assert hashlib.sha256((case/name).read_bytes()).hexdigest()==expected,name
print('CONTENT SEAL PASS',len(manifest['files']),'tool files,',len(manifest['mini_inputs']),'mini inputs')
