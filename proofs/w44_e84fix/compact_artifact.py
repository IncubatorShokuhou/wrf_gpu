"""Discard own disposable compile dumps after RN/PTX/E41 consumers finish.

Keep exact PTX and thunk text losslessly compressed, optimized protobuf/HLO,
all RN source lineage, E41 output and resource receipts. Never touches cases.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('artifact', type=Path)
a = p.parse_args()
assert not Path('/tmp/wrf_gpu2_quiet').exists()
root = Path('<USER_HOME>/wrf_gpu2_lanes/w44-e84fix').resolve()
out = a.artifact.resolve()
assert out.parent == root and out.name.startswith('E84'), out
assert json.loads((out / 'receipt.json').read_text())['phase'] == 'done'
if 'base' not in out.name:
    assert (out / 'RN_AUDIT.json').is_file(), 'RN/PTX proof must finish first'
dump = out / 'dump'
kept = []
removed = 0
for file in list(dump.iterdir()):
    if file.name.endswith(('.ptx', '.thunk_sequence.txt')):
        raw = file.read_bytes()
        packed = file.with_name(file.name + '.gz')
        packed.write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
        assert gzip.decompress(packed.read_bytes()) == raw
        kept.append(dict(name=packed.name, raw_sha256=hashlib.sha256(raw).hexdigest(),
                         raw_bytes=len(raw), compressed_bytes=packed.stat().st_size))
        file.unlink()
    elif file.name.endswith(('.ptx.gz', '.thunk_sequence.txt.gz')):
        pass
    elif file.is_dir():
        removed += sum(p.stat().st_size for p in file.rglob('*') if p.is_file())
        shutil.rmtree(file)
    else:
        removed += file.stat().st_size
        file.unlink()
(out / 'COMPACT.json').write_text(json.dumps(dict(kept=kept, removed_disposable_bytes=removed), indent=2)+'\n')
print(out.name, 'removed', removed, 'preserved exact PTX/thunks', len(kept))
