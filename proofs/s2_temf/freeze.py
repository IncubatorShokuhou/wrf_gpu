"""Pack small pristine column receipts as tracked, pickle-free test fixtures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def freeze(source,destination):
    d=json.loads(Path(source).read_text()); records=d.pop('records')
    arrays={'names':np.asarray([r['name'] for r in records]),'steps':np.asarray([r['step'] for r in records],np.int32)}
    for group in ('inputs','outputs'):
        for k in records[0][group]:
            if k=='carry':
                for ck in records[0][group][k]:
                    arrays[f'{group}__carry__{ck}']=np.asarray([r[group][k][ck] for r in records],np.int32 if ck=='kpbl' else np.float32)
            else:
                arrays[f'{group}__{k}']=np.asarray([r[group][k] for r in records],np.int32 if k=='kpbl' else np.float32)
    dest=Path(destination)
    np.savez_compressed(dest,**arrays)
    d.update(fixture_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),record_count=len(records),
             source_json_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    dest.with_suffix('.json').write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':
    freeze(*sys.argv[1:])
