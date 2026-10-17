"""Audit actual WRF restart operands, never infer ingestion from SUCCESS."""
from pathlib import Path
import json
import sys
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

root = Path(sys.argv[1])
out = {}
with Dataset(root/'wrffdda_d01') as analysis:
    for arm in ('old','repaired'):
        rows = []
        for stamp in ('18','36','54'):
            p = root/arm/f'wrfrst_d01_2023-01-15_00:00:{stamp}'
            with Dataset(p) as restart:
                values = {}
                for field in ('U','V','T','Q','PH'):
                    name = field+'_NDG_NEW'
                    got = np.asarray(restart[name][0])
                    wanted = np.asarray(analysis[name][0])
                    crop = tuple(slice(0,n) for n in got.shape)
                    expected = wanted[crop]
                    values[name] = {'shape':got.shape,'nonzero':int(np.count_nonzero(got)),
                                    'byte_equal':got.tobytes() == expected.tobytes(),
                                    'max_abs':float(np.max(np.abs(got-expected)))}
                rows.append({'step_seconds':int(stamp),'fields':values})
        out[arm] = rows
out['missing_metadata_killed_ingestion'] = all(v['nonzero']==0 and not v['byte_equal']
    for row in out['old'] for v in row['fields'].values())
out['repaired_ingested_byte'] = all(v['byte_equal'] for row in out['repaired'] for v in row['fields'].values())
out['all13_forcing_payloads_byte_equal'] = json.loads((root/'PREPARATION.json').read_text())['all13_variable_payloads_byte_equal']
(root/'INGESTION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('old','repaired')},indent=2))
assert out['missing_metadata_killed_ingestion'] and out['repaired_ingested_byte']
