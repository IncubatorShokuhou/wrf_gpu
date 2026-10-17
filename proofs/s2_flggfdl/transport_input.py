"""Freeze an exact matched-input transfer replay before measuring compiler spread."""
from pathlib import Path
import json
import sys

import numpy as np

p=Path(sys.argv[1]);out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
with np.load(p,allow_pickle=False) as a:
    nr,nz,_=a['transfer_optics'].shape
    parameters=np.stack([a['transfer_'+k] for k in ('albedo','coszen','f0','emissivity','bs')],axis=-1).astype(np.float32)
    with out.open('wb') as f:
        np.asarray([nr,nz],np.int32).tofile(f)
        for values in (a['transfer_band'].astype(np.int32),parameters,a['transfer_bf'],a['transfer_optics']):
            values.ravel(order='F').tofile(f)
out.with_suffix('.preregistered.json').write_text(json.dumps({
    'rows':nr,'levels':nz,'source_operand_fixture':str(p),
    'literal_gate':'no-FMA original WRF replay must be BYTE to FG07 for every row/profile',
    'compiled_bound':'for each matched row, original WRF FMA-vs-no-FMA profile Linf spread plus the original 16 reference ulps + 3e-6 W/m2 allowance; inputs/rows/reference unchanged',
    'authority':'main _WAVE_S1_COMMON compiler contraction/FMA standing policy, reread after manager notice',
    'no_source_deviation_waiver':'This is source-measured compiler spread, never a bound fitted to JAX output; report any excess as FAIL or disclosed deviation pending D6',
},indent=2)+'\n')
