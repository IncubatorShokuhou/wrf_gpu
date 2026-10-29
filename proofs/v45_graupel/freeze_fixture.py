"""Freeze compact G01/G02/G05 operands and pristine outputs, not history files."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LANE=Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel')
DEST=ROOT/'tests/v025/b_thompson/fixtures/graupel_entry_sedimentation.npz'
source=dict(np.load(LANE/'G01/columns.npz'))
baseline=dict(np.load(LANE/'G02/replay.npz'))
sweep=dict(np.load(LANE/'G05/probe.npz'))
pick=[i for i,n in enumerate(source['names']) if str(n).startswith('cpu_h72')]
out={k:np.concatenate((v,source[k][pick])) for k,v in sweep.items() if k.startswith('in_') or k in ('names','dt')}
for key,v in sweep.items():
    if key.startswith('wrf_'):out[key]=np.concatenate((v,baseline[key][pick]))
    elif key.startswith('old_'):out[key]=np.concatenate((v,baseline['port_'+key[4:]][pick]))
out['names'][-len(pick):]=[str(source['names'][i])+'_entry_active' for i in pick]
np.savez_compressed(DEST,**out)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=dict(fixture_id='v45-graupel-entry-sedimentation',source='original WRF 4.7.1 mp_gt_driver',
    source_commit='f52c197ed39d12e087d02c50f412d90d418f6186',scenario='WN3 0227 d03 h67-72 real columns + explicit no-qg/density controls',
    created_utc='2026-10-27',files={DEST.name:sha(DEST)},
    provenance={str(p.relative_to(LANE)):sha(p) for p in
        [LANE/'G01/columns.npz',LANE/'G01/census.json',LANE/'G02/replay.npz',LANE/'G05/probe.npz',LANE/'G02/driver_halo_identity.json']},
    variables=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype),units=('kg/kg' if k.endswith(('qv','qc','qr','qi','qs','qg')) else
        'K' if k.endswith('th') else 'Pa' if k.endswith('_p') else 'm' if k.endswith('dz8w') else 'see mp_gt_driver'),
        absolute_tolerance='existing MIX increment floor',relative_tolerance='existing MIX 1%, Ni 5%')
        for k,v in out.items() if k.startswith(('in_','wrf_'))],
    trace_gate='exact qg zero/nonzero mask equality per cell; finite values; unchanged WRF R1=1e-12',
    columns=len(out['names']),entry_active_control_indices=list(range(len(out['names'])-len(pick),len(out['names']))),
    license_notes='Small derived numerical fixture from original WRF and locally owned GPU output; no WRF history file included.')
DEST.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('fixture',manifest['columns'],DEST.stat().st_size,'bytes',manifest['files'])
