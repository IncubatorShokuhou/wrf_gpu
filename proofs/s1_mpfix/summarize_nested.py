"""Conservative admission decision from the unchanged WN3 PLAN/D6 receipt."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

path=Path(sys.argv[1]).resolve()
data=json.loads(path.read_text())
failures=[]; frames=[]
for frame in data['frames']:
    f2=np.asarray(frame.get('f2_counts',[]))
    guards=np.asarray(frame.get('census_guards',[]))
    negatives=np.asarray(frame.get('f2_min_negative',[]))
    frames.append(dict(domain=frame['domain'],step=frame['own_step'],
        census_present=all(frame.get(k) is not None for k in
            ('f2_counts','census_guards','water_changes','unknown_water_changes')),
        nonfinite=frame['nonfinite_carry_values'],guards=int(guards.sum()),
        f2_events=int(f2.sum()),minimum_negative=float(negatives.min()) if negatives.size else None,
        policy_water_abs_kg=float(np.abs(frame.get('water_changes',[])).sum()),
        unknown_water_abs=float(np.abs(frame.get('unknown_water_changes',[])).sum())))
    for name,row in frame['fields'].items():
        if row.get('status')=='MISSING_REFERENCE':
            if name not in ('QNICE','QNRAIN'):
                failures.append(dict(domain=frame['domain'],step=frame['own_step'],field=name,status='MISSING_REFERENCE'))
        for region,stats in row.get('regions',{}).items():
            if stats['gate']=='FAIL':
                failures.append(dict(domain=frame['domain'],step=frame['own_step'],field=name,region=region,**stats))
complete=data['status']=='COMPLETE' and data.get('own_steps')=={'d01':2,'d02':6}
finite=all(row['nonfinite']==0 for row in frames)
zero_guards=all(row['guards']==0 for row in frames)
zero_f2=all(row['f2_events']==0 for row in frames)
zero_water=all(row['policy_water_abs_kg']==0 and row['unknown_water_abs']==0 for row in frames)
census_present=all(row['census_present'] for row in frames)
passed=complete and finite and zero_guards and zero_f2 and zero_water and census_present and not failures
print(json.dumps(dict(status='PASS' if passed else 'REFUSED',
    result=str(path),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    mp=data['reference_plan']['mp_physics'],complete=complete,finite=finite,
    census_present=census_present,zero_guards=zero_guards,zero_unjustified_f2_events=zero_f2,zero_policy_water=zero_water,
    numerical_regions_pass=not failures,failures=failures,frames=frames,
    wall_s=data.get('wall_s'),namelists=data.get('namelists'),
    persistent_state_bytes=data.get('persistent_state_bytes'),extra_boundary_bytes=data.get('extra_boundary_bytes'),
    scope='Real WN3 homogeneous simultaneous-start N2/N6; frozen original PLAN/D6; CPU release interpretation, no GPU or 24h qualification.'),indent=2))
