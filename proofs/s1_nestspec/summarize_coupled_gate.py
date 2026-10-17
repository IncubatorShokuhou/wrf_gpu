"""Report completed and failed regions explicitly; process exit alone is no gate."""
from pathlib import Path
import json
import sys
import numpy as np

path = Path(sys.argv[1]).resolve()
report = json.loads(path.read_text())
summary = {'input':str(path),'status':report['status'],'frames':[],'failures':[],
           'own_steps':report.get('own_steps'), 'wall_s':report.get('wall_s')}
for frame in report['frames']:
    failures = []
    for field, row in frame['fields'].items():
        if row.get('status')=='MISSING_REFERENCE':
            # QNI/QNR are common port placeholders even in original schemes
            # that do not allocate those Registry members. Preserve inventory.
            if field not in ('QNICE','QNRAIN'):
                failure = {'domain':frame['domain'],'own_step':frame['own_step'],
                           'field':field,'status':'MISSING_REFERENCE'}
                failures.append(failure)
                summary['failures'].append(failure)
            continue
        for region, stats in row['regions'].items():
            if stats['gate']=='FAIL':
                failure = {'domain':frame['domain'],'own_step':frame['own_step'],
                           'field':field,'region':region,**stats}
                failures.append(failure)
                summary['failures'].append(failure)
    summary['frames'].append({'domain':frame['domain'],'own_step':frame['own_step'],
        'failed_regions':len(failures),'nonfinite_values':frame['nonfinite_carry_values'],
        'guard_events':None if frame.get('census_guards') is None else int(np.asarray(frame['census_guards']).sum()),
        'pre_mask_f2_events':None if frame.get('f2_counts') is None else int(np.asarray(frame['f2_counts']).sum()),
        'policy_water_abs_kg':None if frame.get('water_changes') is None else float(np.abs(np.asarray(frame['water_changes'])).sum())})
summary['numerical_regions_pass'] = not summary['failures']
summary['completed_expected_steps'] = report.get('own_steps')=={'d01':2,'d02':6}
summary['finite_pass'] = all(row['nonfinite_values']==0 for row in summary['frames'])
summary['zero_guard_events'] = all(row['guard_events']==0 for row in summary['frames'])
summary['zero_pre_mask_f2_events'] = all(row['pre_mask_f2_events']==0 for row in summary['frames'])
summary['scope'] = 'Predeclared D6/PLAN comparison; default radii gap independently blocks18/24/26/28 admission.'
print(json.dumps(summary,indent=2))
