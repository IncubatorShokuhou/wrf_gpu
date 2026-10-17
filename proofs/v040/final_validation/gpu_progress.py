"""Mutually exclusive, source-bound scientific progress; no final-release inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
P = Path(__file__).resolve().parent
R = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
subprocess.run(['python', str(P/'gpu_status.py')], check=True, stdout=subprocess.DEVNULL)
status = json.loads((P/'GPU_STATUS.json').read_text())
allowed = {'c14': {'REFL_10CM'}, 'c15': {'REFL_10CM','REFD_MAX'},
    'c19': {'QNBCA','TAOD5502D'}, 'c20': {'MSKF_REFL_10CM'},
    'c57': {'DRAINGR_URB3D','QGR_URB4D','TGR_URB4D','EP_PV_URB3D'},
    'c65': {'QNBCA','TAOD5502D'}}
rows = status['rows']
for row in rows:
    c = row['case']
    ds = list(row['domains'].values())
    missing = {n for d in ds for n in d['missing_cpu_fields']}
    clean = all(d['bounded_misses']==0 and d['raw_output_nonfinite']==0 for d in ds)
    reason = []
    if row['full_d6_pass'] and not row['qh_mass_provisional'] and row['e41']=='PASS':
        category = 'full_D6_PASS'
    elif row['full_d6_pass'] and not row['qh_mass_provisional'] and row['e41']=='FAIL':
        category = 'PASS_with_disclosure'
        reason = ['Actual E41 FAIL: opt-in disclosure; hard gate unchanged']
    elif row['forecast_complete'] and clean and c in allowed and missing and missing<=allowed[c]:
        category = 'PASS_with_disclosure'
        reason = ['Named uncomputed fields: '+','.join(sorted(missing)),
                  'Raw full-inventory D6 remains INCOMPLETE', 'E41 '+str(row['e41'])]
    elif row.get('accepted_scope_gate') and row['accepted_scope_gate'].get('accepted_D6')=='PASS':
        category = 'PASS_with_disclosure'
        reason = ['Manager-approved complete6h; full24h missingh24 disclosed']
    elif c!='c01' and (row['resident_nonfinite_failure'] or any(
            (d['bounded_misses'] or 0)>0 or (d['raw_output_nonfinite'] or 0)>0 for d in ds)):
        category = 'FAIL'
        reason = ['Original CPU-WRF frozen D6 or prognostic nonfinite failure']
    else:
        category = 'still_to_close'
        reason = ['Remaining forecast, inventory, or provenance/writer closure']
        if c=='c01':
            reason = ['Superseded raw95 surface misses retained; corrected paid producer/soil writer binding pending']
        if missing:
            reason.append('Unclosed inventory: '+','.join(sorted(missing)))
    row['progress_category'] = category
    row['progress_disclosures_or_pending_reason'] = reason
cats = {k:[r['case'] for r in rows if r['progress_category']==k] for k in
        ('full_D6_PASS','PASS_with_disclosure','FAIL','still_to_close')}
assert sum(map(len,cats.values()))==len(rows)
now = datetime.now(timezone.utc)
left = max(0,round((now.replace(hour=18,minute=45,second=0,microsecond=0)-now).total_seconds()/60,1))
pending = set(cats['still_to_close'])
groups = [['c24','c10','c27'],['c83','c47','c56'],['c57','c73','c08'],
          ['c19','c20'],['c77','c78']]
wsmdz_groups = [['c04','c05','c06'], ['c10','c11','c18']]
wsmdz = {c for g in wsmdz_groups for c in g}
ordinary = (sum(bool((pending-wsmdz).intersection(g)) for g in groups) +
            sum(bool(pending.intersection(g)) for g in wsmdz_groups))
special = len(pending.intersection({'c28','c65','c60','c84','c85'}))
compiler = len(pending.intersection({'c87','c88'}))
estimate = [15*(ordinary+special),30*(ordinary+special+compiler)]
counts = {k:len(v) for k,v in cats.items()}
summary = (f"{len(rows)} composites / {counts['full_D6_PASS']} full-D6 PASS / "
    f"{counts['PASS_with_disclosure']} PASS-with-disclosure / {counts['FAIL']} FAIL "
    f"({','.join(cats['FAIL'])}) / {counts['still_to_close']} pending "
    f"({','.join(cats['still_to_close'])}) / [I]{estimate[0]}–{estimate[1]} GPU-min "
    f"+ unknown repairs vs {left:.0f}min to18:45Z; source-bound/F2OPEN")
result = dict(schema='v040-v-prep-gpu-progress-v2',utc=now.isoformat(),
    active_composites=len(rows),**counts,category_case_ids=cats,
    full_D6_numerical_passes_including_E41_disclosed=status['full_d6_passes'],
    accepted_short_scope_passes=status['accepted_short_scope_passes'],
    full_forecast_completed=status['full_forecasts'],calendar_minutes_until_window_close=left,
    window_end_utc='2026-10-15T18:45:00Z',last_24h_start_utc='2026-10-15T18:15:00Z',
    estimated_remaining_allocated_gpu_minutes=estimate,
    repair_gpu_minutes='UNKNOWN; failed physics waits for merged fixes, no unchanged repeats',
    estimate_scope='[I]15–30min per remaining ordinary cohort/singleton; compiler cases0–30 conditional each. Includes active whole cohort cap conservatively. Foreign queue and new science repairs excluded. NotS1.',
    scope='Measured-source science evidence, not blanket latest-main/release qualification. Full-D6 PASS requires full forecast/inventory/E41PASS; all F2 remainsOPEN. Named missing fields and accepted scope limits disclosed. Raw writer failures remain preserved beside hash-bound corrected-publication gates.',
    summary_line=summary,
    source_status=dict(path=str(P/'GPU_STATUS.json'),
        sha256=hashlib.sha256((P/'GPU_STATUS.json').read_bytes()).hexdigest()), rows=rows)
for path in (R/'GPU_PROGRESS.json',P/'GPU_PROGRESS.json'):
    temporary = path.with_suffix('.json.v-prep-tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(path)
print(summary)
if '--mailbox' in sys.argv:
    with Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md').open('a') as f:
        f.write(now.isoformat()+' v-prep: '+summary+'\n')
