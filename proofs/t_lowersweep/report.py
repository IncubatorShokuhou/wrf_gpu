"""Seal a complete preflight table, retaining UNKNOWN and control distinctions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('sweep',type=Path)
p.add_argument('out',type=Path)
p.add_argument('--candidate-commit',help='Final main commit covered by the selected-program reuse manifest.')
a=p.parse_args()
matrix=json.loads((a.sweep/'MATRIX_SNAPSHOT.json').read_text())
summary=json.loads((a.sweep/'RESULTS.json').read_text())
observed={r['case']:r for r in summary['rows']}
assert set(observed)=={c['id'] for c in matrix['cases']}, 'not every matrix row completed'
rows=[]
for case in matrix['cases']:
    r=observed[case['id']]
    source=Path(r.get('artifact') or a.sweep/case['id'])/'receipt.json'
    original=json.loads(source.read_text()) if source.exists() else {}
    alive=r['lower'] in ('NO_RECEIPT','RESOURCE_OR_PROCESS_FAIL','TIMEOUT','RUNNING','NOT_STARTED',
                        'OOM_AT_CAP','EARLY_EXIT','PROCESS_FAIL','HOST_REBOOT_INTERRUPTION')
    programs=[]
    for pr in r.get('programs',[]):
        for key in ('lower_s','compile_s'):
            if key in pr:assert math.isfinite(pr[key]) and pr[key]>=0
        programs.append({k:pr[k] for k in ('name','lower','lower_s','compile','compile_s','stack',
            'static_kwargs','stablehlo_bytes','stablehlo_sha256','triton_custom_calls','native_bytes',
            'ir_representation','compact_ir_sha256','compact_ir_bytes','full_stablehlo_sha256',
            'dt_s','run_physics','n_steps','cadence','start_step') if k in pr})
        if 'estimated_e41' in pr:
            gate=pr['estimated_e41']
            programs[-1]['estimated_e41']={k:gate[k] for k in
                ('e41_gate','hard_stack_gate','conservative_flags','coverage')}
            programs[-1]['max_stack_B']=max((e['max_stack_B'] for en in gate['entries'] for e in en['elfs']),default=None)
        if 'static_kwargs' in programs[-1]:
            programs[-1]['keyword_arguments']=programs[-1].pop('static_kwargs')
    rows.append(dict(case=case['id'],matrix_status=case['status'],lower=r['lower'],compile=r['compile'],
        source_commit=original.get('source_commit'),input_sha256=original.get('input_sha256'),
        target_sha256=original.get('target_sha256'),harness_sha256=original.get('harness_sha256'),
        native_trace=original.get('native_trace'),programs=programs,exception=r.get('exception'),
        process_result=r.get('process_result'),disposition=r.get('disposition'),
        prior_attempt=r.get('prior_attempt'),reuse_scope=r.get('reuse_scope'),
        hlo_constants=original.get('hlo_constants'),
        init_cache_clear_control=original.get('init_cache_clear_control'),
        traced_pallas_calls=original.get('traced_pallas_calls'),
        traced_interpreted_pallas_calls=original.get('traced_interpreted_pallas_calls'),
        physical_devices_at_exit=original.get('physical_devices_at_exit'),phase=original.get('phase'),
        peak_process_rss_KiB=None if alive else r.get('peak_rss_KiB'),
        last_reported_process_rss_KiB=r.get('last_progress_rss_KiB',r.get('peak_rss_KiB')) if alive else None,
        wall_s=r.get('wall_s'),returncode=r.get('returncode'),dfi_prepass=original.get('dfi_prepass'),
        timeseries_seeded=original.get('timeseries_seeded',False),
        regional_streams_seeded=original.get('regional_streams_seeded',False),
        receipt=str(source),receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None))
source=Path(summary['source'])
repo=Path(__file__).resolve().parents[2]
commit=a.candidate_commit or summary.get('candidate_commit')
if commit is None:
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
tree=subprocess.check_output(['git','-C',str(repo),'rev-parse',commit+':src/gpuwrf'],text=True).strip()
record=dict(utc=datetime.now(timezone.utc).isoformat(),matrix_rows=len(rows),
    matrix_sha256=hashlib.sha256((a.sweep/'MATRIX_SNAPSHOT.json').read_bytes()).hexdigest(),
    measured_source_commit=commit,measured_src_gpuwrf_tree=tree,
    measured_source_counts=dict(Counter(r['source_commit'] for r in rows)),
    source_scope='Candidate source plus per-row exact-source/selected-program reuse; actual measured commits retained per row.',
    lower_counts=dict(Counter(r['lower'] for r in rows)),compile_counts=dict(Counter(r['compile'] for r in rows)),
    admission_scope='Every matrix row was attempted; non-ADMITTED rows are controls, not runnable/qualified schemes.',
    limitations=['CPU frontend / optional AUTOTUNE0 native prescreen only; no GPU, fidelity or forecast-speed claim.',
        'STACK is UNKNOWN unless a native executable was compiled and scanned.',
        'DFI initial actual stage is lowered before any model advance; later filtered stages are NOT_CHECKED.',
        'Chunk count/cadence keywords are traced scalars in the product, not static compiler parameters.',
        'Process-killed cases retain last-progress RSS separately; it is not a final peak.'],rows=rows)
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'RESULTS.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
lines=['# CPU CUDA preflight', '',f'Source `{commit}`; {len(rows)} sealed matrix rows. No GPU execution or fidelity claim.',
       '', '| Case | Matrix status | CUDA lower | Compile s/status | Peak process MiB | Estimated STACK | Exception |',
       '|---|---|---|---|---:|---|---|']
for r in rows:
    comp=sum(pr.get('compile_s',0) for pr in r['programs'])
    stack=','.join(str(pr.get('max_stack_B',pr.get('stack','UNKNOWN'))) for pr in r['programs']) or 'UNKNOWN'
    lines.append('| '+' | '.join((r['case'],r['matrix_status'],r['lower'],
        f'{comp:.2f}' if r['compile']=='OK' else r['compile'],
        f"{r['peak_process_rss_KiB']/1024:.1f}" if r['peak_process_rss_KiB'] is not None else 'UNKNOWN',
        stack,(r['exception'] or '').replace('|','/')[:240]))+' |')
lines.extend(['','DFI: first actual stage only; later filtered-stage lowering remains unmeasured.',
              'Compile-only E41 estimates cannot replace the real GPU E41 gate.'])
(a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:record[k] for k in ('matrix_rows','lower_counts','compile_counts')}))
