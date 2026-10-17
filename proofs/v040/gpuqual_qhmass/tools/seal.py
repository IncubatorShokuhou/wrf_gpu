"""Seal completed measurements; no model execution or gate redefinition."""
import gzip, hashlib, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round4_qhmass')
OUT=Path('<USER_HOME>/src/wrf_gpu2_wt/v-run/proofs/v040/gpuqual_qhmass')
MAIN=Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
assert os.environ['JAX_PLATFORMS']=='cpu' and os.sched_getaffinity(0)=={9}
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert (ROOT/'CONSUMERS_DONE.json').exists()
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
plan=json.loads((ROOT/'RUN_PLAN.json').read_text())
OUT.mkdir(parents=True,exist_ok=True)
rows=[]; raw=[]
for c in plan['configs']:
    n=c['config']; case=ROOT/n; dest=OUT/n; dest.mkdir(exist_ok=True)
    gpu=json.loads((case/'gpu_run.json').read_text())
    d=json.loads((case/'grid_gate.json').read_text())
    assert d['manifest_sha256']=='9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc'
    e=json.loads((case/'e41/wave_gate.json').read_text())
    a=json.loads((case/'WRITER_AUDIT.json').read_text())
    assert gpu['source_rev']==plan['source_rev'] and gpu['platform']=='gpu' and gpu['rc']==0
    assert json.loads((case/'cleanup.json').read_text())['status']=='DONE'
    assert e['coverage']['status']=='PASS'
    counts=json.loads((case/'gpu/f2_negative_summary.json').read_text())
    records=[dict(domain=domain,**r) for domain,dom in counts['domains'].items() for r in dom['records']]
    f2={'status':'OPEN','records':len(records),'fields':sorted({r['field'] for r in records}),'phases':sorted({r['phase'] for r in records}),
        'qh_counter_record_present':any(r['field']=='qh' for r in records),'negative_phase_field_visits':sum(r['negative_count'] for r in records),
        'interior_phase_field_visits':sum(r['interior_count'] for r in records),'raw_counter_nonfinite':sum(r['nonfinite_count'] for r in records),
        'nonzero_records':[r for r in records if r['negative_count'] or r['nonfinite_count']],
        'interpretation':'Repeated phase-field visits, not unique events. No every-event source/magnitude justification; no transfer of another-case disposition.'}
    write(case/'F2_DISCLOSURE.json',f2)
    elfs=[z for entry in e['entries'] for z in entry['elfs']]
    flags=sorted({(k,v) for entry in e['entries'] for k,v in entry.get('kinput_transpose_over500',[])})
    nonleaf=[r for elf in elfs for r in elf['nonleaf_over1k']]
    row={'config':n,'source_rev':gpu['source_rev'],'GPU_rc':gpu['rc'],'frames':d['paired_frames'],
        'CPU_fields':d['cpu_field_count'],'bounded_fields':d['bounded_field_count'],'D6':'FAIL' if d['failures'] else 'PASS',
        'coverage_complete':d['coverage_complete'],'verdict':d['verdict'],'first_failure':d['first_failure'],
        'missing_CPU_fields':sorted({k for frame in d['by_frame'] for k in frame['missing_vs_cpu']}),
        'raw_output_nonfinite':d['raw_gpu_nonfinite'],'E41':e['e41_gate'],'E41_coverage':e['coverage'],
        'max_stack_B':max(z['max_stack_B'] for z in elfs),'kinput_flags':flags,'nonleaf_over1k':nonleaf,
        'variable_metadata_mismatches':a['variable_metadata_mismatch_count'],
        'different_globals':sorted({z['attribute'] for f in a['by_frame'] for z in f['global_mismatches']}),
        'F2':{k:v for k,v in f2.items() if k!='nonzero_records'},'wall_s':gpu['wall_s'],
        'host_peak_rss_kib':gpu['host_peak_rss_kib'],'sampled_peak_vram_mib':gpu['peak_process_vram_mib_sampled'],
        'report_only_fields':{k:{x:v.get(x) for x in ('verdict','rmse','max_abs','frame_count')} for k,v in d['fields'].items() if k in ('REFL_10CM','REFD_MAX')}}
    rows.append(row)
    for fn in ('inputs.json','launch.json','GPU_DONE.json','gpu_run.json','RESULT.json','grid_gate.json','WRITER_AUDIT.json','cleanup.json','F2_DISCLOSURE.json'):
        shutil.copyfile(case/fn,dest/fn)
    for fn in ('nested_pipeline_run.json',): shutil.copyfile(case/'gpu/proofs'/fn,dest/fn)
    for fn in ('f2_negative_summary.json','census.json'):
        if (case/'gpu'/fn).exists(): shutil.copyfile(case/'gpu'/fn,dest/fn)
    shutil.copyfile(case/'e41/wave_gate.json',dest/'E41.json')
    shutil.copyfile(case/'e41/PIN_MAP.json',dest/'E41_PIN_MAP.json')
    shutil.copyfile(case/'input/namelist.input',dest/'namelist.input')
    for fn in ('gpu_cuda.log','e41.log','score.log','writer_audit.log'):
        blob=(case/fn).read_bytes(); (dest/(fn+'.gz')).write_bytes(gzip.compress(blob,mtime=0))
        assert gzip.decompress((dest/(fn+'.gz')).read_bytes())==blob
    for frame in sorted((case/'gpu').glob('wrfout_d01_*')):
        raw.append({'config':n,'kind':'GPU_frame','path':str(frame.resolve()),'bytes':frame.stat().st_size,'sha256':sha(frame)})
    pins=json.loads((case/'e41/PIN_MAP.json').read_text())['pins']
    for pin in pins:
        raw.append({'config':n,'kind':'paid_native_PIN','path':str(Path(pin['pin']).resolve()),'bytes':pin['bytes'],'sha256':pin['sha256'],'seal':'copied+SHAverified by consumer before cache cleanup; no redundant rescan'})
ram=[json.loads(line) for line in (ROOT/'RAM_SAMPLES.jsonl').read_text().splitlines()]
scope_summary=[{'scope':scope,'observed_memory_peak_B':max(int(x['memory.peak']) for x in ram if x['scope']==scope),
    'memory_max_B':sorted({int(x['memory.max']) for x in ram if x['scope']==scope}),
    'max_observed_swap_B':max(int(x['memory.swap.current']) for x in ram if x['scope']==scope),
    'last_observed_memory_events':next(x['memory.events'] for x in reversed(ram) if x['scope']==scope)} for scope in sorted({x['scope'] for x in ram})]
qualification={'schema':'v-run-qhmass-final-gpu24h-v1','utc':datetime.now(timezone.utc).isoformat(),'source_rev':plan['source_rev'],
    'frozen_D6_manifest_sha256':'9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc',
    'scientific_scope':'Original CPU-WRF Swiss 24h; exact matched inputs/namelists, four affected hail configs on immutable627 only.',
    'configs':rows,'GPU_pair_RAM_scopes':scope_summary,'timing_scope':'Cold concurrent qualification, shared CPU/IO; NOT S1 or isolated speed benchmark',
    'release_clearance':'NOT_CLAIMED','model_code_changes_by_lane':0,
    'open_limits':'Opt-in E41 failures and F2 negative visits remain disclosed; direct-writer followup and unrelated configurations are not qualified by this packet.'}
write(ROOT/'QUALIFICATION.json',qualification);write(OUT/'QUALIFICATION.json',qualification);write(OUT/'RAW_ARTIFACTS.json',{'artifacts':raw})
for fn in ('RUN_PLAN.json','PREFLIGHT.json','SOURCE_IDENTITY.json','SOURCE_CLEANUP.json','RAM_SAMPLES.jsonl','SECOND_PAIR_ORDER.json','mp18_mp18_cu5_COMMAND.json','mp18_mp18_cu5_WRAPPER_DONE.json','mp24_mp26_COMMAND.json','mp24_mp26_WRAPPER_DONE.json','CONSUMERS_DONE.json','STATUS.json'):
    if (ROOT/fn).exists(): shutil.copyfile(ROOT/fn,OUT/fn)
(OUT/'tools').mkdir(exist_ok=True)
for fn in (ROOT/'tools').glob('*.py'): shutil.copyfile(fn,OUT/'tools'/fn.name)
lines=['# Hail-mass final-source GPU requalification','',f"Source `{plan['source_rev']}`. Original CPU-WRF Swiss 42×42×44, dt18s, full24h, hourly h0–24; original input and namelist hashes match. Frozen D6 bounds unchanged. No model edits by v-run.",'','| Config | D6 / coverage | E41 | Cold wall (s) | Sampled VRAM (MiB) |','|---|---|---|---:|---:|']
for r in rows:
    lines.append(f"| {r['config']} | {r['D6']} / {'complete' if r['coverage_complete'] else 'INCOMPLETE'} | {r['E41']} | {r['wall_s']:.3f} | {r['sampled_peak_vram_mib']} |")
lines+=['','[M] Each verdict uses every paired hourly frame and all original CPU inventory. See per-config grid_gate.json for first failure, missing fields and report-only values. E41 scans the real paid JIT+AOT artifacts before deleting caches; exact native PINs and GPU files remain external with hashes.','', 'F2 negative phase-field visits remain OPEN. Finite output alone does not clear pre-mask events. Cold concurrent wall is qualification timing, not S1, a warm benchmark, or a speedup claim. No release clearance claimed.','', 'The lock released between pairs; a real intervening v-prep cohort is bound in SECOND_PAIR_ORDER.json. Source snapshots and caches are disposable; original CPU references and previous science are preserved. Logs are lossless gzip, verified against originals.']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
write(OUT/'MANIFEST.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']})
with (MAIN/'EXPERIMENTS.tsv').open('a') as f:
    f.write(f"{qualification['utc']}\tv-run\tGPT-6.1 Sol\tVR-R4-QH-SEAL\tFinal affected hail qualification counters/metadata/provenance\tsealed realGPUframes+nativePINs, no numerical rerun\ttruthful4x25 D6/E41/F2/fullCPUinventory\toldhailPASSprovisional\t[M CPU]{[(r['config'],r['D6'],r['verdict'],r['E41']) for r in rows]}\tKEEP measured evidence\tartifact={OUT}/QUALIFICATION.json;cacheNONE after fullE41;workCPU9JAXcpuQUIETguard streamingSHA,alloriginalrefs preserved\n")
print(json.dumps(qualification,indent=1))
