"""Add completed MP18 selector arms; preserve all earlier frozen decisions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
prior=root/'proofs/t_e41opt_updates/c03_final'
inventory=json.loads((prior/'INVENTORY.json').read_text())
sources=json.loads((prior/'SOURCE_MAPS.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for number,name in ((31,'mp18'),(32,'mp18_cu5')):
    case=Path('<USER_HOME>/wrf_gpu2_lanes/v-run')/name
    mapped=Path(f'<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO{number}_{name}/source_map.json')
    mapping=json.loads(mapped.read_text())
    gate_path=case/'e41/wave_gate.json';gate=json.loads(gate_path.read_text())
    run=json.loads((case/'gpu_run.json').read_text());d6=json.loads((case/'grid_gate.json').read_text())
    summary=json.loads((case/'RESULT.json').read_text())
    assert gate['e41_gate']=='FAIL' and run['rc']==0
    assert d6['verdict']==summary['verdict']=='INCOMPLETE' and summary['bounded_D6']=='PASS' and d6['paired_frames']==25
    assert not d6['failures'] and d6['first_failure'] is None
    assert set(summary['missing_CPU_fields'])=={'REFD_MAX','REFL_10CM'} and not d6['raw_gpu_nonfinite']
    jit=next(e for e in gate['entries'] if e['source_kind']=='jax_cache')
    assert mapping['blob_sha256']==jit['blob_sha256']
    flags={};scans=[];resources={}
    for e in gate['entries']:
        for elf in e['elfs']:
            assert not elf['nonleaf_over1k'] and not elf['callgraph_unknown']
            for n,v in elf['stack_over256']:flags.setdefault(n.replace('.','_'),{})['stack_B']=v
        for n,v in e['kinput_transpose_over500']:flags.setdefault(n.replace('.','_'),{})['kinput_HLO']=v
    for e in gate['entries']:
        label=hashlib.sha256(e['path'].encode()).hexdigest()[:12]
        for i,_ in enumerate(e['elfs']):
            path=case/f'e41/cubin_scan_{label}_e{i}.json';scans.append({'path':str(path),'sha256':sha(path)})
            for r in json.loads(path.read_text()):
                if r['kernel'] in flags:
                    assert resources.setdefault(r['kernel'],r['stack'])==r['stack']
    assert set(flags)==set(resources)==set(mapping['flags'])
    for n,f in flags.items():
        f['stack_B']=resources[n];f['sources']=mapping['flags'][n]['sources'][:8]
        f['instructions']=mapping['flags'][n]['instructions']
        if 'kinput_HLO' in f:assert f['kinput_HLO']==mapping['flags'][n]['kinput_HLO']
    assert len(flags)==(3 if name=='mp18' else 2)
    namelist=(case/'input/namelist.input').read_text();options={}
    for key in ('mp_physics','cu_physics','bl_pbl_physics','sf_sfclay_physics','sf_surface_physics','ra_lw_physics','ra_sw_physics'):
        options[key]=int(re.search(r'(?im)^\s*'+key+r'\s*=\s*(\d+)',namelist)[1])
    inputs_path=case/'inputs.json';input_meta=json.loads(inputs_path.read_text())
    inventory['records'].append({'producer':'v-run','case':name,'options':options,
        'source_commit':run.get('source_commit',run.get('source_rev',input_meta['source_rev'])),
        'gate_path':str(gate_path),'gate_sha256':sha(gate_path),
        'inputs_path':str(inputs_path),'inputs_sha256':sha(inputs_path),'flags':flags,
        'attribution_path':str(mapped),'program_max_stack_B':max(x['max_stack_B'] for e in gate['entries'] for x in e['elfs']),
        'owner':'t-e41opt disclosure','D6_verdict':'INCOMPLETE','D6_path':str(case/'grid_gate.json'),
        'D6_paired_frames':25,'D6_sha256':sha(case/'grid_gate.json'),
        'producer_summary_path':str(case/'RESULT.json'),'producer_summary_sha256':sha(case/'RESULT.json'),
        'bounded_D6':'PASS','missing_CPU_fields':['REFD_MAX','REFL_10CM'],
        'gpu_run_path':str(case/'gpu_run.json'),'gpu_run_sha256':sha(case/'gpu_run.json'),
        'native_resource_scans':scans,'runtime_impact':'UNKNOWN; no exact-artifact per-kernel profile',
        'whole_run_wall_s':run.get('monotonic_wall_s',run.get('wall_s')),
        'timing_scope':'completed before GPU STOP/hibernation; concurrent cold qualification, not S1'})
    packed={k:mapping[k] for k in ('pin','pin_sha256','blob_sha256','hlo_sha256')}
    packed.update(case=name,producer='v-run',map_path=str(mapped),map_sha256=sha(mapped),
        flags={n:{k:v for k,v in f.items() if k not in ('call','stacks')} for n,f in mapping['flags'].items()})
    for flag in packed['flags'].values():flag['sources']=flag['sources'][:10]
    sources.append(packed)
assert len(inventory['records'])==14 and sum(len(r['flags']) for r in inventory['records'])==43
inventory['cutoff_utc']=datetime.now(timezone.utc).isoformat()
inventory['scope']='14 selector/configuration receipts / 43 active flags; C03 final retained; MP18 radar coverage incomplete'
(out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2,allow_nan=False)+'\n')
(out/'SOURCE_MAPS.json').write_text(json.dumps(sources,indent=2,allow_nan=False)+'\n')
doc=root/'docs/release/E41_OPT_IN_v0.4.md'
previous=subprocess.check_output(['git','show','56c9f789c:docs/release/E41_OPT_IN_v0.4.md'],text=True)
tail=previous[previous.index('## Final C03 successor'):]
tail=tail.replace('The active inventory\nhas 12 configurations and 38 flags.',
    'At that update, the inventory\nhad 12 configurations and 38 flags; it now has 14 selector/configuration receipts and 43 flags.')
renderer=root/'proofs/t_e41opt/render_inventory.py'
sys.argv=[str(renderer),str(out/'INVENTORY.json'),str(doc)]
exec(compile(renderer.read_text(),str(renderer),'exec'),{'__file__':str(renderer),'__name__':'__main__'})
text=doc.read_text().replace('The compact evidence files are in `proofs/t_e41opt/`.',
    'Active inventory/maps are in `proofs/t_e41opt_updates/mp18/`; original packets remain unchanged in `proofs/t_e41opt/` and `c03_final/`.')
text=text.replace('C03 Eta/mixed-surface remedies belong to `t-c03rain`; SF1/PBL work belongs\nto `t-c02nan`. Their files were not edited.',
    'C03 E41 disclosure/remedies belong to `t-e41opt` by manager transfer.\nSF1/PBL numerical files retain their owners; they were not edited.')
text+='\n'+tail+'\n## MP18 successor\n\n'
text+=('Both completed MP18 arms retain 25-frame bounded D6 PASS and zero raw\n'
    'nonfinite values, but full coverage is INCOMPLETE: `REFD_MAX` and `REFL_10CM`\n'
    'are missing. No diagnostic is zero-filled and no F2 event justification is inferred.\n'
    'MP18 has STACK 504 B plus kInput 665/728 HLO. MP18+CU5 has the two 665/728 HLO\n'
    'flags, with no STACK-over-256 failure. Resource properties and source maps were\n'
    'read from each exact executable; the 504 B frame is not transferred by name\n'
    'to the CU5 program. Both GPU processes completed before the 15:07Z STOP.\n'
    'These are distinct selector/source receipts; legacy MP17 (C12) normalizes to MP18.\n')
text=text.replace('`SOURCE_FAMILIES.json`','`proofs/t_e41opt_updates/c03_final/SOURCE_FAMILIES.json`')
doc.write_text(text)
seal={'verdict':'DISCLOSURE_UPDATE_READY','parent_packet_commit':'56c9f789c',
    'parent_qualification_sha256':sha(prior/'QUALIFICATION.json'),'active_receipts':14,'active_native_flags':43,
    'new_science':'MP18/MP18+CU5 boundedD6PASS25/NF0; fullcoverageINCOMPLETE radar2',
    'E41':'all43failures retained, no waiver','production_code_changed':False,'new_GPU_runs':0,
    'files':{str(p.relative_to(root)):sha(p) for p in (out/'INVENTORY.json',out/'SOURCE_MAPS.json',Path(__file__),doc)}}
(out/'QUALIFICATION.json').write_text(json.dumps(seal,indent=2)+'\n')
print('PASS:14selector receipts43flags; MP18 STACK failure not transferred to CU5; old packets unchanged')
