"""Append a source-bound successor disclosure without mutating the old packet."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
base = root/'proofs/t_e41opt'
out = Path(__file__).resolve().parent
case = Path('<USER_HOME>/wrf_gpu2_lanes/t-c03rain/CR29_c03')
mapped = Path('<USER_HOME>/wrf_gpu2_lanes/t-e41opt/EO25_CR29_c03/source_map.json')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

inventory = json.loads((base/'INVENTORY.json').read_text())
sources = json.loads((base/'SOURCE_MAPS.json').read_text())
gate_path = case/'e41/wave_gate.json'
gate = json.loads(gate_path.read_text())
run = json.loads((case/'gpu_run.json').read_text())
inputs = json.loads((case/'inputs.json').read_text())
d6 = json.loads((case/'D6_d01.json').read_text())
mapping = json.loads(mapped.read_text())
assert gate['e41_gate'] == 'FAIL'
assert run['platform'] == 'gpu' and run['rc'] == 0
assert d6['verdict'] == 'PASS' and d6['coverage_complete'] and d6['paired_frames'] == 25
assert not d6['failures'] and not d6['nonfinite']
jit = next(e for e in gate['entries'] if e['source_kind'] == 'jax_cache')
assert jit['blob_sha256'] == mapping['blob_sha256']
flags = {}
scans = []
for entry in gate['entries']:
    label = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
    for index, elf in enumerate(entry['elfs']):
        assert not elf['nonleaf_over1k'] and not elf['callgraph_unknown']
        scan = case/f'e41/cubin_scan_{label}_e{index}.json'
        scans.append({'path':str(scan),'sha256':sha(scan)})
        for name, size in elf['stack_over256']:
            flags.setdefault(name.replace('.','_'),{})['stack_B'] = size
    for name, size in entry['kinput_transpose_over500']:
        flags.setdefault(name.replace('.','_'),{})['kinput_HLO'] = size
assert len(flags) == 6 and max(f['stack_B'] for f in flags.values()) == 2456
assert set(flags) == set(mapping['flags'])
for name, flag in flags.items():
    source = mapping['flags'][name]
    assert source['stack_B'] == flag['stack_B'] and source['sources']
    if 'kinput_HLO' in flag:
        assert source['kinput_HLO'] == flag['kinput_HLO'] == 917
        assert flag['stack_B'] == 392
    flag['sources'] = source['sources'][:8]
    flag['instructions'] = source['instructions']
row = {'producer':'t-c03rain','case':'c03','producer_case':'CR29_c03',
       'options':inputs['options'],'source_commit':run['source_commit'],
       'gate_path':str(gate_path),'gate_sha256':sha(gate_path),
       'inputs_path':str(case/'inputs.json'),'inputs_sha256':sha(case/'inputs.json'),
       'flags':flags,'attribution_path':str(mapped),'program_max_stack_B':2456,
       'owner':'t-e41opt E41, manager transfer; numerical callers retain their owners',
       'D6_verdict':'PASS','D6_path':str(case/'D6_d01.json'),'D6_paired_frames':25,
       'D6_sha256':sha(case/'D6_d01.json'),
       'gpu_run_path':str(case/'gpu_run.json'),'gpu_run_sha256':sha(case/'gpu_run.json'),
       'native_resource_scans':scans,'runtime_impact':'UNKNOWN; no exact-artifact per-kernel profile',
       'whole_run_wall_s':run['monotonic_wall_s'],'sampled_process_vram_MiB':run['sampled_process_vram_mib_peak'],
       'timing_scope':run['timing_scope']}
old = next(r for r in inventory['records'] if r['case'] == 'c03')
assert old['producer'] == 'v-prep' and len(old['flags']) == 5
inventory['records'] = [row if r['case']=='c03' else r for r in inventory['records']]
inventory['cutoff_utc'] = datetime.now(timezone.utc).isoformat()
inventory['historical_c03'] = old
inventory['scope'] = '12 active configurations / 38 flags; final C03 replaces old active row; old evidence preserved'
(out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2,allow_nan=False)+'\n')
packed = {k:mapping[k] for k in ('pin','pin_sha256','blob_sha256','hlo_sha256')}
packed.update(case='c03',producer='t-c03rain',map_path=str(mapped),map_sha256=sha(mapped),
              flags={n:{k:v for k,v in f.items() if k not in ('call','stacks')} for n,f in mapping['flags'].items()})
for flag in packed['flags'].values():flag['sources']=flag['sources'][:10]
sources = [packed if r['case']=='c03' else r for r in sources]
(out/'SOURCE_MAPS.json').write_text(json.dumps(sources,indent=2,allow_nan=False)+'\n')
doc = root/'docs/release/E41_OPT_IN_v0.4.md'
renderer = base/'render_inventory.py'
sys.argv = [str(renderer),str(out/'INVENTORY.json'),str(doc)]
exec(compile(renderer.read_text(),str(renderer),'exec'),{'__file__':str(renderer),'__name__':'__main__'})
text = doc.read_text().replace('The compact evidence files are in `proofs/t_e41opt/`.',
    'The active evidence files are in `proofs/t_e41opt_updates/c03_final/`; the original packet remains in `proofs/t_e41opt/`.')
text = text.replace('C03 Eta/mixed-surface remedies belong to `t-c03rain`; SF1/PBL work belongs\nto `t-c02nan`. Their files were not edited.',
    'C03 E41 disclosure/remedies transferred to `t-e41opt` by manager instruction;\nSF1/PBL numerical files retain their owners, including `t-c02nan`. Their files were not edited.')
text += ('\n## Final C03 successor\n\n'
    'The active C03 row is the final `1612acc49` / `CR29_c03` GPU arm: original CPU-WRF\n'
    'D6 PASS over all 25 frames, complete coverage, zero bounded failures and raw\n'
    'nonfinite values. Its six STACK flags (max 2,456 B) and Lin kInput 917 HLO\n'
    'remain FAIL. The new Lin fusion uses 392 B; its source stack includes\n'
    '`microphysics_lin.lin_run`, `_lin_update._conservation_and_update`,\n'
    '`microphysics_lin._process_block` and `_lin_graupel._graupel_block`.\n\n'
    'The earlier v-prep C03 five-flag / D6 FAIL arm is historical evidence, preserved\n'
    'unchanged in the original packet at commit `8c4ede6fc`, and under `historical_c03`\n'
    'in the successor inventory. It is not relabelled as passing. The active inventory\n'
    'has 12 configurations and 38 flags. No Lin/GSFC firewall is shipped.\n\n'
    '**Source-family correction:** the five older stack fusions are dominated by\n'
    'GSFC shortwave `ra_sw_gsfc._deledd` / `_sagpol`, not Eta surface code. The\n'
    'exact rich-HLO body census attributes 1,979/2,095; 1,842/1,933; 964/1,103;\n'
    '964/1,103; and 1,652/1,723 instructions to `ra_sw_gsfc.py`. These counts\n'
    'are static source attribution, not runtime weights. `SOURCE_FAMILIES.json`\n'
    'binds the complete body census to the retained blob and rich-HLO SHA.\n\n'
    'The separate MP24/26 output-firewall branch `lane/t-e41opt-firewall` (`c4c9ddb65`)\n'
    'has 322/322 CPU caller results BYTE and exact Swiss/PROD default identities,\n'
    'but remains NOT ADMITTED pending captured-target native STACK clearance and\n'
    'the next actual MP24/MP26 GPU arms. It is not part of this disclosure update.\n')
doc.write_text(text)
seal = {'verdict':'DISCLOSURE_UPDATE_READY','parent_packet_commit':'8c4ede6fc',
        'parent_qualification_sha256':sha(base/'QUALIFICATION.json'),
        'active_configs':12,'active_native_flags':38,'C03_science':'original CPU-WRF D6 PASS 25 frames, complete coverage',
        'C03_E41':'FAIL retained: 6 STACK flags / max2456B / Lin392B+917HLO',
        'production_code_changed':False,'new_GPU_runs':0,'native_runtime_impact':'UNKNOWN',
        'historical_packet':'unchanged; use original commit for its frozen document',
        'files':{str(p.relative_to(root)):sha(p) for p in (out/'INVENTORY.json',out/'SOURCE_MAPS.json',out/'SOURCE_FAMILIES.json',Path(__file__),doc)}}
(out/'QUALIFICATION.json').write_text(json.dumps(seal,indent=2)+'\n')
print('PASS: final C03 6 flags / D6 PASS; active12 configs38 flags; old packet unchanged')
