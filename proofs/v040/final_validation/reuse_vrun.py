"""Score identical-work v-run GPU bytes against our own original references."""
from pathlib import Path
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from prepare import ROOT, sha, write_json
from gpuwrf.io.namelist_parser import parse_namelist

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ident = sys.argv[1]
name = {'c13':'mp18_cu5','c18':'mp26','c19':'mp28','c20':'mp40'}[ident]
case = ROOT/ident
other = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path('<USER_HOME>/wrf_gpu2_lanes/v-run')/name
assert other.is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/v-run'))
source = Path(sys.argv[2]).resolve()
receipt = json.loads((other/'gpu_run.json').read_text())
assert receipt['platform'] == 'gpu' and receipt['rc'] == 0
assert (other/'RESULT.json').exists(), 'Wait for producer consumers and PIN sealing'
assert not (case/'GPU_REUSE.json').exists() and not (case/'gpu_run.json').exists()
def tree(commit, name):
    return subprocess.check_output(['git','rev-parse',commit+':'+name],cwd=source,text=True).strip()
assert tree(receipt['source_rev'],'src') == tree('HEAD','src')
assert tree(receipt['source_rev'],'data') == tree('HEAD','data')
assert {p.name:sha(p) for p in (other/'input').iterdir() if p.is_file()} == receipt['actual_input_sha256']
for name in ('wrfinput_d01','wrfbdy_d01'):
    assert sha(case/'input'/name) == receipt['actual_input_sha256'][name]
a,b = (parse_namelist(p/'input/namelist.input') for p in (case,other))
differences=[(g,k,a.get(g,{}).get(k),b.get(g,{}).get(k)) for g in set(a)|set(b)
    for k in set(a.get(g,{}))|set(b.get(g,{})) if a.get(g,{}).get(k)!=b.get(g,{}).get(k)]
assert not differences or differences == [('physics','use_mp_re',1,None)], differences
# Both the actual pipeline binder and mp_re_config default omitted use_mp_re to1.
assert '_domain_physics_int(run, "use_mp_re", name, 1)' in (source/'src/gpuwrf/integration/nested_pipeline.py').read_text()
from gpuwrf.physics import rrtmg_mp_re as radii
original = radii._MP_RE
active=[]
assert receipt['measurement_flags']['GPUWRF_CENSUS'] == '1'
assert receipt['measurement_flags']['GPUWRF_FAST_DEFAULTS'] in (None,'1')
assert receipt['argv'][receipt['argv'].index('--hours')+1] == '24'
for flag in (receipt['measurement_flags']['GPUWRF_RRTMG_MP_RE'], '1'):
    radii._MP_RE = flag == '1'
    active.append(radii.mp_re_active(a))
radii._MP_RE = original
assert active[0] == active[1], 'Different active radiation program'
target=case/'gpu';assert not target.exists();target.symlink_to(other/'gpu',target_is_directory=True)
scorer = Path(__file__).with_name('score_with_handles.py')
if not scorer.exists():scorer = Path(__file__).with_name('score.py')
run=subprocess.run(['timeout','1800','python',str(scorer),
    str(source),str(case),'--domains','d01'],stdout=(case/'score_reuse.log').open('w'),stderr=subprocess.STDOUT)
assert (case/'D6_d01.json').exists(),run.returncode
gate=json.loads((case/'D6_d01.json').read_text())
pins=other/'e41/PIN_MAP.json';e41=other/'e41/wave_gate.json'
assert pins.exists() and e41.exists()
result=dict(schema='v040-gpu-arm-reuse-v1',case=ident,producer_case=str(other),
    producer_receipt=str(other/'gpu_run.json'),producer_receipt_sha256=sha(other/'gpu_run.json'),
    model_src_tree=tree('HEAD','src'),model_data_tree=tree('HEAD','data'),
    namelist_differences=differences,mp_re_active_both=active,
    gpu_files={p.name:sha(p) for p in sorted(target.glob('wrfout*'))},
    d6=gate['verdict'],coverage_complete=gate['coverage_complete'],
    e41=json.loads(e41.read_text())['e41_gate'],external_e41=str(e41),external_e41_sha256=sha(e41),
    external_pins=str(pins),external_pins_sha256=sha(pins),
    original_cpu_receipt_sha256=sha(case/'cpu_run.json'),
    original_cpu_manifest_sha256=sha(case/'CPU_OUTPUTS.json'),
    matched_inputs_sha256=sha(case/'inputs.json'),
    producer_result_sha256=sha(other/'RESULT.json'),
    f2_disposition='Producer census retained; D6 PASS does not close negative-event/F2 questions',
    scope='No GPU rerun; actual producer GPU data rescored against this matched original CPU-WRF reference; no new native compiler realization inferred',
    utc=datetime.now(timezone.utc).isoformat())
write_json(case/'GPU_REUSE.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('gpu_files',)},indent=2))
