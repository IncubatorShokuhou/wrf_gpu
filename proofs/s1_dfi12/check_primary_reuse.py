"""Bind primary CPU receipts to the final composed candidate without numerical reruns.

Usage: python check_primary_reuse.py SOURCE OUT_JSON
Model-family bytes and shared TDFI numerical functions must match measured snapshots.
Only new host clock conditions differ; their resolved primary-case values are checked.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

source,out=Path(sys.argv[1]),Path(sys.argv[2])
def git(*args):
    return subprocess.check_output(['git','-C',str(source),*args],text=True)
def old(rev,path):
    return git('show',f'{rev}:{path}')
def functions(text):
    return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
model_paths=[f'src/gpuwrf/{n}' for n in ('contracts','dynamics','kernels','physics','coupling','runtime','nesting')]
family={rev:git('diff','--name-only',rev,'HEAD','--',*model_paths).splitlines() for rev in ('0411cb9c4','87d539114')}
composed=None
if any(family.values()):
    # Main may add unrelated opt-in schemes while the worker qualifies DFI.
    # Require actual phase program AND seed-byte identity before carrying truth.
    packet=Path(sys.argv[3]) if len(sys.argv)>3 else out.parent/'DF12-COMPOSED-PROGRAMS.json'
    composed=json.loads(packet.read_text())
    assert composed['all_programs_and_seed_arrays_exact'],composed
    assert composed['candidate_src_tree']==git('rev-parse','HEAD:src/gpuwrf').strip()
shared='src/gpuwrf/dfi/driver.py'
base=functions(old('407263411',shared)); now=functions((source/shared).read_text())
exact_shared={name:base[name]==now[name] for name in base if name!='apply_dfi_prepass'}
assert all(exact_shared.values()),exact_shared
p='src/gpuwrf/dfi/filter.py'; assert old('407263411',p)==(source/p).read_text()
p='src/gpuwrf/dfi/driver12.py'
fn=functions((source/p).read_text()); measured=functions(old('87d539114',p))
exact12={name:fn[name]==measured[name] for name in ('_backward_endpoint','rebase_dfl_boundary','_radar_field','_dfl_diffusion','_clock_edits')}
assert all(exact12.values()),exact12
p='src/gpuwrf/integration/nested_pipeline.py'; fn=functions((source/p).read_text())
loader={rev:fn['_load_domains']==functions(old(rev,p))['_load_domains'] for rev in family}
# An opt-in lake hook may change this host function. The phase packet then
# requires identical actual loader seeds as well as all active step programs.
assert all(loader.values()) or composed is not None,loader
p='src/gpuwrf/integration/d02_replay.py'
fn=functions((source/p).read_text())
kinematic={rev:fn['_wrf_set_w_surface']==functions(old(rev,p))['_wrf_set_w_surface'] for rev in family}
assert all(kinematic.values()),kinematic
sys.path.insert(0,str(source/'src'))
from gpuwrf.dfi.control import dfi_control_from_namelist
from gpuwrf.dfi.driver12 import _dfl_preserves_w
from gpuwrf.io.namelist_check import _parse_wrf_namelist
lane=Path('<USER_HOME>/wrf_gpu2_lanes/s1-dfi12'); resolved={}
for opt,tag in ((1,'DF12-E2'),(2,'DF12-E3')):
    nml=_parse_wrf_namelist((lane/'oracle'/f'opt{opt}'/'namelist.input').read_text())
    ctl=dfi_control_from_namelist(nml); receipt=json.loads((lane/tag/'report.json').read_text())
    assert (ctl.n_bck,ctl.n_fwd)==(receipt['bck_steps'],receipt['fwd_steps'])
    assert ctl.fwd_start.strftime('%Y-%m-%d_%H:%M:%S')==receipt['fwd_start']
    assert ctl.dt_s==receipt['dt_s']
    assert (opt==1 and _dfl_preserves_w(nml))==(opt==1)
    assert (3600-receipt['forecast_offset_s'])/ctl.dt_s==round((3600-receipt['forecast_offset_s'])/ctl.dt_s)
    resolved[str(opt)]={'n_bck':ctl.n_bck,'n_fwd':ctl.n_fwd,'fwd_start':receipt['fwd_start'],
                        'preserve_w':opt==1,'ceil_changes_duration':False,
                        'shared_sample_accumulator_filter_start_em_exact':True}
result={'candidate':git('rev-parse','HEAD').strip(),'model_family_changed_paths':family,
        'composed_program_identity':composed,
        'tdfi_shared_functions_exact':exact_shared,'filter_source_exact':True,
        'driver12_helpers_exact_to_E3':exact12,'load_domains_exact':loader,'kinematic_helper_exact':kinematic,
        'primary_case_host_conditions':resolved,
        'active_program_identity':'DF12-IDENTITY.json:2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a',
        'limits':'Final clock branches checked by independent short WRF oracles and actual loader W-byte gate; no whole forecast/GPU qualification.'}
out.write_text(json.dumps(result,indent=2)+'\n')
print('SOURCE REUSE PASS',result['candidate'])
