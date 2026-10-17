"""One RAM slot; immutable source; fresh, bounded CPU test arms and receipts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

revision=sys.argv[1]
tag=sys.argv[2] if len(sys.argv)>2 else 'TWF05'
repo=Path(__file__).resolve().parents[2]
scratch=Path('<DATA_ROOT>/wrf_gpu2/scratch/t-writer')
receipts=Path('<USER_HOME>/wrf_gpu2_lanes/t-writer')
tree=scratch/(tag+'-tree')
main=Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
python=sys.executable


def quiet():
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit('QUIET: no heavy command started')


def run(arm, command, *, cwd=repo, env=None, cap=600, expected_assertions=0):
    quiet()
    # Main-only previous-work search immediately before this arm.
    search=subprocess.run(['rg','-i',arm+'|WRF-inactive|writer producer',
        *[str(main/f) for f in ('FINDINGS.md','EXPERIMENTS.tsv','INBOX.md')]],
        stdout=subprocess.PIPE,text=True)
    (receipts/(arm+'_search.log')).write_text(search.stdout)
    started=time.monotonic()
    with (receipts/(arm+'.log')).open('w') as log:
        p=subprocess.run(['timeout','--kill-after=10',str(cap),*command],cwd=cwd,
            env=env or os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    record=dict(arm=arm,source=revision,command=command,rc=p.returncode,
        wall_s=time.monotonic()-started,cache='NONE GPUWRF_JAX_CACHE=0',
        caps='one shared RAM slot; MemoryMax6G/Swap0; fresh child',
        cores='12,13,28,29 nice19',artifact=str(receipts/(arm+'.log')))
    xml=receipts/(arm+'.xml')
    if xml.exists():
        import xml.etree.ElementTree as ET
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        record.update({key:sum(int(s.get(key,'0')) for s in suites)
                       for key in ('tests','failures','errors','skipped')})
        if expected_assertions:
            raw=xml.read_text()
            record['genuine_assertion_mutant']=(record['failures']==expected_assertions
                and record['errors']==0 and record['skipped']==0 and 'AssertionError' in raw)
    (receipts/(arm+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    now=datetime.now(timezone.utc).isoformat()
    with (main/'EXPERIMENTS.tsv').open('a') as f:
        f.write('\t'.join((now,'t-writer','GPT Sol',arm,
            'Producer retention/inactive history and affected shared contracts',
            'immutable'+revision,'CPU-only bounded arm','no forecast or GPU claim',
            '[M]'+json.dumps(record,separators=(',',':')),
            'KEEP expected assertion deletion' if record.get('genuine_assertion_mutant') else
            'KEEP evidence' if p.returncode==0 else 'FAIL investigate',
            f'artifact={record["artifact"]};cacheNONE;work=JAXcpu oneRAMslot6GSwap0 timeout{cap}'))+'\n')
    print(arm,'rc',p.returncode,'wall',round(record['wall_s'],2),
          {k:record[k] for k in ('tests','failures','errors','skipped') if k in record},flush=True)
    return p.returncode


quiet()
assert shutil.disk_usage(scratch).free > 120*2**30
old=scratch/{'TWF05':'TW19-tree','TWF06':'TWF05-tree','TWF07':'TWF06-tree','TWF08':'TWF07-tree','TWF09':'TWF08-tree','TWF10':'TWF09-tree'}[tag]
if old.exists():
    assert not subprocess.check_output(['git','-C',str(old),'status','--porcelain'],text=True).strip()
    if run(tag+'-clean-old',['git','worktree','remove',str(old)],cap=300):
        raise SystemExit('old snapshot cleanup failed')
if not tree.exists():
    if run(tag+'-setup',['git','worktree','add','--detach',str(tree),revision],cap=600):
        raise SystemExit('snapshot setup failed; no dependent tests')
link=tree/'data/wrf_pristine'
if not link.exists():
    link.symlink_to('<USER_HOME>/src/wrf_pristine')
assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip().startswith(revision)
assert not subprocess.check_output(['git','-C',str(tree),'status','--porcelain'],text=True).strip()
env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',OMP_NUM_THREADS='1',
         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(tree/'src'))
if not (receipts/'TWF03_fasdas_inactive.json').exists():
    run('TWF03b',[python,str(tree/'proofs/t_writer_fields/inactive_constants.py'),
        '<USER_HOME>/wrf_gpu2_lanes/v-prep',str(receipts/'TWF03_fasdas_inactive.json'),'--fasdas'],env=env,cap=180)
selections=[
    ('TWF05-own',['tests/test_writer_producer_fields.py']),
    ('TWF05-schema',['tests/test_writer_composite_schema.py','tests/test_writer_schema_gate.py']),
    ('TWF05-mp',['tests/test_microphysics_history_inventory.py']),
    ('TWF05-god',['tests/test_goddard_operational.py','tests/test_goddard_history.py']),
    ('TWF05-aero',['tests/test_v016_thompson_aero_threading.py','-k',
        'coldstart or adapter_smoke or wrfrst_restart_threading']),
    ('TWF05-restart',['tests/test_restart_schema_held_physics.py',
                     'tests/test_m7_restart_checkpoint_roundtrip.py', 'tests/test_netcdf_process_lock.py']),
]
if '--retry' in sys.argv:
    selections[0]=(tag+'-own-retry',[
        'tests/test_writer_producer_fields.py::test_authorized_constants_are_original_cpu_bits_and_schema[False-c46]',
        'tests/test_writer_producer_fields.py::test_authorized_constants_are_original_cpu_bits_and_schema[True-c46]',
        'tests/test_writer_producer_fields.py::test_goddard_keeps_actual_input_tensor_after_mp_changes'])
if tag=='TWF07':
    selections[0]=(tag+'-own-retry',[
        'tests/test_writer_producer_fields.py::test_goddard_keeps_actual_input_tensor_after_mp_changes',
        'tests/test_writer_producer_fields.py::test_nonfinite_producer_profile_is_refused'])
if tag=='TWF08':
    import xml.etree.ElementTree as ET
    failures=[c.get('classname').replace('.', '/')+'.py::'+c.get('name')
        for c in ET.parse(receipts/'TWF07-restart.xml').getroot().iter('testcase') if c.find('failure') is not None]
    selections=[
        (tag+'-restart-retry',failures),
        (tag+'-own-aero-restart',[
            'tests/test_writer_producer_fields.py::test_mp28_input_emission_is_held_and_consumed_without_recomputation',
            'tests/test_writer_producer_fields.py::test_mp28_missing_ccn_profile_runs_original_fallback_before_holding_emission',
            'tests/test_writer_producer_fields.py::test_profile_restart_bytes_and_missing_active_refusal']),
        (tag+'-aero',['tests/test_v016_thompson_aero_threading.py','-k',
            'coldstart or adapter_smoke or wrfrst_restart_threading']),
    ]
    shared=['test_v060_physics_dispatch.py','test_cli.py','test_scheme_catalog_fail_closed.py',
        'test_v034_implemented_notes.py','test_namelist_binding.py','test_namelist_check.py',
        'contracts','test_v018_mp_family_fail_closed.py']
    selections.extend((tag+'-shared'+str(i),['tests/'+file]) for i,file in enumerate(shared))
    # Only these scheme callers changed; the other completed smoke branches
    # are byte/source-bound reuse, not repeated numerical qualification.
    selections.extend((tag+'-smoke'+str(i),[node]) for i,node in enumerate([
        'tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[13]',
        'tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[28]',
        'tests/test_v013_operational_smoke.py::test_pbl_operational_runs_and_mixes[12]']))
if tag=='TWF09':
    selections=[(tag+'-changed-artifact',[
        'tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[28]',
        'tests/test_writer_producer_fields.py::test_authorized_constants_are_original_cpu_bits_and_schema[False-c20]'])]
if tag=='TWF10':
    selections=[(tag+'-approved-fasdas',[
        'tests/test_writer_producer_fields.py::test_noahmp_fasdas_writes_only_qualified_inactive_cpu_values',
        'tests/test_writer_producer_fields.py::test_active_branches_never_receive_inactive_constants',
        'tests/test_writer_producer_fields.py::test_sbu_existing_solve_riming_output_is_retained',
        'tests/test_writer_schema_gate.py::test_c53_one_frame_cannot_become_a_24_hour_qualification'])]
for arm,selection in selections:
    arm=arm.replace('TWF05',tag)
    quiet()
    temp=scratch/'pytest'/arm
    temp.parent.mkdir(parents=True,exist_ok=True)
    rc=run(arm,[python,'-m','pytest','-q','-p','no:cacheprovider',
        '--basetemp',str(temp),'--junitxml',str(receipts/(arm+'.xml')),*selection],cwd=tree,env=env)
    # Exact owned path, immediate cleanup; retained XML/log is sufficient.
    quiet()
    if temp.exists():shutil.rmtree(temp)
    if rc:
        print('Stopping after failed arm; no redundant later validation',flush=True)
        raise SystemExit(rc)
if tag=='TWF09':
    # Both official default program identities; this traces the native program
    # on CPU without executing any GPU work. Historical peakRSS 2.9/3.8GiB.
    for case,expected in [('swiss','f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d'),
                          ('prod','6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b')]:
        output=receipts/(tag+'_'+case+'.json')
        args=[python,str(receipts/'run_identity.py'),str(tree),str(output)]
        if case=='prod':args.append('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
        if run(tag+'-identity-'+case,['/usr/bin/time','-v',*args],cwd=tree,env=env,cap=600):
            raise SystemExit('default identity probe failed')
        result=json.loads(output.read_text())
        assert result['arms']['release']['sha']==expected,(case,result['arms'])
    # Three genuine source deletions; setup errors must never count as kills.
    mutant_env=dict(env,PYTHONPATH=os.pathsep.join([str(tree/'src'),str(tree/'proofs/t_writer_fields')]))
    for kind,node in [('rimi','test_sbu_existing_solve_riming_output_is_retained'),
                      ('goddard','test_goddard_keeps_actual_input_tensor_after_mp_changes'),
                      ('ccn','test_authorized_constants_are_original_cpu_bits_and_schema[False-c20]')]:
        arm=tag+'-delete-'+kind
        temp=scratch/'pytest'/arm;temp.parent.mkdir(parents=True,exist_ok=True)
        menv=dict(mutant_env,T_WRITER_MUTANT=kind,T_WRITER_MUTANT_RECEIPT=str(receipts/(arm+'_source.json')))
        run(arm,[python,'-m','pytest','-q','-p','no:cacheprovider','-p','delete_plugin',
            '--basetemp',str(temp),'--junitxml',str(receipts/(arm+'.xml')),
            'tests/test_writer_producer_fields.py::'+node],cwd=tree,env=menv,cap=120,expected_assertions=1)
        quiet()
        if temp.exists():shutil.rmtree(temp)
        assert json.loads((receipts/(arm+'.json')).read_text()).get('genuine_assertion_mutant'),arm
    # Clean owned archive: exercise tracked fixtures without untracked assets.
    import tarfile
    archive=scratch/'TWF09-owned-archive';archive.mkdir(exist_ok=True)
    tarpath=scratch/'TWF09-owned.tar'
    quiet()
    with tarpath.open('wb') as f:
        subprocess.run(['git','archive',revision,'proofs/t_writer_fields',
            'tests/test_writer_producer_fields.py'],cwd=repo,stdout=f,check=True)
    with tarfile.open(tarpath) as tar:tar.extractall(archive,filter='data')
    temp=scratch/'pytest/TWF09-archive';temp.parent.mkdir(parents=True,exist_ok=True)
    aenv=dict(env,PYTHONPATH=os.pathsep.join([str(tree/'src'),str(tree/'tests')]))
    if run(tag+'-archive',[python,'-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(temp),
        '--junitxml',str(receipts/(tag+'-archive.xml')),str(archive/'tests/test_writer_producer_fields.py'),
        '-k','original_cpu_profile_values_survive_writer'],cwd=archive,env=aenv,cap=120):
        raise SystemExit('clean archive assets failed')
    quiet();shutil.rmtree(temp);shutil.rmtree(archive);tarpath.unlink()
    # Reinitialize only selectors affected by this lane. Other actual measured
    # inventories retain their old source/hash; this never claims a new run.
    root=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
    matrix=json.loads((root/'MATRIX.json').read_text())
    fresh={c['id'] for c in matrix['cases'] if c['options'].get('mp_physics') in (13,17,18,19,21,22,28,40,55)
           or c['options'].get('bl_pbl_physics')==12 or c['options'].get('ra_lw_physics')==5
           or c['id']=='c46'}
    update=receipts/'TWF09_updates';combined=receipts/'TWF09_inventories'
    update.mkdir(exist_ok=True);combined.mkdir(exist_ok=True)
    for case in matrix['cases']:
        name=case['id'];target=combined/name;target.mkdir(exist_ok=True)
        if name in fresh:
            dest=update/name;dest.mkdir(exist_ok=True)
            if case['status']!='ADMITTED':
                rec=dict(case=name,status='NOT_GPU_ELIGIBLE',reason=case['status'],source_commit='NOT_INITIALIZED')
            else:
                folder=scratch/'TWF09_init'/name;folder.mkdir(parents=True,exist_ok=True)
                init_env=dict(env,GPUWRF_FAST_DEFAULTS='1',GPUWRF_FULL_WRFOUT_VARIABLES='1',GPUWRF_CENSUS='0')
                run(tag+'-init-'+name,[python,str(tree/'proofs/t_writer/initialized_schema.py'),str(root/name),str(folder)],
                    cwd=tree,env=init_env,cap=300)
                raw=folder/'inventory.json'
                if not raw.exists():raise SystemExit('missing initialized receipt '+name)
                rec=json.loads(raw.read_text());rec['source_commit']=revision
                rec['scope']='Actual final-source zero-step initialization and both real writer phases; no forecast'
                quiet();shutil.rmtree(folder)
            (dest/'inventory.json').write_text(json.dumps(rec,indent=2)+'\n')
        else:
            raw=receipts/'FINAL_inventories'/name/'inventory.json'
            rec=json.loads(raw.read_text())
            import hashlib
            rec['reuse']=dict(original_receipt=str(raw),original_receipt_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                candidate_source_commit=revision,
                scope='Schema-only actual old receipt; this configuration does not select new A/B outputs. All history metadata/header/soil functions unchanged. No new-source initialization or numerical qualification claimed.')
        (target/'inventory.json').write_text(json.dumps(rec,indent=2)+'\n')
    if run(tag+'-compare',[python,str(tree/'proofs/t_writer/compare_schema.py'),str(root),str(combined),
        str(receipts/'TWF09_schema.json'),str(tree)],cwd=tree,env=env,cap=300):
        raise SystemExit('schema comparison did not finish')
if tag=='TWF10':
    # Model/init trees are byte-identical to the final measured default probe;
    # only the newly approved pure-host FASDAS branch changes product source.
    changed=subprocess.check_output(['git','diff','--name-only','29a258f68',revision,
        '--','src/gpuwrf'],cwd=repo,text=True).splitlines()
    assert set(changed)=={'src/gpuwrf/io/producer_history.py','src/gpuwrf/io/wrfout_writer.py'},changed
    menv=dict(env,PYTHONPATH=os.pathsep.join([str(tree/'src'),str(tree/'proofs/t_writer_fields')]),
        T_WRITER_MUTANT='rimi',T_WRITER_MUTANT_RECEIPT=str(receipts/'TWF10-delete-rimi_source.json'))
    temp=scratch/'pytest/TWF10-delete-rimi';temp.parent.mkdir(parents=True,exist_ok=True)
    run('TWF10-delete-rimi',[python,'-m','pytest','-q','-p','no:cacheprovider','-p','delete_plugin',
        '--basetemp',str(temp),'--junitxml',str(receipts/'TWF10-delete-rimi.xml'),
        'tests/test_writer_producer_fields.py::test_sbu_existing_solve_riming_output_is_retained'],
        cwd=tree,env=menv,cap=120,expected_assertions=1)
    quiet()
    if temp.exists():shutil.rmtree(temp)
    assert json.loads((receipts/'TWF10-delete-rimi.json').read_text()).get('genuine_assertion_mutant')
    # The preceding clean archive qualified all original assets. New FASDAS
    # fixtures are already tracked and exercised on this clean immutable tree.
    root=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
    combined=receipts/'TWF10_inventories';combined.mkdir(exist_ok=True)
    for case in json.loads((root/'MATRIX.json').read_text())['cases']:
        name=case['id'];target=combined/name;target.mkdir(exist_ok=True)
        if name in ('c53','c78'):
            folder=scratch/'TWF10_init'/name;folder.mkdir(parents=True,exist_ok=True)
            init_env=dict(env,GPUWRF_FAST_DEFAULTS='1',GPUWRF_FULL_WRFOUT_VARIABLES='1',GPUWRF_CENSUS='0')
            run('TWF10-init-'+name,[python,str(tree/'proofs/t_writer/initialized_schema.py'),str(root/name),str(folder)],
                cwd=tree,env=init_env,cap=300)
            raw=folder/'inventory.json';rec=json.loads(raw.read_text());rec['source_commit']=revision
            rec['scope']='Actual final-source zero-step initialized real serialization; no forecast claim'
            quiet();shutil.rmtree(folder)
        else:
            raw=receipts/'TWF09_inventories'/name/'inventory.json';rec=json.loads(raw.read_text())
            import hashlib
            rec['fasdas_only_reuse']=dict(original_receipt=str(raw),
                original_receipt_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                candidate_source_commit=revision,
                scope='Only Noah-MP4/grid_sfdda2 host constants differ from final29a source. This configuration does not select that branch; its actual source remains recorded.')
        (target/'inventory.json').write_text(json.dumps(rec,indent=2)+'\n')
    if run('TWF10-compare',[python,str(tree/'proofs/t_writer/compare_schema.py'),str(root),str(combined),
        str(receipts/'TWF10_schema.json'),str(tree)],cwd=tree,env=env,cap=300):
        raise SystemExit('final schema comparison did not finish')
print('STAGE DONE',flush=True)
