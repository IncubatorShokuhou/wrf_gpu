"""CPU E41/D6/PIN consumers for completed arms while subsequent GPU batches run."""
from datetime import datetime,timezone
import json,os,re,shlex,signal,subprocess,sys,time,shutil
from pathlib import Path
from prepare import ROOT,write_json,sha
from cpu_references import append,MAIN
from gpu_authorization import CLOSE
assert os.environ.get('JAX_PLATFORMS')=='cpu'
source=Path(sys.argv[1]).resolve();helper=Path(__file__).parent
deadline=CLOSE.timestamp()+3600  # Allow CPU cleanup through Oct15 20:15Z.
done={}  # A retry gets a new launch receipt; never suppress it by case ID alone.
def reference_view(case, row, launch_sha):
    # Manager22:51: preserve raw CPU references, score P3 against the separately
    # completed compiler-defined ZERO original. Never write a foreign namelist.
    if row['id'] not in ('c21','c22','c23','c24'):return case
    zero=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP32')/f"{row['options']['mp_physics']}a"
    receipt=json.loads((zero/'receipt.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['frames']==25 and not receipt['nonfinite']
    assert receipt['source_executable_sha256']=='9a67a93dd891115c095f64a95ce8f50e7ff0942ef1abf4185faf20bd7065742a'
    assert receipt['input_sha256']==row['input_sha256']
    assert sha(zero/'outputs.json')==receipt['output_manifest_sha256']
    view=Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms')/row['id']/'reference-score'/launch_sha
    if not view.exists():
        (view/'input').mkdir(parents=True)
        (view/'input/namelist.input').write_bytes((case/'input/namelist.input').read_bytes())
        assert not (view/'input/namelist.input').is_symlink()
        assert sha(view/'input/namelist.input')==receipt['input_sha256']['namelist.input']
        (view/'cpu_run.json').write_bytes((zero/'receipt.json').read_bytes())
        (view/'CPU_OUTPUTS.json').write_bytes((zero/'outputs.json').read_bytes())
        (view/'cpu').symlink_to(zero,target_is_directory=True)
        (view/'gpu').symlink_to((case/'gpu').resolve(),target_is_directory=True)
    assert not (view/'input/namelist.input').is_symlink()
    assert sha(view/'input/namelist.input')==receipt['input_sha256']['namelist.input']
    assert (view/'cpu').resolve()==zero.resolve()
    assert (view/'gpu').resolve()==(case/'gpu').resolve()
    assert sha(view/'cpu_run.json')==sha(zero/'receipt.json')
    assert sha(view/'CPU_OUTPUTS.json')==receipt['output_manifest_sha256']
    write_json(case/'REFERENCE_OF_RECORD.json',dict(authority='Manager22:51 MYNN rule',
        reference=str(zero),receipt_sha256=sha(zero/'receipt.json'),
        output_manifest_sha256=receipt['output_manifest_sha256'],input_sha256=receipt['input_sha256'],
        preserved_raw_reference=str(case/'cpu'),scoring_view=str(view),
        original_vs_zero_spread=str(Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP35_spread')/row['id']),
        scope='ZERO original WRF reference; raw original kept; frozen D6 bounds unchanged'))
    return view
def run(args,log,limit):
    while Path('/tmp/wrf_gpu2_quiet').exists():time.sleep(1)
    cmd='if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='+shlex.quote(str(source/'src'))+' taskset -c 16 nice -n 19 timeout '+str(limit)+' '+shlex.join(args)
    with log.open('w') as out:
        # Manager20:50: every heavy CPU consumer must take a GLOBAL slot;
        # a private 6G scope alone cannot enforce the aggregate nightly cap.
        p=subprocess.Popen(['<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh',
            '--label','v-prep','--timeout','1200','--','bash','-c',cmd],
            cwd=source,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        paused=False
        while p.poll() is None:
            quiet=Path('/tmp/wrf_gpu2_quiet').exists()
            if quiet!=paused:os.killpg(p.pid,signal.SIGSTOP if quiet else signal.SIGCONT);paused=quiet
            time.sleep(.3)
    return p.returncode
while time.time()<deadline:
    matrix=json.loads((ROOT/'MATRIX.json').read_text())
    for row in matrix['cases']:
        ident=row['id'];case=ROOT/ident
        launch=case/'gpu_launch.json'
        if not launch.exists():continue
        launch_sha=sha(launch)
        if done.get(ident)==launch_sha:continue
        domains=row['domains']
        finalized=case/'QUALIFICATION.json'
        cleared_failure=(finalized.exists() and not (case/'cache').exists() and
            json.loads(finalized.read_text()).get('cache_remaining') is False)
        if all((case/f'D6_{d}.json').exists() for d in domains) and (
            list(case.glob('e41*/PIN_MAP.json')) or cleared_failure):
            done[ident]=launch_sha;continue
        proof=case/'gpu_proof/nested_pipeline_run.json'
        gate=case/'e41'
        if not (gate/'wave_gate.json').exists():
            args=['python',str(source/'scripts/wave_cubin_gate.py'),str(case/'cache/jax'),str(gate)]
            if proof.exists():args+=['--run-proof',str(proof)]
            run(args,case/'e41_consumer.log',900)
        for domain in domains:
            target=case/f'D6_{domain}.json'
            if not target.exists():
                view=reference_view(case,row,launch_sha)
                scorer=helper/'score_with_handles.py'
                if not scorer.exists():scorer=helper/'score.py'
                rc=run(['python',str(scorer),str(source),str(view),'--domains',domain],case/f'score_consumer_{domain}.log',1800)
                if view!=case:
                    for name in (f'D6_{domain}.json',f'compare_{domain}.json'):
                        if (view/name).exists():
                            temporary=case/(name+'.v-prep-tmp')
                            shutil.copyfile(view/name,temporary);temporary.replace(case/name)
                if not target.exists():write_json(target,dict(verdict='INCOMPLETE',coverage_complete=False,score_rc=rc,
                    reason='Scorer could not complete; no synthetic fields/frames or numerical PASS',log=str(case/f'score_consumer_{domain}.log')))
        if (gate/'wave_gate.json').exists():run(['python',str(helper/'pin_cleanup.py'),ident],case/'cleanup_consumer.log',180)
        receipt=dict(case=ident,utc=datetime.now(timezone.utc).isoformat(),final_run_proof_available=proof.exists(),
            d6={d:json.loads((case/f'D6_{d}.json').read_text())['verdict'] for d in domains},
            e41=None if not (gate/'wave_gate.json').exists() else json.loads((gate/'wave_gate.json').read_text())['e41_gate'],
            cache_remaining=(case/'cache').exists(),gpu_process=json.loads((case/'gpu_launch.json').read_text()))
        write_json(case/'QUALIFICATION.json',receipt)
        append(MAIN/'EXPERIMENTS.tsv','\t'.join((receipt['utc'],'v-prep','GPT Sol','VP-D6-E41-'+ident,
            'Full original-WRF inventory/D6 plus actual compiled E41/PIN audit',json.dumps(receipt['d6'],separators=(',',':'))+' E41='+str(receipt['e41']),
            f'artifact={case}/QUALIFICATION.json,{case}/D6_*.json,{gate}/wave_gate.json;cache={case}/cache remaining={receipt["cache_remaining"]};workCPU16NI19JAXcpuGPUWRF0OMP1timeout900/1800/180;frozenD6SHAunchanged')))
        print(receipt['utc'],ident,receipt['d6'],receipt['e41'],flush=True)
        done[ident]=launch_sha
    time.sleep(3)
