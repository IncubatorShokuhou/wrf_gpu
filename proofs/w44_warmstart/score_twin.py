"""Invoke unchanged frozen D6 on the identical-input 3h cycling pair."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart')
CPU=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/cpu_native_may04_00_guarded_3r')
SOURCE=BASE/'WS19_snapshot'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    receipt,output=map(Path,sys.argv[1:3]);record=json.loads(receipt.read_text())
    assert record['source_commit']=='0c315f3108eeb08222b79c30468aea24d5ad7669'
    assert record['phases']['cold_gpu']['rc']==0
    cpu=json.loads((CPU/'cpu_run.json').read_text())
    assert cpu['status']=='DONE_CLEAN'and cpu['input_identity']and cpu['complete_times']
    assert cpu['executable_sha256']=='1c1269fb10b7aa135bf42417f1aa7f7a36f7137ea40cd82769316e8ac5bb4918'
    assert cpu['original_gen2_executable_sha256']=='1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b'
    assert sha(CPU/'CPU_OUTPUTS.json')==cpu['output_manifest_sha256']
    for row in json.loads((CPU/'CPU_OUTPUTS.json').read_text()):
        assert sha(CPU/'cpu'/row['file'])==row['sha256'],row['file']
    gpu=receipt.parent/'cold/forecast/wrfout'
    assert len(list(gpu.glob('wrfout_d0*')))==12
    output.mkdir(exist_ok=False,parents=True)
    script=SOURCE/'scripts/wn3_score.py';manifest=SOURCE/'proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json'
    provenance={'gpu_receipt':str(receipt),'gpu_receipt_sha256':sha(receipt),'cpu_result':str(CPU/'cpu_run.json'),
        'cpu_result_sha256':sha(CPU/'cpu_run.json'),'original_gen2_sha256':cpu['original_gen2_executable_sha256'],
        'private_guarded_cpu_sha256':cpu['executable_sha256'],'guard_scope':'start_em allocated-package diagnostic bucket resets only',
        'gpu_source':record['source_commit'],'D6_script_sha256':sha(script),'D6_manifest_sha256':sha(manifest),
        'scope':'3h cycling pair; not the full24/72h release gate; all fields scored, only frozen bounded fields carry tolerances'}
    (output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return subprocess.call([sys.executable,str(script),'--cpu-dir',str(CPU/'cpu'),'--gpu-dir',str(gpu),
                            '--out',str(output/'d6'),'--hours','3','--sequential-domains'])


if __name__=='__main__':raise SystemExit(main())
