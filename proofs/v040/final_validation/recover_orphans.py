"""Audit retained original arms after a supervisor error; never invent exit codes."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time
from cpu_references import inventory,append,MAIN,QUIET
from prepare import ROOT,sha,write_json

assert os.environ.get('JAX_PLATFORMS')=='cpu'
jobs=[dict(id='c62',group=1998675,pid=1998681,core=0),
      dict(id='c64',group=1998677,pid=1998683,core=2),dict(id='c63',group=None,pid=None,core=1)]
started='2026-10-14T10:45:36.500773+00:00'
matrix=json.loads((ROOT/'MATRIX.json').read_text())
recovery_start=datetime.now(timezone.utc).isoformat()
while jobs:
    for job in list(jobs):
        proc=None if job['pid'] is None else Path('/proc')/str(job['pid'])
        alive=proc is not None and proc.exists() and (proc/'stat').read_text().split(') ',1)[1][0]!='Z'
        if alive:
            assert (proc/'cwd').resolve()==ROOT/job['id']/'cpu'
            os.killpg(job['group'],signal.SIGSTOP if QUIET.exists() else signal.SIGCONT)
            continue
        if QUIET.exists():continue
        case=ROOT/job['id'];row=next(r for r in matrix['cases'] if r['id']==job['id'])
        frames,nonfinite,complete=inventory(case,row['domains'],row['hours'])
        text=(case/'cpu.log').read_text(errors='replace')
        marker='SUCCESS COMPLETE WRF' in text
        write_json(case/'CPU_OUTPUTS.json',frames)
        now=datetime.now(timezone.utc)
        receipt=dict(schema='v040-original-cpu-reference-v1',case=job['id'],
            started_utc=started,finished_utc=now.isoformat(),
            monotonic_wall_s=(now-datetime.fromisoformat(started)).total_seconds(),
            timing_scope='UTC-derived recovery wall includes supervisor pause/audit; NOT measured process wall or performance benchmark',
            rc=None,exit_code_scope='UNKNOWN: original parent exited during corrupt-NetCDF inventory; never imputed0',
            success_marker=marker,status='COMPLETE' if marker and complete else 'FAILED',
            completion_basis='Original WRF SUCCESS + all25actualframes/domain + everyrawfieldfinite; processRC unavailable',
            core=job['core'],nice=19,ranks=1,timeout_s=5400,
            source_commit=matrix['source_commit'],launcher_sha256=sha(__file__),
            input_sha256=row['input_sha256'],
            actual_input_sha256={p.name:sha(p) for p in (case/'input').iterdir() if p.is_file()},
            executable=json.loads((case/'CPU_executable.json').read_text()),
            nonfinite=nonfinite,inventory_errors=[r for r in frames if 'inventory_error' in r],
            frame_count=len(frames),output_manifest=str(case/'CPU_OUTPUTS.json'),
            output_manifest_sha256=sha(case/'CPU_OUTPUTS.json'),recovery_monitor_started_utc=recovery_start)
        assert receipt['input_sha256']==receipt['actual_input_sha256']
        write_json(case/'cpu_run.json',receipt)
        append(MAIN/'EXPERIMENTS.tsv','\t'.join((now.isoformat(),'v-prep','GPT Sol','VP35-'+job['id'],
            'Recover original reference after supervisor inventory exception',
            f"[M]WRFsuccess{marker} frames{len(frames)} scannedNFrows{len(nonfinite)} readErrors{len(receipt['inventory_errors'])} {receipt['status']}; [U]processRC; [I]UTCwall{receipt['monotonic_wall_s']:.3f}s",
            f'artifact={case}/cpu_run.json,{case}/CPU_OUTPUTS.json;cacheNONE;workCPU{job["core"]}NI19JAXcpuGPUWRF0 originaltimeout5400 actualinputSHAs/optionsinreceipt')))
        for p in (case/'cpu/wrf.exe',case/'wrf.exe'):
            if p.is_symlink() or p.exists():p.unlink()
        print(job['id'],receipt['status'],len(frames),flush=True)
        jobs.remove(job)
    time.sleep(.5)
