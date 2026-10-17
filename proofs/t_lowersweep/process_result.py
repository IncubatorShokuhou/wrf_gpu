"""Classify a child using its own systemd scope, never rc124 alone."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def classify(receipt, out, returncode, started_epoch, finished_epoch, budget_s=900):
    out=Path(out)
    scope=receipt.get('systemd_scope')
    boot=receipt.get('boot_id')
    base=['journalctl','--user','--no-pager','-o','json',
          '--since',f'@{int(started_epoch)-2}','--until',f'@{int(finished_epoch)+2}']
    if boot:base += [f'--boot={boot.replace("-", "")}']
    diagnostic=None
    def query(args):
        try:
            return subprocess.run(args,capture_output=True,text=True,timeout=10)
        except (OSError,subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(args,1,'',repr(exc))
    if not scope:
        found=query(base+['--grep',re.escape(str(out))])
        for line in found.stdout.splitlines():
            entry=json.loads(line)
            if str(out) in entry.get('MESSAGE','') and 'lower_case.py' in entry.get('MESSAGE',''):
                scope=entry.get('USER_UNIT')
                if scope:break
        if found.returncode:diagnostic=found.stderr.strip()
    entries=[]
    if scope:
        result=query(base+[f'USER_UNIT={scope}'])
        entries=[json.loads(line) for line in result.stdout.splitlines()]
        if result.returncode:diagnostic=result.stderr.strip()
    messages=[e.get('MESSAGE','') for e in entries]
    oom=any('oom-kill' in m or 'kernel OOM killer' in m for m in messages)
    elapsed=finished_epoch-started_epoch
    timestamps=[int(e['__MONOTONIC_TIMESTAMP']) for e in entries if '__MONOTONIC_TIMESTAMP' in e]
    scope_elapsed=(max(timestamps)-min(timestamps))/1e6 if len(timestamps)>1 else None
    kind=('OOM_AT_CAP' if oom else 'COMPLETED' if returncode==0 else
          'TIMEOUT' if returncode==124 and scope_elapsed is not None and scope_elapsed>=budget_s-1 else
          'EARLY_EXIT' if returncode==124 else 'PROCESS_FAIL')
    evidence=dict(classification=kind,returncode=returncode,elapsed_wall_s=elapsed,
        scope_elapsed_wall_s=scope_elapsed,elapsed_scope='parent includes RAM queue; scope clock excludes queue',
        systemd_scope=scope,oom_kill_confirmed=oom,
        journal_status='READ' if entries else 'UNKNOWN',journal_diagnostic=diagnostic,
        memory_peak_messages=[m for m in messages if 'memory peak' in m],
        messages=messages)
    raw=(json.dumps(evidence,indent=2)+'\n').encode()
    (out/'process_result.json').write_bytes(raw)
    evidence['artifact']=str(out/'process_result.json')
    evidence['sha256']=hashlib.sha256(raw).hexdigest()
    return evidence
