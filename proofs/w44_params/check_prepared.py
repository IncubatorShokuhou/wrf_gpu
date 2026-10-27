"""Authenticate all four family/domain preparations before any warm model step."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from phase_journal import write_receipt
from prepared_inventory import artifact_bindings

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('cache',type=Path);p.add_argument('family')
p.add_argument('--cpu-smoke',action='store_true',help='CPU launcher rehearsal: accept only CPU-smoke receipts')
a=p.parse_args();source=Path(__file__).resolve().parents[2]
pin=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
selected={}
for path in sorted(a.run.glob('*-d??-*/receipt.json')):
    row=json.loads(path.read_text())
    if row['pin']!=pin or row['status']!='PASS' or row['family']!=a.family:continue
    # The CPU-smoke mini model steps to prove replay; a GPU preparation never steps.
    assert row['platform']==('cpu' if a.cpu_smoke else 'gpu') and bool(row['cpu_smoke'])==a.cpu_smoke and (a.cpu_smoke or not row['stepping_executed'])
    assert Path(row['cache']).resolve()==(a.cache/'jax').resolve()
    selected[row['variant'],row['domain']]={'receipt':str(path),'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'artifacts':artifact_bindings(row['aot'],a.cache/'jax')}
assert set(selected)=={(v,d) for v in ('off','on') for d in ('d01','d02')},selected
write_receipt(a.run/'prepared.json',{'pin':pin,'preparations':{':'.join(k):v for k,v in selected.items()},
              'metadata':{'nested_aot':{'domains':{'d01':{},'d02':{}}}}})
