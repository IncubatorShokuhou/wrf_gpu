"""Regenerate host metadata corrections from the original-frame audit packet."""
import argparse
import hashlib
import json
from pathlib import Path


def generate(audit, root):
    packet = json.loads(audit.read_text())
    variables = {}
    corrections = set()
    for row in packet['rows']:
        variables.update({k:v for k,v in row['cpu_inventory'].items()
                          if k not in variables and k not in ('Times','XTIME')})
        corrections.update(k.rsplit('.',1)[0] for k in row['metadata'])
    fixture = dict(source=str(audit), audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
    supplement=root/'proofs/t_writer/jensen_metadata_source.json'
    if supplement.exists():
        variables.update(json.loads(supplement.read_text())['variables'])
        fixture['supplement_sha256']=hashlib.sha256(supplement.read_bytes()).hexdigest()
    (root/'proofs/t_writer/cpu_schema.json').write_text(json.dumps(fixture)[:-1]+',\n"variables": {\n'+
        ',\n'.join(json.dumps(k)+': '+json.dumps(v) for k,v in sorted(variables.items()))+'\n}}\n')
    metadata = {k:{a:v['attrs'][a] for a in ('description','units','MemoryOrder','stagger','coordinates')
                   if a in v['attrs']} for k,v in variables.items()}
    (root/'src/gpuwrf/io/wrf_variable_metadata.py').write_text(
        '"""Original CPU-WRF v4.7.1 metadata corrections (host serialization only).\n\n'
        'Source inventory and audit SHA: proofs/t_writer/cpu_schema.json.\n"""\n\n'
        'WRF_VARIABLE_METADATA = {\n'+''.join(f'    {k!r}: {v!r},\n' for k,v in sorted(metadata.items()))+'}\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('audit',type=Path)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    generate(args.audit,args.root)
