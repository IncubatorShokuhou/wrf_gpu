"""Pristine public columns under alternate compiler-defined local REAL init."""
from pathlib import Path
import hashlib,json,sys,types

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and not artifact.exists()
builder=tree/'proofs/s2_mpmore/jensen55/build_reference.py'
source=builder.read_text();assert source.count("'-finit-real=zero'")==1
alternate=source.replace("'-finit-real=zero'","'-finit-real=inf'")
module=types.ModuleType('original_with_alternate_init');module.__file__=str(builder)
exec(compile(alternate,str(builder),'exec'),module.__dict__)
module.main(tree,artifact)
rows=[]
for mode in ('fp32','fp64'):
    for case in range(1,18):
        baseline=tree/f'proofs/s2_mpmore/jensen55/fixtures/{mode}_{case}.json'
        observed=artifact/f'fixtures/{mode}_{case}.json'
        a=json.loads(baseline.read_text());b=json.loads(observed.read_text())
        changed=[name for name in a['output'] if a['output'][name]!=b['output'][name]]
        scalar=[name for name in a['precipitation'] if a['precipitation'][name]!=b['precipitation'][name]]
        rows.append(dict(mode=mode,case=case,BYTE=baseline.read_bytes()==observed.read_bytes(),
                         changed_fields=changed,changed_scalars=scalar,finite=b['finite'],
                         alternate_sha256=hashlib.sha256(observed.read_bytes()).hexdigest()))
out=artifact.parent/'MPM55V-init-sensitivity.json'
out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(module.SOURCE.read_bytes()).hexdigest(),
    scope='Identical original public callers/inputs/tables and original bodies; -finit-real=inf versus frozen zero;FP traps off',
    rows=rows),indent=2)+'\n')
print(dict(columns=len(rows),BYTE=sum(r['BYTE'] for r in rows),finite=sum(r['finite'] for r in rows)),flush=True)
