"""Freeze original19/21 init observations, before canonical runtime remapping."""
from pathlib import Path
import hashlib,json,sys

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
declarations={}
for line in (tree/'proofs/v034/oracle/nssl2mom/module_vars_dims.txt').read_text().splitlines():
    if line.strip():
        name,typ,kind,parameter,dims=line.split()
        declarations[name]=[typ,kind,dims]
asset={'decls':{},'provenance':{
    'reference':'proofs/s2_mpmore/nssl1mom/fixtures',
    'scope':'Exact original init module_vars; canonical padding and index remap applied by get_one_moment_constants',
    'source_sha256':'29f42e76fbeca027a3ffb2ac805b148aaa1425c7f99ed1d8f19a90ebc1d015f8'}}
for mp in (19,21):
    asset[str(mp)]={}
    for mode in ('fp32','fp64'):
        fixture=tree/f'proofs/s2_mpmore/nssl1mom/fixtures/v{mp}_{mode}_1.json'
        values=json.loads(fixture.read_text())['module_vars']
        asset[str(mp)][mode]=values
        for name in values:
            asset['decls'][name]=declarations[name]
out.write_text(json.dumps(asset,sort_keys=True)+'\n')
original=tree/'src/gpuwrf/physics/nssl2mom/data/one_moment_constants.json'
assert out.read_bytes()==original.read_bytes(), 'Generated asset must be BYTE vs shipped original observations'
print(dict(BYTE=True,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
