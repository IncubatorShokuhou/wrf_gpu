"""Read original CPU-WRF tables as explicit oracle-probe operands, never baked IO."""
from functools import lru_cache
from pathlib import Path
import hashlib,json,re
import numpy as np


@lru_cache(maxsize=2)
def load(tree,mode):
    tree=Path(tree);folder=tree/'proofs/s2_mpmore/jensen55'
    R=np.float32 if mode=='fp32' else np.float64;tables={}
    manifest=json.loads((folder/'fixtures/manifest.json').read_text())
    source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F').read_bytes()
    assert hashlib.sha256(source).hexdigest()==manifest['source_sha256']
    values=re.search(r'igrdata\s*=\s*&\s*\(/(.*?)/\)',source.decode(),re.S|re.I)[1]
    values=' '.join(line.split('!')[0].strip().strip('&') for line in values.splitlines())
    tables['igrdata']=np.asarray([float(a) for a in values.split(',')],R)
    assert len(tables['igrdata'])==60
    for field,file,nchan in [('itab','ishmael-qi-qc.bin',2),('itabr','ishmael-qi-qr.bin',6)]:
        path=Path('<USER_HOME>/src/wrf_pristine/WRF/run')/file
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['tables'][file]
        dtype=np.dtype([('head','>i4'),('data','>f4',(nchan,)),('tail','>i4')])
        raw=np.fromfile(path,dtype)
        assert len(raw)==51*51*51*11 and np.all(raw['head']==4*nchan) and np.all(raw['tail']==4*nchan)
        tables[field]=raw['data'].astype(R).reshape((51,51,51,11,nchan))
    with np.load(folder/'stages'/(mode+'.npz')) as data:
        for field in ('gamma_table','coltab','coltabn'):
            tables['gamma_tab' if field=='gamma_table' else field]=data[field].copy()
    for value in tables.values():
        assert np.isfinite(value).all()
        value.flags.writeable=False
    return tables
