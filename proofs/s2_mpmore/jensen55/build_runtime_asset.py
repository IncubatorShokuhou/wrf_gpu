"""Package independently observed original tables; preserve native REAL4 LUTs."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
import numpy as np

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and not out.exists()
out.mkdir()
folder=tree/'proofs/s2_mpmore/jensen55'
spec=importlib.util.spec_from_file_location('original_tables',folder/'reference_tables.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
single=reference.load(tree,'fp32')
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F').read_bytes()
text=re.search(r'igrdata\s*=\s*&\s*\(/(.*?)/\)',source.decode(),re.S|re.I)[1]
text=' '.join(line.split('!')[0].strip().strip('&') for line in text.splitlines())
data={n:single[n] for n in ('itab','itabr','gamma_tab')}
data.update(igrdata=np.asarray([float(a) for a in text.split(',')],np.float64),
            coltab_fp32=single['coltab'],coltabn_fp32=single['coltabn'])
assert np.array_equal(data['igrdata'].astype(np.float32),single['igrdata'])
manifest=json.loads((folder/'stages/manifest.json').read_text())
row=next(r for r in manifest['rows'] if r['mode']=='fp64');path=folder/'stages/fp64.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
with np.load(path) as values:
    data.update(coltab_fp64=values['coltab'].copy(),coltabn_fp64=values['coltabn'].copy())
    assert np.array_equal(values['gamma_table'],data['gamma_tab'].astype(np.float64))
records={n:dict(shape=a.shape,dtype=a.dtype.str,bytes=a.nbytes,
                sha256=hashlib.sha256(a.tobytes()).hexdigest()) for n,a in data.items()}
assert all(np.isfinite(a).all() for a in data.values())
path=out/'lookup_tables.npz';np.savez_compressed(path,**data)
old=json.loads((folder/'fixtures/manifest.json').read_text())
result=dict(source_sha256=hashlib.sha256(source).hexdigest(),original_LUT_file_sha256=old['tables'],
            REAL4_aggregation_source_sha256=manifest['source_sha256'],REAL8_aggregation_fixture_sha256=row['sha256'],
            data=records,raw_host_array_bytes=sum(a.nbytes for a in data.values()),asset_bytes=path.stat().st_size,
            asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope='Original REAL4 QC/QR/gamma binary payloads; native REAL4/8 original aggregation init tables; native literal IGR60. No fitting or generated physics.')
(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(dict(raw_bytes=result['raw_host_array_bytes'],asset_bytes=result['asset_bytes'],sha256=result['asset_sha256']),flush=True)
