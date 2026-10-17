"""Recompile unchanged WRF sources and check the sealed corpus and envelopes."""
from pathlib import Path
import ctypes, hashlib, importlib.util, json, shutil, subprocess, sys
import numpy as np

root = Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-ysuqc/snapshot')
out = root.parent / 'reference'
out.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('producer_builder',root/'proofs/t_ysuqc/build_oracle.py')
module = importlib.util.module_from_spec(spec)
sys.argv = [str(root/'proofs/t_ysuqc/build_oracle.py'),str(out)]
spec.loader.exec_module(module)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((root/'proofs/t_ysuqc/ORACLE_MANIFEST.json').read_text())
corpus = Path('<USER_HOME>/wrf_gpu2_lanes/t-ysuqc/oracle.npz')
assert sha(corpus) == manifest['oracle_npz_sha256']
shutil.copyfile(corpus,out/'oracle.npz')
d = np.load(out/'oracle.npz')
mask = np.ones(d['out_zero'].shape,bool)
mask[~d['flag_qi'],:,5] = False
assert np.array_equal(d['out_zero'][mask], d['out_inf'][mask])
report = dict(commit='f68ca7296',corpus_sha256=sha(corpus),source_sha256={},builds={},rows=len(d['nz']))
report['init_sensitivity_consumed_byte'] = True
for name,flags in module.VARIANTS.items():
    fn,build = module.build(name,flags)
    actual = np.zeros_like(d['out_zero'])
    for i,nz in enumerate(d['nz']):
        value = np.zeros((nz+1,9),np.float32,order='F')
        fn(int(nz),int(d['topdown'][i]),1,int(d['flag_qi'][i]),np.asfortranarray(d['profiles'][i]),d['surface'][i].copy(),value)
        actual[i]=value
    expected = d['out_'+name]
    assert np.array_equal(actual[mask],expected[mask]), name
    assert np.isfinite(actual[mask]).all(), name
    report['builds'][name]=dict(build,consumed_values_byte=int(mask.sum()),nonfinite_consumed=0)
report['source_sha256']={k:sha(out/'oracle'/Path(k).name) for k in module.SOURCES}
assert report['source_sha256']==manifest['source_sha256']
assert sha(module.WRAPPER)==manifest['wrapper_sha256']
sys.argv=[str(root/'proofs/t_ysuqc/ulp_envelope.py'),str(out),'96']
run=subprocess.run([sys.executable,*sys.argv],capture_output=True,text=True,check=True,timeout=90)
fresh = np.load(out/'ulp_envelope.npz')
fix=np.load(root/'proofs/t_ysuqc/fixtures/ysu_wrf_oracle.npz')
meta=json.loads((root/'proofs/t_ysuqc/fixtures/ysu_wrf_oracle.json').read_text())
rows=[r['oracle_row'] for r in meta['rows']]
for key in ('profiles','surface','nz','topdown','flag_qi'):
    assert np.array_equal(d[key][rows],fix[key]),key
assert np.array_equal(d['out_zero'][rows],fix['wrf'])
for key in ('lower','upper'):
    assert np.isfinite(fix[key]).all(),key
    assert np.array_equal(fresh[key][rows],fix[key]),key
report['fixture_rows']=len(rows)
report['fixture_envelope_byte']=True
report['envelope']=json.loads((out/'ulp_envelope.json').read_text())
(root.parent/'oracle_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
