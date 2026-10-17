"""Small diagnostic controls; synthetic bytes are parser tests, never WRF truth."""
import importlib.util,json,pathlib,struct,tempfile
import numpy as np
spec=importlib.util.spec_from_file_location('f2_reader',pathlib.Path(__file__).with_name('analyze.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/t-f2neg/reader_test');root.mkdir(exist_ok=True)
p=root/'sample.bin'
base=b''.join(struct.pack('>iii',step,1 if step==1 else 0,0)+(struct.pack('>iiif',0,2,3,-.25) if step==1 else b'') for step in range(1,4801))
p.write_bytes(base);r,c=m.read_field(p)
assert r['negative_count']==r['ring0_count']==1 and r['interior_count']==r['nonfinite_count']==0
assert r['minimum_negative']==dict(step=1,index_kji=[3,2,0],value=-.25)
assert all(x==-.25 for x in r['negative_value_quantiles']['all'].values())
mutants={'truncated':base[:-1],'nonfinite':base[:24]+struct.pack('>f',float('-inf'))+base[28:], 'nonnegative':base[:24]+struct.pack('>f',.25)+base[28:],'out_of_bounds':base[:12]+struct.pack('>i',42)+base[16:],'missing_step':base[:28]+struct.pack('>i',3)+base[32:]}
killed=[]
for name,raw in mutants.items():
 p.write_bytes(raw)
 try:m.read_field(p)
 except AssertionError:killed.append(name)
 else:raise AssertionError(f'{name} accepted')
assert len(killed)==len(mutants)
p.unlink();root.rmdir()
receipt=dict(known_big_endian_record='PASS',quantiles_counts_indices='PASS',mutants_rejected=killed,physical_evidence=False)
(pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/t-f2neg')/'FN04_READER.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
