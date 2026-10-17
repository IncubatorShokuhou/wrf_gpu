"""Pack independently compiled original-WRF arrays and input-ULP envelopes."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

FIELDS={'theta':'TH_OUT','qv':'QV_OUT','qc':'QC_OUT','qr':'QR_OUT','qi':'QI_OUT','qs':'QS_OUT','qg':'QG_OUT','qh':'QH_OUT'}
FIELDS.update({'re_'+k+'_gsfc':'RE_'+k.upper() for k in ['cloud','rain','ice','snow','graupel','hail']})
FIELDS.update({k:k.upper() for k in ['phys'+x for x in 'cedsmf']+['acphys'+x for x in 'cedsmf']+['preci3d','precs3d','precg3d','prech3d','precr3d']})
SURFACE={'rainnc':'RAINNCV','snownc':'SNOWNCV','graupelnc':'GRAUPELNCV','hailnc':'HAILNCV','sr':'SR'}
INPUT=['TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','QH_IN','DEN','PII','P','DELZ','Z','W']
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def get(data,key):
 return np.asarray(data['columns'].get(key,data['scalars'].get(key)),np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);a=p.parse_args()
 payload={};meta={'groups':{},'oracle':'unmodified pristine WRF 4.7.1 module_mp_gsfcgce_4ice_nuwrf.F','compiler_policy':'literal REAL4 BYTE; compiled independent input-ULP and original FMA spread, disclosed excess pending D6'}
 for group,folder,fma,ulp in [('base','fullzero','fma90','ulp90'),('water18','holdout18','fma18','ulp18'),('land54','holdout54','fma54','ulp54')]:
  directory=a.artifacts/folder;provenance=json.loads((directory/'provenance.json').read_text())
  meta['groups'][group]={'caller':provenance.get('caller',{'dt':90.,'dx':3000.,'steps':1,'land':1}),'source_sha256':provenance['sources']['module_mp_gsfcgce_4ice_nuwrf.F'],'cases':{}}
  for case in range(1,7):
   base=json.loads((directory/f'case{case}.json').read_text());native=json.loads((a.artifacts/fma/f'case{case}.json').read_text())
   assert all(get(base,k).tobytes()==get(native,k).tobytes() for k in INPUT),'FMA compiled driver changed inputs'
   for key in INPUT:payload[f'{group}_{case}__{key}']=get(base,key)
   info={'species_nonzero':{key:int(np.count_nonzero(get(base,key))) for key in INPUT[1:8]},'hail_surface_mm':float(get(base,'HAILNCV'))}
   meta['groups'][group]['cases'][str(case)]=info
   for name,key in {**FIELDS,**SURFACE}.items():
    ref=get(base,key);spread=np.abs(get(native,key).astype(float)-ref.astype(float));bound=np.zeros_like(spread)
    for index in range(1,15):
     pair=[]
     for direction in ('p','m'):
      variant=json.loads((a.artifacts/ulp/f'case{case}_v{index}_{direction}.json').read_text())
      pair.append(np.abs(get(variant,key).astype(float)-ref.astype(float)))
     bound+=np.maximum(*pair)
    payload[f'{group}_{case}__{name}']=ref
    payload[f'{group}_{case}__{name}_bound']=bound+spread
    payload[f'{group}_{case}__{name}_fma']=spread
 output=ROOT/'tests/v040/mpmisc/goddard4ice_columns.npz';np.savez_compressed(output,**payload)
 meta['sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
 output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 print(output,output.stat().st_size)
if __name__=='__main__':main()
