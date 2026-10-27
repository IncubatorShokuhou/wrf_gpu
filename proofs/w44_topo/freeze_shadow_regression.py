"""Freeze real d03 pristine-WRF temporal operands/answers for portable pytest."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
from oracle import build,load,oracle,sha

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);a.fixture.parent.mkdir(parents=True,exist_ok=True)
case=Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227/wrfinput_d03')
f=load(case);lib,metadata=build(Path('<USER_HOME>/src/wrf_pristine/WRF'),a.out)
arr=np.ctypeslib.ndpointer(dtype=np.float32,flags='F_CONTIGUOUS')
lib.solar_probe.argtypes=[ctypes.c_int]*2+[arr]*2+[ctypes.c_float]*4+[arr]*5
ny,nx=f['HGT'].shape;F=lambda x:np.asfortranarray(np.asarray(x,np.float32).T)
old=np.zeros((ny,nx),np.float32);wanted={15:'sunrise',37:'sunset',38:'night',63:'held_next_sunrise'}
data={k:f[k] for k in ('HGT','XLAT','XLONG','SINALPHA','COSALPHA')};rows=[]
for step in range(max(wanted)+1):
 minute=np.float32(step*30);declin=np.zeros(1,np.float32);outputs=[np.zeros((nx,ny),np.float32,order='F') for _ in range(4)]
 lib.solar_probe(nx,ny,F(f['XLAT']),F(f['XLONG']),np.float32(58+minute/1440),minute,0.,30.,declin,*outputs)
 mask,height,_=oracle(lib,f,minute,old,declin_override=declin[0])
 if step in wanted:
  key=wanted[step]
  for suffix,value in dict(old=old,mask=mask,height=height,csza=outputs[2].T,hrang=outputs[3].T,declin=declin[0]).items():data[key+'_'+suffix]=np.asarray(value)
  edge=np.zeros_like(mask,bool);edge[:2]=True;edge[-2:]=True;edge[:,:2]=True;edge[:,-2:]=True
  rows.append(dict(key=key,minute=float(minute),edge_above_terrain=int(np.count_nonzero(edge&(old>f['HGT']))),shaded=int(mask.sum())))
 old=height
np.savez_compressed(a.fixture,**data)
metadata.update(schema=1,case_sha256=sha(case),domain='d03',shape=[ny,nx],dx_m=f['dx'],dy_m=f['dy'],gmt=0,radt_minutes=30,shadow_length_m=25000,
 rows=rows,fixture_sha256=sha(a.fixture),height_tolerance=dict(atol=.01,rtol=2e-5),mask_tolerance='exact',clock_atol=3e-6,
 source='verbatim pristine WRF REAL topography/solar routines; 30-minute cold-zero trajectory; no port output used')
a.fixture.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(dict(rows=rows,bytes=a.fixture.stat().st_size,fixture_sha256=metadata['fixture_sha256'])),flush=True)
assert 'jax' not in sys.modules
