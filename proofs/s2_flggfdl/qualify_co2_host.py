"""Measure extracted native initializer and compare all native provider bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from capture_gfdl_lw import MATRICES,LAYERS
from extract_co2_host import CO2


def main(executable,operands,reference,output):
    with np.load(operands,allow_pickle=False) as f:
        full=f['in_znw'].copy();half=f['in_znu'][:-1].copy();ptop=f['in_scalars'][0]
    nz=len(half);input_file=output.with_suffix('.input.bin');result_file=output.with_suffix('.tables.bin')
    with input_file.open('wb') as f:
        np.asarray(nz,np.int32).tofile(f);np.asarray(ptop,np.float32).tofile(f)
        full.tofile(f);half.tofile(f)
    timings=[]
    for _ in range(3):
        started=time.perf_counter()
        subprocess.run([str(executable),str(input_file),str(result_file)],cwd=executable.parent,
                       timeout=30,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        timings.append(time.perf_counter()-started)
    with result_file.open('rb') as f:
        assert np.fromfile(f,np.int32,1)[0]==nz
        tables=dict(domain_ltop=np.fromfile(f,np.int32,3),domain_ptopc=np.fromfile(f,np.float32,4))
        for raw_name in CO2:
            key=raw_name.lower();shape=(nz+1,nz+1) if key in MATRICES else (nz,) if key in LAYERS else (nz+1,)
            value=np.fromfile(f,np.float32,int(np.prod(shape))).reshape(shape,order='F')
            assert np.isfinite(value).all();tables['co2_'+key]=value
        assert not f.read(1)
    with np.load(reference,allow_pickle=False) as f:
        verdict={k:tables[k].dtype==f[k].dtype and tables[k].shape==f[k].shape and tables[k].tobytes()==f[k].tobytes() for k in tables}
    report=dict(all28_BYTE=all(verdict.values()),fields=verdict,levels=nz,process_wall_s=timings,
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
                scope='source-preserving native CPU init only; explicit grid input; original native CO2/domain table oracle; no forecast-step host call/GPU claim')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('executable','operands','reference','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.executable,a.operands,a.reference,a.output)
