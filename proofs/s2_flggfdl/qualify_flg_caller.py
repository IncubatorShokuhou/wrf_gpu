"""Final FLG preprocessing+physics caller vs original native 48-column call."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gpuwrf.physics import ra_flg
from literal import load,_wrap
from parse import operands,read_array


def main(input_path,native_path,output):
    data=operands(input_path);fields=data['fields'];sfc=data['sfc'];nc,n1,_=fields.shape;nz=n1-1
    with native_path.open('rb') as stream:
        assert np.array_equal(read_array(stream,(2,),np.int32),[nc,nz])
        rates=read_array(stream,(nc,nz,3));flux=read_array(stream,(nc,14));assert not stream.read(1)
    port=load(Path(ra_flg.__file__))
    mapping=dict(glw=0,gsw=1,swdn=2,sw_up_top=4,lw_up_top=5,direct=6,diffuse=7,dni=8)
    report=[]
    for row in range(nc):
        f=fields[row];s=sfc[row];ptop,gmt,radt,dt=data['scalars']
        pre=port.preprocess_column(*_wrap(tuple(f[:,i] for i in (12,4,0,3,5,6,7,8,9,10))),
            *_wrap(tuple(s[i] for i in (2,1,0,4,5))),data['julday'][row],gmt,data['tau'][row]*np.float32(60.),
            radt,data['declin'][row],data['solcon'][row])
        candidate=port._column_from_preprocessed(pre,_wrap(f[:,2]))
        checks={}
        for key,value in candidate.items():
            value=np.asarray(value)
            ref=rates[row,:,('total_rate','lw_rate','sw_rate').index(key)] if key.endswith('_rate') else flux[row,mapping[key]]
            assert value.dtype==np.float32 and np.isfinite(value).all() and value.shape==ref.shape
            checks[key]=value.tobytes()==ref.tobytes()
        report.append(dict(row=row,fields=checks));print(row,all(checks.values()),flush=True)
        output.write_text(json.dumps(dict(all48_BYTE=all(all(r['fields'].values()) for r in report),completed=len(report),
            results=report,native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
            scope='changed actual-source T8W and allocatedQVghost0 caller; 40realstructured+8mid/polarseasonstress; literalREAL4'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('input','native','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.input,a.native,a.output)
