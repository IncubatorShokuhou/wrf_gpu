"""Capture unchanged native ETARA preprocessing and RADTN cloud fractions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from parse import read_array,operands


def main(input_path,final,reference,output):
    assert final.read_bytes()==reference.read_bytes(),'Diagnostic native final not BYTE'
    inputs=operands(input_path)
    nc,n1,_=inputs['fields'].shape;nz=n1-1
    arrays={k:[] for k in ('pre_t','pre_p','pre_p8w','pre_q','pre_qc','pre_qi','pre_cutop','pre_cubot',
                             'grid_t','grid_p','grid_q','grid_qc','grid_qi','grid_cloud','grid_cutop','grid_cubot')}
    with Path(str(final)+'.capture').open('rb') as f:
        for _ in range(nc):
            assert read_array(f,(),np.int32)==nz
            for k in ('t','p','p8w','q','qc','qi'):
                arrays['pre_'+k].append(read_array(f,(n1 if k=='p8w' else nz,)))
            for k in ('cutop','cubot'):arrays['pre_'+k].append(read_array(f,()))
            assert read_array(f,(),np.int32)==nz
            for k in ('t','p','q','qc','qi','cloud'):
                arrays['grid_'+k].append(read_array(f,(nz,)))
            for k in ('cutop','cubot'):arrays['grid_'+k].append(read_array(f,()))
        assert not f.read(1)
    arrays={k:np.asarray(v,np.float32) for k,v in arrays.items()}
    arrays.update({'in_'+k:v for k,v in inputs.items()})
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(dict(
        columns=nc,levels=nz,final_BYTE=True,
        fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        native_final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        cloudy_levels=int(np.count_nonzero(arrays['grid_cloud'])),
        scope='real-structured CAM01 operands; declared constructed eta/cumulus; native routine boundaries, not original forecast replay'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('input','final','reference','output'):p.add_argument(k,type=Path)
    a=p.parse_args();main(a.input,a.final,a.reference,a.output)
