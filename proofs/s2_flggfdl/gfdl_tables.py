"""Seal grid-independent native GFDL tables; CO2 tables are grid-dependent."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


SHAPES = dict(em1=(28,180),em1wde=(28,180),table1=(28,180),
              table2=(28,180),table3=(28,180),em3=(28,180),
              source=(28,15),dsrce=(28,15),prgfdl=(81,),
              xduo3n=(37,81),xdo3n2=(37,81),xdo3n3=(37,81),xdo3n4=(37,81),
              ab15wd=(),sko2d=(),skc1r=(),sko3r=())


def main(raw, output):
    tables = {}
    with raw.open('rb') as f:
        for key,shape in SHAPES.items():
            value=np.fromfile(f,np.float32,int(np.prod(shape)))
            assert value.size==np.prod(shape) and np.isfinite(value).all()
            tables[key]=value.reshape(shape,order='F')
        assert not f.read(1)
    np.savez_compressed(output,**tables)
    output.with_suffix('.json').write_text(json.dumps(dict(
        shapes=SHAPES,native_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        scope='unchanged WRF TABLE/O3CLIM; CO2 and seasonal/solar state excluded'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.raw,a.output)
