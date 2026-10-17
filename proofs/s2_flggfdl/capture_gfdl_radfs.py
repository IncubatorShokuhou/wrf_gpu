"""Exact native RADFS caller and its dynamic R1/environment/provider."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture_gfdl_lw import read


def main(raw,manifest,final,reference,lw_fixture,output):
    assert final.read_bytes()==reference.read_bytes(),'Native writes changed full GFDL output'
    data,meta=read(raw,manifest)
    assert all(np.all(data[k]==v) for k,v in dict(ko3=1,ibeg=1,kalb=0).items())
    with np.load(lw_fixture,allow_pickle=False) as f:
        assert all(data[k].tobytes()==f[k].tobytes() for k in data if k.startswith('co2_')),'CO2 provider differs from native LW'
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(dict(
        native_final_BYTE=True,calls=len(data['nlayers']),layers=np.unique(data['nlayers']).tolist(),
        environment_R1=np.unique(data['environment_r1']).tolist(),co2_provider_BYTE_native_LW=True,
        fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        native_final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),source_capture=meta,
        scope='actual native WRF-ARW RADFS allcallerarrays/tenphysicaloutputs; CO2oracleprovider only, notgenericinit'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','manifest','final','reference','lw_fixture','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.raw,a.manifest,a.final,a.reference,a.lw_fixture,a.output)
