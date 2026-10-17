"""Stage complete actual native RADTN call/state, not a JAX comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture_gfdl_lw import read


def main(raw,manifest,final,reference,lw_fixture,output):
    assert final.read_bytes()==reference.read_bytes(),'Native writes changed final original GFDL outputs'
    data,meta=read(raw,manifest)
    with np.load(lw_fixture,allow_pickle=False) as f:
        assert all(data[k].tobytes()==f[k].tobytes() for k in data if k.startswith('co2_'))
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(dict(
        native_final_BYTE=True,calls=len(data['nlayers']),layers=np.unique(data['nlayers']).tolist(),
        fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        native_final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),source_capture=meta,
        co2_provider_BYTE_native_LW=True,
        scope='actual native RADTN complete physical+accumulator outputs; only ETARA-unfilled unread mass padding zero; staticLTOP/PTOPC+CO2provider native'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','manifest','final','reference','lw_fixture','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.raw,a.manifest,a.final,a.reference,a.lw_fixture,a.output)
