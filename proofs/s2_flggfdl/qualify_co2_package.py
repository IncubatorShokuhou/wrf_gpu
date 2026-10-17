"""Check the actual self-contained product loader, compiler/cache and provider."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gpuwrf.physics.gfdleta_co2 import co2_tables,_tables,_ASSETS


def main(operands,reference,cache,output):
    with np.load(operands,allow_pickle=False) as f:
        full=f['in_znw'];half=f['in_znu'][:-1];ptop=f['in_scalars'][0]
    identity=json.loads((_ASSETS/'source_identity.json').read_text())
    for name,digest in identity['asset_sha256'].items():
        assert hashlib.sha256((_ASSETS/name).read_bytes()).hexdigest()==digest,name
    start=time.perf_counter();co2,domain=co2_tables(full,half,ptop,cache_directory=cache)
    cold=time.perf_counter()-start
    start=time.perf_counter();again=co2_tables(full,half,ptop,cache_directory=cache)
    warm=time.perf_counter()-start
    values={**{'co2_'+k:v for k,v in co2.items()},**{'domain_'+k:v for k,v in domain.items()}}
    with np.load(reference,allow_pickle=False) as f:
        verdict={k:v.dtype==f[k].dtype and v.shape==f[k].shape and v.tobytes()==f[k].tobytes() for k,v in values.items()}
    assert all(not v.flags.writeable for v in values.values())
    assert all(again[0][k] is v for k,v in co2.items())
    invalid=False
    try:co2_tables(full[::-1],half,ptop,cache_directory=cache)
    except ValueError:invalid=True
    assert invalid
    report=dict(all28_BYTE=all(verdict.values()),fields=verdict,cold_build_and_grid_s=cold,
        cached_host_provider_s=warm,readonly=True,invalid_sigma_rejected=True,
        table_cache_entries=_tables.cache_info().currsize,assets_identity=True,
        scope='self-contained product native init; no pristine path dependency; CPU only noGPU')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('operands','reference','cache','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.operands,a.reference,a.cache,a.output)
