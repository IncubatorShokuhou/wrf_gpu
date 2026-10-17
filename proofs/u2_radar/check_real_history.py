"""Full-frame saved-REFL parity and offline CPU cost, never forecast timing."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import time
import tracemalloc

import numpy as np
import xarray as xr

from gpuwrf.post.radar import radar


def check(reference,output, *, cost_only=False):
    paths=sorted(reference.glob('wrfout_d01_*'))
    if len(paths)!=4:
        raise ValueError('expected all four Swiss radar-ON frames')
    results=[];small=[]
    for p in (paths[-1:] if cost_only else paths):
        with xr.open_dataset(p,decode_times=False) as raw:
            ds=raw[['REFL_10CM','PH','PHB','HGT']].load()
            raw_dbz=ds.REFL_10CM.values.copy()
            ix=np.unravel_index(raw_dbz.argmax(),raw_dbz.shape)
            y0,x0=ix[-2:]
            ys=np.clip(np.arange(y0-1,y0+2),0,ds.sizes['south_north']-1)
            xs=np.clip(np.arange(x0-1,x0+2),0,ds.sizes['west_east']-1)
            small.append({name:ds[name].isel(south_north=ys,west_east=xs).values for name in ds.data_vars})
        out=radar(ds)
        assert out.radar_dbz_s.values.tobytes()==raw_dbz.tobytes()
        assert out.radar_dbz_s.dtype==raw_dbz.dtype
        assert np.array_equal(out.radar_composite_s.values,raw_dbz.max(axis=1))
        assert all(np.isfinite(v.values).all() for v in out.data_vars.values())
        results.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'shape':list(raw_dbz.shape),'byte_exact':True,'max_dbz':float(raw_dbz.max()),
                        'echo_columns':int((out.radar_echo_top_status_s.values>0).sum())})
    times=[]
    for _ in range(9):
        begin=time.perf_counter();radar(ds);times.append(time.perf_counter()-begin)
    tracemalloc.start();radar(ds);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    output.mkdir(parents=True,exist_ok=True)
    if not cost_only:
        np.savez_compressed(output/'saved_swiss.npz',**{name:np.concatenate([v[name] for v in small],axis=0) for name in small[0]})
    manifest={'frames':results,'component_cpu_seconds_median':statistics.median(times),
              'component_cpu_seconds_all':times,'tracemalloc_peak_bytes':peak,
              'timing_scope':'saved S-band full42x42x44 postprocessing only, data already read; shared nice19CPU28; not forecast/GPU timing',
              'fixture_scope':'none, cost-only' if cost_only else 'exact3x3 crop at each frame maximum; original44levels;4UTCframes',
              'fixture_sha256':None if cost_only else hashlib.sha256((output/'saved_swiss.npz').read_bytes()).hexdigest()}
    (output/'saved_swiss_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cost-only',action='store_true')
    a=p.parse_args();check(a.reference,a.output,cost_only=a.cost_only)
