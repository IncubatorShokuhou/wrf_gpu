"""Compare complete seeded pattern driver with original CPU-WRF history/seed state."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.stochastic_config import bind_stochastic
from gpuwrf.physics.stochastic_driver import initial_stochastic,advance_stochastic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "cpu"
    nml = parse_namelist(args.reference/"namelist.input")
    cfg = bind_stochastic(nml,"d01")
    with Dataset(args.reference/"wrfinput_d01") as ds:
        grid = SimpleNamespace(nx=len(ds.dimensions['west_east']),ny=len(ds.dimensions['south_north']),
            nz=len(ds.dimensions['bottom_top']),dx=float(ds.DX),dy=float(ds.DY))
        start = datetime.fromisoformat(str(ds.START_DATE).replace('_','T'))
    dt = nml['domains']['time_step']
    carry = initial_stochastic(cfg,grid,dt,start)
    fn = jax.jit(lambda c: advance_stochastic(c,cfg,dx=grid.dx,dy=grid.dy))
    rows = []
    mapping = ((0,0,'RU_TENDF_STOCH'),(0,1,'RV_TENDF_STOCH'),(1,0,'RT_TENDF_STOCH'),
        (2,0,'RSTOCH'),(3,0,'RAND_PERT'),(4,0,'PATTERN_SPP_CONV'),(5,0,'PATTERN_SPP_PBL'),(6,0,'PATTERN_SPP_LSM'))
    histories = sorted(args.reference.glob('wrfout_d01_*'))
    assert len(histories) == 4
    for step,path in enumerate(histories):
        if step:
            carry = fn(carry)
        with Dataset(path) as ds:
            for index,component,name in mapping:
                if name not in ds.variables:
                    continue
                reference = np.asarray(ds[name][0])
                value = carry.patterns[index].field
                value = value[component] if index==0 else value
                value = np.asarray(value)
                if value.shape[0] == 1:
                    value = np.broadcast_to(value,(reference.shape[0],*value.shape[1:]))
                value = value[:,:reference.shape[1],:reference.shape[2]]
                assert np.isfinite(reference).all() and np.isfinite(value).all()
                error = float(np.max(np.abs(value-reference)))
                scale = float(np.max(np.abs(reference)))
                floor = 64*np.finfo('float32').eps*scale
                np.testing.assert_allclose(value,reference,rtol=8e-6,atol=floor,err_msg=f'step{step}:{name}')
                rows.append({'step':step,'field':name,'max_abs':error,'scale':scale})
        if step:
            restart = args.reference/path.name.replace('wrfout_','wrfrst_')
            with Dataset(restart) as ds:
                keys=('ISEEDARR_SKEBS','ISEEDARR_SPPT','ISEEDARR_RAND_PERTURB','ISEEDARRAY_SPP_CONV','ISEEDARRAY_SPP_PBL','ISEEDARRAY_SPP_LSM')
                for seed,name in zip(carry.seeds,keys):
                    np.testing.assert_array_equal(np.asarray(seed),np.asarray(ds[name][...]).reshape(-1),err_msg=f'step{step}:{name}')
    result={'backend':'cpu','rows':rows,'seed_words_exact':3*6*8,'source':'ORIGINAL CPU-WRF3step trueSwissIC',
            'scope':'complete seeded patterns and restart seeds; model coupling not asserted here'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
