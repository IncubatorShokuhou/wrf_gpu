"""Actual Swiss CLI loader + two production CPU steps with synthetic turbines.

Real meteorology/grid are unmodified. This is coupled integration evidence,
not CPU-WRF forecast parity nor GPU performance qualification.
"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/v025'))
import cpu_guard  # noqa: E402,F401
sys.path.insert(0,str(ROOT/'proofs/o1_nlbind'))
import cpu_safe  # noqa: E402
cpu_safe.install()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from gpuwrf.contracts import state as sc  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.io.restart import write_restart,read_restart  # noqa: E402
from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry  # noqa: E402
from gpuwrf.io.netcdf_lock import Dataset  # noqa: E402
from gpuwrf.runtime.restart_store import RestartStore,nested_identity  # noqa: E402


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=Path,default=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    sc._gpu_device=lambda:jax.devices('cpu')[0]
    case=args.out/'case';case.mkdir()
    for file in args.case.iterdir():
        if file.name!='namelist.input':(case/file.name).symlink_to(file)
    with Dataset(case/'wrfinput_d01') as ds:
        x,y=len(ds.dimensions['west_east'])//2,len(ds.dimensions['south_north'])//2
    text=(args.case/'namelist.input').read_text()
    text=text.replace('&physics','&physics\n windfarm_opt=1, windfarm_ij=1, windfarm_tke_factor=.25,',1)
    (case/'namelist.input').write_text(text)
    (case/'windturbines-ij.txt').write_text(f'{x+1} {y+1} 1\n{x+1} {y+1} 1\n{x+2} {y+1} 1\n')
    (case/'wind-turbine-1.tbl').write_text('6\n80 80 .12 2000\n.5 .9 0\n2 .85 20\n8 .8 500\n12 .7 1600\n20 .4 2000\n25 .1 2000\n')
    report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),backend='cpu',
                scope='real Swiss meteorology/grid, three synthetic turbines; CPU coupled integration only',
                wrfinput_sha256=hashlib.sha256((case/'wrfinput_d01').read_bytes()).hexdigest(),
                fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS'),steps=[])
    def save():(args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    save();start=time.monotonic()
    cfg=pipeline.NestedPipelineConfig(case,args.out/'output',args.out/'proof',hours=.015,max_dom=1)
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
    n=bundles['d01'].namelist;c=carries['d01'];clock=om.build_clock_base(n)
    assert n.windfarm_opt==1 and len(n.windfarm.turbines)==3 and c.windfarm_power is not None
    report.update(load_s=time.monotonic()-start,turbine_fingerprint=n.windfarm.fingerprint,
                  shape=list(c.state.theta.shape),dt_s=n.dt_s)
    save();print('LOADED',report,flush=True)
    for step in (1,2):
        start=time.monotonic()
        c=om._advance_chunk_fori(c,n,step,clock,n_steps=1,cadence=int(n.radiation_cadence_steps))
        jax.block_until_ready(c)
        bad=sum(int((~np.isfinite(np.asarray(a))).sum()) for a in jax.tree.leaves(c)
                if np.issubdtype(np.asarray(a).dtype,np.floating))
        guard=int(np.asarray(c.census.guards).sum()) if c.census is not None else None
        row=dict(step=step,wall_s=time.monotonic()-start,nonfinite=bad,guard_total=guard,
                 total_power_W=float(np.asarray(c.windfarm_power).sum()),qke_max=float(np.asarray(c.state.qke).max()))
        report['steps'].append(row);save();print(row,flush=True)
        assert bad==0 and guard==0 and row['total_power_W']>0
        if step==1:
            stamp=n.time_utc
            pk=args.out/'coupled.pkl';nc=args.out/'coupled.nc'
            write_restart(c,n,n.grid,step,pk)
            write_wrfrst_carry(c,n.grid,n,nc,step_index=step,run_start=stamp,valid_time=stamp+timedelta(seconds=n.dt_s))
            resumed,_,_,idx=read_restart(pk,namelist=n)
            restored,_=read_wrfrst_carry(nc,namelist=n)
            assert idx==step
            for other in (resumed,restored):
                for a,b in zip(jax.tree.leaves(c),jax.tree.leaves(other),strict=True):np.testing.assert_array_equal(a,b)
            report['restart_all_leaves_byte_exact']=['pickle','netcdf']
            identity=nested_identity(cfg,{'d01':bundles['d01']},n.time_utc)
            store=RestartStore(args.out/'native-store',100*2**20,2,0)
            generation,_=store.save({'d01':c},{'d01':step},{'d01':n.dt_s},identity)
            native,_=store.read(generation,expected_identity=identity)
            native=native['carries']['d01']
            for a,b in zip(jax.tree.leaves(c),jax.tree.leaves(native),strict=True):np.testing.assert_array_equal(a,b)
            report['restart_all_leaves_byte_exact'].append('native-store')
            c=native
            pk.unlink();nc.unlink()
    report['status']='PASS';save()


if __name__=='__main__':main()
