"""Diagnostic replay of a retained actual AOT executable; never compile a substitute."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--hlo',required=True)
    parser.add_argument('--stop-step',type=int,required=True)
    args=parser.parse_args()
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
    fd=int(os.environ['GPUWRF_GPU_LOCK_FD'])
    assert os.fstat(fd).st_ino==os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
    import gpuwrf
    import jax
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.aot_precompile import load_domain_blob
    from gpuwrf.runtime.finite_state_guard import first_nonfinite_state_location
    assert jax.devices()[0].platform=='gpu'
    source=Path(om.__file__).resolve().parents[3]
    assert subprocess.run(['git','-C',str(source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode!=0
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    args.output.mkdir(parents=True,exist_ok=True)
    with args.snapshot.open('rb') as f: saved=pickle.load(f)
    carry=jax.device_put(saved['carry']); ns=jax.device_put(saved['namelist']); clock=jax.device_put(saved['clock_base'])
    advance,status=load_domain_blob('d01',args.cache,hlo_sha256=args.hlo,return_status=True)
    assert advance is not None and status['loaded'],status
    print('ACTUAL_AOT '+json.dumps(status),flush=True)
    rows=[]; first=None; t0=time.monotonic()
    for step in range(int(saved['step'])+1,args.stop_step+1):
        # This host diagnostic is outside the production timestep. Donation
        # requires preserving the predecessor before calling the same executable.
        before=jax.device_get(carry)
        carry=om.unalias_donated_carry(carry,by_buffer=True)
        carry=advance(carry,ns,jax.numpy.asarray(step,jax.numpy.int32),clock,
            n_steps=1, cadence=int(ns.radiation_cadence_steps))
        loc=first_nonfinite_state_location(carry.state,domain='d01',step=step,sim_time_s=step*float(ns.dt_s))
        rows.append(dict(step=step,first_bad=None if loc is None else asdict(loc)))
        if loc is not None:
            first=asdict(loc)
            for label,value,at in [('last_finite',before,step-1),('first_nonfinite',jax.device_get(carry),step)]:
                with (args.output/(label+'.pkl')).open('xb') as f:
                    pickle.dump(dict(carry=value,namelist=saved['namelist'],clock_base=saved['clock_base'],step=at),f,protocol=5)
            print('FIRST_NONFINITE '+json.dumps(first),flush=True)
            break
        if step%10==0: print('finite step='+str(step),flush=True)
    receipt=dict(platform='gpu',source=str(source),source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        snapshot=str(args.snapshot),snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache=str(args.cache),actual_aot=status,
        first_bad=first,rows=rows,wall_s=time.monotonic()-t0,finished_utc=datetime.now(timezone.utc).isoformat(),
        scope='diagnostic own-step boundaries using the actual saved native executable; host predecessor snapshots; no performance or original-WRF fidelity claim')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
