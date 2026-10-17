"""Stage real adapter operands and run the actual unmodified WRF wrapper oracle."""
import argparse
from pathlib import Path
import struct
import subprocess
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.coupling.physics_couplers import _wrf_phy_prep_p8w


def main():
    p=argparse.ArgumentParser(); p.add_argument('build');p.add_argument('out');args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parents[3]/'tests/v040/milbrandt'
    with np.load(here/'wrf_real4.npz') as f: a,dt=f['inputs'].copy(),f['dt']
    n,nz,_=a.shape
    # Inputs to the WRF physics driver include pi_phy, computed by the retained
    # port's already-qualified phy_prep. The independent wrapper receives EXACTLY
    # those same rounded operands, avoiding a different Exner realization.
    pii=np.asarray(jax.jit(lambda p:(p/jnp.float32(100000.))**jnp.float32(2./7.))(jnp.asarray(a[:,:,15])))
    dry=(a[:,:,0]/pii).astype(np.float32)
    ratio=np.float32(461.6)/np.float32(287.)
    theta=(dry*(np.float32(1.)+ratio*a[:,:,1])).astype(np.float32)
    a[:,:,0]=theta
    z=np.broadcast_to((np.arange(nz+1)*100).astype(np.float32)[:,None,None],(nz+1,1,n)).copy()
    pressure=jnp.asarray(a[:,:,15].T[:,None,:])
    ps=np.asarray(jax.jit(lambda p,z:_wrf_phy_prep_p8w(p,z,jnp.full(nz,.5,jnp.float32),jnp.full(nz,.5,jnp.float32)))(pressure,jnp.asarray(z)))[0,0]
    raw=np.concatenate((a,pii[:,:,None]),axis=2)
    inp=root/'wrapper_input.bin';out=root/'wrapper_output.bin'
    inp.write_bytes(struct.pack('iif',n,nz,float(dt))+raw.tobytes(order='F')+ps.astype(np.float32).tobytes())
    subprocess.run([str(Path(args.build)/'wrapper'),str(inp),str(out)],check=True)
    data=np.fromfile(out,np.float32);end=n*nz*14
    fields=data[:end].reshape((n,nz,14),order='F').transpose(0,2,1)
    rain=data[end:end+n*9].reshape((n,9),order='F')
    refl=data[end+n*9:].reshape((n,nz),order='F')
    np.savez_compressed(root/'wrapper_real4.npz',inputs=a,ps=ps,dt=dt,pii=pii,z=z,ref_fields=fields,ref_rain=rain,ref_zet=refl)
    meta={'source':'unmodified module_mp_milbrandt2mom.F full module with error stub (no errors called)',
          'source_sha256':'76d11c00976f3c7dac30ff6ebb3efac1698bf43c13131b3aae43e464f22d23e5',
          'operand_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
          'fixture_sha256':hashlib.sha256((root/'wrapper_real4.npz').read_bytes()).hexdigest(),
          'work':'12 active columns x12 levels dt18; theta_m prep+finish, all actual WRF wrapper outputs',
          'pii':'same rounded retained phy_prep operand passed to independent WRF wrapper',
          'tolerances':'same registered rtol3e-5, fields atol3e-5/3e-10/3e-3; amounts3e-7mm/radar3e-4dBZ'}
    (root/'wrapper_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta)

if __name__=='__main__':main()
