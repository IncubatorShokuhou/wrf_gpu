"""Seven 1000-step original-WRF AR/FFT trajectories and spectra/amplitudes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.stochastic_rng import seed_array,gaussian_innovations
from gpuwrf.physics.stochastic_setup import initialize_spectrum
from gpuwrf.physics.stochastic import update_coefficients,spectral_field


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oracle",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert jax.devices()[0].platform == "cpu"
    jax.config.update("jax_enable_x64",True)
    source=(args.oracle/"oracle.bin").read_bytes()
    offset=0
    def read(shape,dtype=np.float32):
        nonlocal offset
        n=int(np.prod(shape))
        result=np.frombuffer(source,dtype=dtype,count=n,offset=offset).reshape(shape)
        offset+=n*4
        return result
    records=[]
    start=time.perf_counter()
    for variable in "WTPRSQO":
        plan=initialize_spectrum(18,16,5,variable=variable,vertical=1,dx=9000.,dy=9000.,dt=54,
            stddev=.5,lengthscale=15000.,timescale=21600.,backscatter=1.e-6,
            decorrelation=10800.,exponent=-1.83)
        amp,alpha=jnp.asarray(plan["amplitude"]),jnp.asarray(plan["alpha"])
        kw={k:jnp.asarray(plan[k]) for k in ("vertical_cosine","vertical_sine","vertical_amplitude")}
        def evolve(carry):
            s,c,seed=carry
            ns,nc,seed=gaussian_innovations(seed,s.shape)
            s,c=update_coefficients(s,c,amp,alpha,ns,nc)
            return s,c,seed
        def sample(carry):
            carry=jax.lax.fori_loop(0,10,lambda _,c:evolve(c),carry)
            s,c,seed=carry
            a=spectral_field(s,c,component="U" if variable=="W" else "T",dx=9000.,dy=9000.,**kw)
            b=spectral_field(s,c,component="V",dx=9000.,dy=9000.,**kw) if variable=="W" else a
            return carry,(s,c,jnp.clip(a,-1.,1.),jnp.clip(b,-1.,1.),seed)
        step=jax.jit(sample)
        carry=(jnp.zeros((16,18),jnp.float32),jnp.zeros((16,18),jnp.float32),
               jnp.asarray(seed_array(811,7,datetime(2026,10,13,18))))
        actual,truth=[],[]
        error=0.
        for index in range(100):
            expected=(read((16,18)),read((16,18)),read((16,5,18)).transpose(1,0,2),
                      read((16,5,18)).transpose(1,0,2),read((8,),np.int32))
            carry,got=step(carry)
            got=jax.device_get(got)
            for component,(a,b) in enumerate(zip(got[:4],expected[:4])):
                assert np.isfinite(a).all() and np.isfinite(b).all()
                error=max(error,float(np.max(np.abs(a-b))/max(1.e-30,np.max(np.abs(b)))))
                np.testing.assert_allclose(a,b,rtol=8.e-6,atol=64*np.finfo(np.float32).eps*np.max(np.abs(b)),
                                          err_msg=f"{variable}:step{(index+1)*10}:component{component}")
            np.testing.assert_array_equal(got[4],expected[4])
            actual.append(got[2:4]);truth.append(expected[2:4])
        a,b=np.asarray(actual),np.asarray(truth)
        # Analysis uses double precision; model carry and original source are REAL.
        moments=lambda x:dict(mean=float(x.astype(np.float64).mean()),std=float(x.astype(np.float64).std()),
                              max_abs=float(np.abs(x).max()))
        power_a=np.mean(np.abs(np.fft.rfft2(a,axes=(-2,-1)))**2,axis=(0,1,2))
        power_b=np.mean(np.abs(np.fft.rfft2(b,axes=(-2,-1)))**2,axis=(0,1,2))
        np.testing.assert_allclose(power_a,power_b,rtol=1.e-4,atol=1.e-6*power_b.max())
        records.append(dict(variable=variable,steps=1000,samples=100,max_normalized_error=error,
            actual_moments=moments(a),original_moments=moments(b),seed_words_exact=800,
            spectral_power_normalized_max_error=float(np.abs(power_a-power_b).max()/power_b.max())))
        jax.clear_caches()
    assert offset==len(source),(offset,len(source))
    receipt=dict(source="original WRF module_stoch + original FFTPACK5, GNU14 REAL4",records=records,
        backend="cpu",GPU_execution=False,whole_wall_s=time.perf_counter()-start,
        work="Each mode 1000ARsteps dt54 vertical1 nx18ny16nz5, fixed811/nens7/date20261013_18; sample each10; W outputs U/V",
        bounds="fields8e-6+64epsmaxref; spectra1e-4+1e-6maxpower; all5600seedwordsEXACT; no NaN",
        qualification="standalone long patterns, not coupled forecast or performance")
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
