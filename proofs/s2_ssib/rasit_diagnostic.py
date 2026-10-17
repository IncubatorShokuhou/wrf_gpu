#!/usr/bin/env python3
"""CPU-only original-WRF RASIT5 operand replay; no forecast or speed claim.

Callbacks exist ONLY in this diagnostic process. Baseline vs instrumented full
column bytes are checked, so instrumentation cannot silently change the witness.
"""
from pathlib import Path
import argparse,ctypes,inspect,json,subprocess,sys,time,hashlib
import numpy as np
from audit_oracle import STUBS
from literal import load
from literal_gate import operands
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v025'),str(ROOT/'tests')]
import cpu_guard
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from gpuwrf.physics import _ssib_generated as model
from gpuwrf.physics import lsm_ssib
import test_s2_ssib as tests


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);build=args.output/'build';build.mkdir()
    source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_ssib.F')
    (build/source.name).write_bytes(source.read_bytes());(build/'stubs.f90').write_text(STUBS)
    (build/'wrapper.f90').write_text('''subroutine rasit_oracle(a) bind(C,name='rasit_oracle')
use iso_c_binding, only:c_float
use module_sf_ssib, only:RASIT5
real(c_float),intent(inout)::a(20)
call RASIT5(a(1),a(2),a(3),a(4),a(5),a(6),a(7),a(8),a(9),a(10), &
            a(11),a(12),a(13),a(14),a(15),a(16),a(17),a(18),a(19),a(20))
end subroutine
''')
    flags=['-O2','-ffree-form','-ffree-line-length-none','-fallow-argument-mismatch',
           '-fPIC','-shared','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    run=subprocess.run(['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran',*flags,source.name,
        'stubs.f90','wrapper.f90','-o','rasit.so'],cwd=build,capture_output=True,text=True,timeout=180)
    (args.output/'compile.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,'oracle build failed'
    lib=ctypes.CDLL(str(build/'rasit.so'));native=lib.rasit_oracle
    native.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,shape=(20,),flags='C_CONTIGUOUS')];native.restype=None
    literal=load(ROOT/'src/gpuwrf/physics/_ssib_generated.py');calls=[]
    original_literal=literal._rasit5
    def capture_literal(*xs):
        calls.append(np.asarray(xs,np.float32));return original_literal(*xs)
    literal._rasit5=capture_literal
    with np.load(ROOT/'proofs/s2_ssib/native_reference.npz',allow_pickle=False) as f:
        names=f['ordinary_1_names'].tolist();frame=dict(zip(names,f['ordinary_1'][0,:,4]))
    names=tuple(inspect.signature(literal._ssib).parameters)
    values=operands(names,frame,False)
    with np.errstate(all='ignore'):literal_result=literal._ssib(*(values[n] for n in names))
    last_literal=calls[-1]
    land,forcing=tests._inputs(frame)
    fn=jax.jit(lambda l,f:lsm_ssib.ssib_column(land=l,forcing=f))
    begin=time.monotonic();baseline=jax.block_until_ready(fn(land,forcing));baseline_s=time.monotonic()-begin
    compiled_calls=[];original=model._rasit5
    def record(*xs):compiled_calls.append(np.asarray(xs,np.float32))
    def instrumented(*xs):
        jax.debug.callback(record,*xs,ordered=True)
        return original(*xs)
    model._rasit5=instrumented;jax.clear_caches()
    fn2=jax.jit(lambda l,f:lsm_ssib.ssib_column(land=l,forcing=f))
    begin=time.monotonic();instrumented_result=jax.block_until_ready(fn2(land,forcing));instrumented_s=time.monotonic()-begin
    model._rasit5=original
    byte_changed=[name for name,a,b in zip(lsm_ssib.STATE_FIELDS,baseline,instrumented_result)
                  if np.asarray(a).tobytes()!=np.asarray(b).tobytes()]
    last_compiled=compiled_calls[-1]
    native_literal=last_literal.copy();native(native_literal)
    native_compiled=last_compiled.copy();native(native_compiled)
    rn=tuple(inspect.signature(original).parameters)
    differences=[{'field':name,'literal':float(a),'compiled':float(b),
                  'ulp':int(abs(np.int64(a.view(np.int32))-np.int64(b.view(np.int32))))}
                 for name,a,b in zip(rn,last_literal,last_compiled) if a.tobytes()!=b.tobytes()]
    sensitivity=[]
    for name in ('bps','trib','tmm'):
        index=rn.index(name)
        for direction in (-1,1):
            for ulps in (1,2,4):
                x=last_literal.copy()
                for _ in range(ulps):x[index]=np.nextafter(x[index],np.float32(-np.inf if direction<0 else np.inf))
                input_value=float(x[index]);native(x)
                sensitivity.append({'field':name,'direction':direction,'ulps':ulps,'input':input_value,'rib':float(x[16]),'delta_rib':float(x[16]-native_literal[16])})
    report={'kind':'CPU-only-original-WRF-RASIT5-replay','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'flags':flags,'backend':str(jax.devices()),'case':'ordinary1column5firststep','cache':'NONE',
        'baseline_compile_execute_s':baseline_s,'instrumented_compile_execute_s':instrumented_s,
        'instrumentation_changed_fields':byte_changed,'literal_calls':len(calls),'compiled_calls':len(compiled_calls),
        'baseline_rib':float(baseline.rib),'instrumented_rib':float(instrumented_result.rib),
        'literal_rib':float(dict(zip(names,literal_result))['rib']),
        'native_rib_on_literal_inputs':float(native_literal[16]),'native_rib_on_compiled_inputs':float(native_compiled[16]),
        'input_differences':differences,'sensitivity':sensitivity}
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'operands.npz',literal=last_literal,compiled=last_compiled,
                        native_literal=native_literal,native_compiled=native_compiled)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
