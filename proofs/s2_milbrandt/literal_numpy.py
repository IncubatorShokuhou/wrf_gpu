"""Literal noncontracting REAL4 diagnostic of the same generated source, using scalar libm.

This is an independent CPU execution of the port's equations, not the native
GPU executable and not a replacement for the pristine Fortran reference.
"""
from pathlib import Path
from types import SimpleNamespace
import ctypes
import ctypes.util
import json
import numpy as np
from gpuwrf.physics.milbrandt2mom import column,local,math,constants


def install():
    lib=ctypes.CDLL(ctypes.util.find_library('m'))
    for name,nargs in (('exp',1),('log',1),('log10',1),('pow',2)):
        for suffix,kind in (('f',ctypes.c_float),('',ctypes.c_double)):
            fn=getattr(lib,name+suffix);fn.argtypes=[kind]*nargs;fn.restype=kind
    def op(name,a,b):
        return {'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[name]()
    def trans(name,*args):
        arrays=np.broadcast_arrays(*args);dt=np.result_type(*arrays)
        fn=getattr(lib,name+('f' if dt==np.float32 else ''))
        return np.asarray([fn(*map(float,vals)) for vals in zip(*(a.flat for a in arrays))],dtype=dt).reshape(arrays[0].shape)
    def power(a,b):
        if isinstance(b,int):
            if b==0:return np.ones_like(a)
            if b==1:return a
            if b<0:return np.ones_like(a)/power(a,-b)
            h=power(a,b//2);s=h*h
            return s*a if b%2 else s
        return trans('pow',a,b)
    def fori(lo,hi,body,carry):
        for i in range(int(lo),int(hi)):carry=body(i,carry)
        return carry
    for mod in (column,local,math,constants):
        mod.jnp=np
        mod._op=op;mod._pow=power
        mod._exp=lambda x:trans('exp',x)
        mod._log=lambda x:trans('log',x)
        mod._log10=lambda x:trans('log10',x)
    constants.constants.cache_clear()
    column.jax=SimpleNamespace(lax=SimpleNamespace(fori_loop=fori,cond=lambda pred,yes,no,x:yes(x) if pred else no(x)))
    column.cuda_native=lambda:False


def main():
    import sys
    install()
    with np.load(sys.argv[1]) as f:data=dict(f)
    a=data['inputs'];outputs=[]
    with np.errstate(all='ignore'):
        for i in range(a.shape[0]):
            f,r,d,z,zc=column.column(a[i,:,:14].T,a[i,:,15],data['ps'][i],a[i,:,14],data['dt'])
            outputs.append((np.stack(f),np.stack(r),np.stack(d),z,zc))
    result=[np.stack([o[j] for o in outputs]) for j in range(5)]
    report={}
    for name,got in zip(('fields','rates','diam','zet','zec'),result):
        ref=data['ref_'+name]
        report[name]={'byte_equal':bool(got.tobytes()==ref.tobytes()),'different_values':int(np.count_nonzero(got!=ref)),
                      'max_abs':float(np.max(np.abs(got-ref)))}
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
    np.savez(Path(sys.argv[2]).with_suffix('.npz'),**dict(zip(('fields','rates','diam','zet','zec'),result)))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
