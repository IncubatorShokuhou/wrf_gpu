"""Scalar compiler libm at the literal-reference boundary (not product math)."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np

class CompilerLibm:
    def __init__(self):
        self.library=ctypes.CDLL('libm.so.6')

    def __getattr__(self,name):
        return getattr(np,name)

    def call(self,name,*values):
        values=np.broadcast_arrays(*values)
        dtype=np.result_type(*values)
        scalar=ctypes.c_float if dtype==np.dtype('float32') else ctypes.c_double
        fn=getattr(self.library,name+('f' if scalar==ctypes.c_float else ''))
        fn.argtypes=[scalar]*len(values);fn.restype=scalar
        result=np.fromiter((fn(*row) for row in zip(*(v.flat for v in values))),dtype=dtype,count=values[0].size)
        return result.reshape(values[0].shape)

    def power(self,left,right):return self.call('pow',left,right)
    def exp(self,value):return self.call('exp',value)
    def log(self,value):return self.call('log',value)
    def tanh(self,value):return self.call('tanh',value)
    def log10(self,value):return self.call('log10',value)

backend=CompilerLibm()


def aggregation_backend(directory):
    """Literal boundary of the measured original gfortran vector log call."""
    output=Path(directory)/'original_libmvec.so'
    subprocess.run(['gcc','-O2','-shared','-fPIC',str(Path(__file__).with_name('libmvec_pair.c')),
                    '-lm','-o',str(output)],check=True,timeout=30)
    library=ctypes.CDLL(str(output));fn=library.original_pair_log
    pointer=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ctypes.c_size_t,pointer,pointer,pointer,pointer];fn.restype=None
    selected=CompilerLibm()
    def pair_log(left,right):
        if np.result_type(left,right)==np.dtype('float32'):
            return selected.log(left),selected.log(right)
        left,right=(np.ascontiguousarray(a) for a in np.broadcast_arrays(left,right))
        a=np.empty_like(left);b=np.empty_like(right)
        fn(left.size,*(v.ctypes.data_as(pointer) for v in (left,right,a,b)))
        return a,b
    selected.original_pair_log=pair_log
    selected._pair_library=library
    return selected
