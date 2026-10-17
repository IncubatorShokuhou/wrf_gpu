"""Time the normal TTIR + TTGIR Triton stages of one TTIR file (SM120, CPU only).
usage: ttgir_time.py FILE.ttir [num_warps]   -> prints 'TTGIR <file> <ttir_s> <ttgir_s>'"""
import sys
import time
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import IRSource, make_backend
from triton._C.libtriton import ir as triton_ir

path, warps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
backend = make_backend(GPUTarget('cuda', 120, 32))
context = triton_ir.context()
source = IRSource(path, context, backend)
source.module.context = context
options = backend.parse_options({'num_warps': warps, 'enable_fp_fusion': True})
t = time.monotonic()
mod = backend.make_ttir(source.module, {}, options, 120)
t1 = time.monotonic()
backend.make_ttgir(mod, {}, options, 120)
print('TTGIR', path, round(t1 - t, 2), round(time.monotonic() - t1, 2), flush=True)
