"""Independent CPU review checks; never edits the installed compiler or candidate."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
from gpuwrf.kernels import _triton_kernel_call as route
from triton._C import libtriton

original_file = libtriton.__file__
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
    fake = Path(folder) / 'libtriton.so'
    fake.write_bytes(b'compiler-A')
    status = fake.stat()
    libtriton.__file__ = str(fake)
    first = route.stock_compiler_identity()
    fake.write_bytes(b'compiler-B')
    os.utime(fake, ns=(status.st_atime_ns, status.st_mtime_ns))
    second = route.stock_compiler_identity()
    libtriton.__file__ = original_file
    collision = dict(different_content=True, same_identity=first == second,
                     identity=first, before_sha=hashlib.sha256(b'compiler-A').hexdigest(),
                     after_sha=hashlib.sha256(b'compiler-B').hexdigest())
    print(json.dumps({'compiler_content_collision': collision}), flush=True)
    assert first == second, 'review reproduction no longer applicable'

test = runpy.run_path(str(root/'tests/v040/test_cammg_kernel_call_route.py'))
test['test_cammg_column_selects_kernel_call_route']()
from gpuwrf.kernels import phys_cammg
phys_cammg.KernelCallCompilerParams = route.triton_core.CompilerParams
try:
    test['test_cammg_column_selects_kernel_call_route']()
except AssertionError:
    print(json.dumps({'mutant': 'replace stock route with bundled route',
                      'result': 'KILLED', 'exception': 'AssertionError', 'setup_errors': 0}), flush=True)
else:
    raise AssertionError('route deletion mutant survived')
