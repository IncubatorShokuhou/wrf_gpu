"""CPU optimized-HLO precision/resource audit of the native REAL families.

No GPU claim: the two GFS DOUBLE islands (94/95) are deliberately excluded.
"""
from pathlib import Path
import functools
import hashlib
import json
import re
import sys
import time
from collections import Counter

import jax
import numpy as np

from gpuwrf.physics.cumulus_ksas14 import cu_ksas14_columns
from gpuwrf.physics.cumulus_nsas96 import cu_nsas96_columns
from gpuwrf.physics.cumulus_kf99 import kf99_column

jax.config.update('jax_enable_x64', True)
assert jax.devices()[0].platform == 'cpu'
root = Path(__file__).resolve().parent
report = {'platform': 'cpu', 'gpu_measured': False, 'schemes': {}}
for option in (14, 96, 99):
    if option == 99:
        with np.load(root/'fixtures/kf99_r4.npz') as data:
            args = [data[k].astype(np.float32) for k in
                    ('T', 'QV', 'P', 'DZ', 'RHO', 'W0AVG', 'U', 'V', 'PI')]
        fn = jax.vmap(lambda *a: kf99_column(*a[:8], np.float32(54.),
                      np.float32(9000.), pi_phy=a[8], mode='r4'))
    else:
        with np.load(root/f'fixtures/native{option}_inputs.npz') as raw:
            data = {k: v.astype(np.float32) for k, v in raw.items() if v.dtype.kind in "fiu"}
        keys = ('T', 'QV', 'QC', 'QI', 'P', 'PI', 'RHO', 'DZ', 'U', 'V', 'W')
        opts = dict(dt=18., stepcu=1, dx=9000., mp_physics=8, literals='r4')
        if option == 14:
            fn = functools.partial(cu_ksas14_columns, **opts)
            keys += ('XLAND', 'HPBL', 'HPBL_HOLD', 'ZNU')
        else:
            fn = functools.partial(cu_nsas96_columns, row_width=455, pgcon=.55,
                                   dx_factor_nsas=0, **opts)
            keys += ('P8W', 'XLAND', 'HPBL', 'HFX', 'QFX')
        args = [data[k] for k in keys]
    start = time.monotonic()
    exe = jax.jit(fn).lower(*args).compile()
    hlo = exe.as_text()
    memory = exe.memory_analysis()
    f64_instructions = [line for line in hlo.splitlines()
                        if re.search(r'=\s*(?:\([^\n]*\b)?f64\[', line)]
    row = dict(compile_s=time.monotonic()-start,
               f64_instruction_count=len(f64_instructions),
               f64_opcodes=dict(Counter(line.split(' = ',1)[1].split(' ',1)[1].split('(',1)[0]
                                        for line in f64_instructions if '= f64[' in line)),
               optimized_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
               argument_bytes=memory.argument_size_in_bytes,
               output_bytes=memory.output_size_in_bytes,
               temp_bytes=memory.temp_size_in_bytes)
    report['schemes'][str(option)] = row
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2)+'\n')
    print(option, row, flush=True)
    if len(sys.argv)>2:
        actual={k:np.asarray(v) for k,v in exe(*args).items()}
        output=Path(sys.argv[2])/f'{option}.npz'
        if output.exists():
            with np.load(output) as previous:
                assert set(previous)==set(actual)
                assert all(previous[k].tobytes()==v.tobytes() for k,v in actual.items()), option
            row['baseline_output_byte_exact']=True
        else:
            np.savez_compressed(output,**actual)
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2)+'\n')
    if len(sys.argv)<4 or sys.argv[3]!='diagnostic':
        assert not f64_instructions, f64_instructions[:5]
