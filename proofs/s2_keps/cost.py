"""CPU-only full-column cost/memory; no GPU or forecast performance claim."""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

from full_compare import inputs
from gpuwrf.physics.pbl_keps import keps_column
from gpuwrf.physics.pbl_eeps import eeps_column

assert jax.devices()[0].platform=='cpu'
fixture=json.loads(Path(sys.argv[1]).read_text())
out={'backend':'cpu','scope':'full columns only; GPU launch/traffic/compile costs unknown','schemes':{}}
for scheme,fn in ((16,eeps_column),(17,keps_column)):
    rows=[r for r in fixture['records'] if r['scheme']==scheme and r['n']==44][::5]
    batch={k:np.stack([inputs(r)[k] for r in rows]) for k in inputs(rows[0])}
    start=time.perf_counter();lowered=jax.jit(jax.vmap(lambda x:fn(**x))).lower(batch);lower_s=time.perf_counter()-start
    assert 'f64[' not in lowered.compiler_ir('hlo').as_hlo_text()
    start=time.perf_counter();compiled=lowered.compile();compile_s=time.perf_counter()-start
    stats=compiled.memory_analysis();jax.block_until_ready(compiled(batch));times=[]
    for _ in range(20):
        start=time.perf_counter();jax.block_until_ready(compiled(batch));times.append(time.perf_counter()-start)
    out['schemes'][scheme]=dict(columns=len(rows),levels=44,lower_s=lower_s,compile_s=compile_s,
        warm_median_s=float(np.median(times)),argument_bytes=stats.argument_size_in_bytes,
        output_bytes=stats.output_size_in_bytes,temporary_bytes=stats.temp_size_in_bytes,alias_bytes=stats.alias_size_in_bytes,
        hlo_no_fp64=True)
    Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
