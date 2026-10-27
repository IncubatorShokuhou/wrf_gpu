"""Real-model CPU replay through the identical warm child command and AOT loader."""
import os
import json

from phase_journal import write_receipt
from product_schedule import product_schedule


def run(args, source, receipt, journal):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from cold_prepare import prepare
    from family_work import cpu_operands, defaults
    from root_gpu import differences
    assert jax.devices()[0].platform == 'cpu' and os.environ['JAX_PLATFORMS'] == 'cpu'
    outputs = {}
    for variant in ('off', 'on'):
        fn, operands, key, work = cpu_operands(source, args.family, variant)
        name = 'launch_smoke_' + args.family + '_' + variant
        call, status = prepare(name, fn, operands, key, journal)
        assert status['loaded'], 'cold preparations must precede warm replay'
        with journal.phase('warm_mini_' + variant):
            outputs[variant] = jax.block_until_ready(call(*operands))
        if variant == 'on':
            values = defaults(args.family)
            if args.family == 'diffusion':
                values['c_s'] = .5
            else:
                values = {name: value * 1.1 for name,value in values.items()}
            with journal.phase('changed_mini_same_executable'):
                changed = jax.block_until_ready(call(operands[0],{k:jnp.float32(v) for k,v in values.items()}))
            assert any(np.asarray(a).tobytes() != np.asarray(b).tobytes()
                       for a,b in zip(jax.tree.leaves(changed),jax.tree.leaves(outputs['on']),strict=True))
    host = lambda tree:{str(i):np.asarray(a) for i,a in enumerate(jax.tree.leaves(tree))}
    delta = differences(host(outputs['off']),host(outputs['on']))
    assert not delta, delta
    _, schedule = product_schedule(args.input_dir,('d01','d02'),{'d01':54.,'d02':18.})
    receipt.update(status='PASS',platform='cpu',gpu_executed=False,cpu_smoke=True,
                   model_work=work,default_byte_exact=True,changed_active=True,
                   same_loaded_executable=True,product_schedule=schedule,
                   output_leaves=len(jax.tree.leaves(outputs['on'])),
                   scope='CPU real-model launch/AOT replay; no GPU, root-performance or fidelity claim')
    write_receipt(args.out / ('measure_profile_cpu_smoke.json' if args.profile else 'measure_cpu_smoke.json'),receipt)
