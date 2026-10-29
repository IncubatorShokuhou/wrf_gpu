"""Seal the CPU-only attribution, changed-program resource gate and OFF proof."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v45-seam')
load = lambda path: json.loads(path.read_text())
receipts = {name: load(ROOT / name / 'receipt.json')
            for name in ('OFF01', 'ON01', 'FENCE01', 'FENCE_OFF01')}
producer = load(Path('<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams/NATIVE05/receipt.json'))
base, on, fix, off = (receipts[n] for n in receipts)
assert [r['phase'] for r in receipts.values()] == ['done', 'done', 'done', 'lower_done']
same_work = ('input_sha256', 'static_kwargs', 'case', 'domain', 'target_sha256', 'environment',
             'harness_sha256', 'physical_backend', 'lowering_platform', 'gpu_executed')
assert all(r[k] == base[k] for r in (*receipts.values(), producer) for k in same_work)
assert all(r['physical_backend'] == 'cpu' and not r['gpu_executed'] and
           r['traced_pallas_calls'] == 163 and r['pallas']['interpreted'] == 0 and
           r['census_errors'] == [] for r in receipts.values())
assert base['selected'] == off['selected'] == []
assert on['selected'] == fix['selected'] == producer['selected']
identity_keys = ('raw_hlo_full_sha256', 'raw_hlo_sha256', 'stablehlo_sha256', 'pallas')
assert all(base[k] == off[k] for k in identity_keys)
assert all(producer[k] == on[k] for k in identity_keys)
native_key = lambda r: sorted((k['name'], k['ttir_sha16'], k['ttir_loc_sha16'],
                               k['num_warps'], k['num_stages'], k['calls']) for k in r['native_bodies'])
assert native_key(base) == native_key(off)
assert native_key(on) == native_key(fix)
assert base['native_gate'] == fix['native_gate'] == 'PASS' and on['native_gate'] == 'FAIL'
assert on['compiled_memory'] == fix['compiled_memory']

summaries = {}
for name, rec in receipts.items():
    keys = ('source_commit', 'selected', 'static_kwargs', 'raw_hlo_full_sha256',
            'full_constant_raw_proto_sha256', 'stablehlo_sha256', 'traced_pallas_calls',
            'census_errors', 'native_gate', 'hard_stack_gate', 'max_stack_B', 'kernels',
            'n_fusions', 'transpose_fusions', 'compiled_memory', 'compile_s', 'wall_s', 'peak_rss_KiB')
    summary = {k: rec[k] for k in keys if k in rec}
    summary['receipt'] = str(ROOT / name / 'receipt.json')
    summary['receipt_sha256'] = hashlib.sha256((ROOT / name / 'receipt.json').read_bytes()).hexdigest()
    if 'native_gate' in rec:
        summary['largest_kInput_HLO'] = max(n for n, kind, _ in rec['top_transpose'] if kind == 'kInput')
        summary['kInput_above500'] = rec['kinput_transpose_over500']
        gate = load(ROOT / name / 'e41/wave_gate.json')
        summary['nonleaf_over1000'] = [x for entry in gate['entries'] for elf in entry['elfs'] for x in elf['nonleaf_over1k']]
    summaries[name] = summary
result = dict(
    verdict='CPU_PRESCREEN_PASS; actual GPU and whole-root qualification pending',
    scope='Same PROD d01 parent probe as NATIVE05, fresh CPU init and product force-down, native CUDA lowering, sm120 CompileOnly autotune0; no GPU execution.',
    attribution='Introduced in candidate ON: OFF max406/STACK32, ON max574/STACK240. The 574-node multi-output body joins a 203-node U boundary target and 379-node U momentum tendency with nine shared nodes plus the tuple root. No Noah operation lies in this body. Do not transfer the older fused_d01 component-root664 exception or timing to it.',
    causal_limit='The producer change touches inverse-SMCMAX/dry-conductivity/PSISAT consumers in Noah. Their combined graph change relocates the downstream fusion boundary; no single Noah edit is claimed uniquely causal. The exact four-line boundary-output intervention restores the gate, and deleting it recovers the measured failing program.',
    candidate_composition=fix['source_commit'], main_patch='src/gpuwrf/runtime/operational_mode.py: output-only U boundary barrier, native REAL + SMCMAX only',
    limits=dict(stack_bytes=256, nonleaf_SASS=1000, kInput_HLO=500),
    immutable_source='git archive, all tracked blobs checked against pinned git object IDs before each arm; same import alias, atomic source replacement only between completed arms',
    original_harness_sha256=on['harness_sha256'], wrapper_sha256=hashlib.sha256((ROOT / 'run_prescreen.py').read_bytes()).hexdigest(),
    target_sha256=on['target_sha256'], input_sha256=on['input_sha256'],
    versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib')},
    same_work_options_verified=True, producer_reproduction_full_constant_HLO_identical=True,
    off_full_constant_HLO_and_all_native_bodies_identical=True,
    native_bodies_unchanged_by_fence=True, arms=summaries,
    physics_prefix_reuse=load(ROOT / 'PREFIX_REUSE.json'),
    compiler_comparator_checks=load(ROOT / 'COMPARATOR_CHECK.json'),
    performance=dict(static_kernel_delta=fix['kernels']-on['kernels'],
        static_kernel_percent=100*(fix['kernels']/on['kernels']-1),
        compiled_memory_delta_bytes={k:fix['compiled_memory'][k]-v for k,v in on['compiled_memory'].items()},
        compile_wall_scope='Measured on shared CPU cores; not a controlled compiler speed comparison.',
        dynamic_launches='UNKNOWN', GPU_wall='UNKNOWN', GPU_transfers='UNKNOWN', GPU_peak_VRAM='UNKNOWN'),
    next_gate='Manager/producer: final composed candidate child/root numerical gates, actual GPU E41 and matched timing/transfer/memory measurements. No release, admission or WRF fidelity claim from these CPU prescreens.',
)
if (ROOT / 'COLLECTION.json').exists():
    result['archive_collection'] = load(ROOT / 'COLLECTION.json')
if (ROOT / 'CLEANUP_FINAL.json').exists():
    result['cleanup'] = load(ROOT / 'CLEANUP_FINAL.json')
(ROOT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('verdict', 'off_full_constant_HLO_and_all_native_bodies_identical', 'performance')}, indent=2))
