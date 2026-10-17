"""CPU-only: run pip Triton's make_ttir + make_ttgir pass lists (cuda:120, the
sm>=100 branch of triton/backends/nvidia/compiler.py) ONE PASS AT A TIME on a
saved native TTIR, verifying after each pass; record op counts per pass and the
first failing pass. Optionally save IR after selected passes. Diagnostic only.
usage: pass_bisect.py TTIR OUTDIR [--save P1,P2] [--skip P1,P2] [--stages N]"""
import argparse,json,re,time
from pathlib import Path
from triton._C.libtriton import ir,passes,nvidia
from triton.backends.nvidia.compiler import CUDABackend

ap=argparse.ArgumentParser();ap.add_argument('ttir',type=Path);ap.add_argument('out',type=Path)
ap.add_argument('--save',default='');ap.add_argument('--skip',default='');ap.add_argument('--stages',type=int,default=3)
ap.add_argument('--stop-after',default='')
ap.add_argument('--track',default='',help='comma list of FILE:LINE source sites; counts ops located there after each pass')
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
save=set(filter(None,a.save.split(',')));skip=set(filter(None,a.skip.split(',')))
N=a.stages
import types
from triton.backends.nvidia.compiler import get_ptx_version_from_options
try:PTX=get_ptx_version_from_options(types.SimpleNamespace(ptx_version=None),120)
except Exception as e:print('ptx version',repr(e));PTX=88
PASSES=[
 ('inliner',lambda pm:passes.common.add_inliner(pm)),
 ('rewrite_tensor_pointer',lambda pm:passes.ttir.add_rewrite_tensor_pointer(pm)),
 ('canonicalizer0',lambda pm:passes.common.add_canonicalizer(pm)),
 ('combine',lambda pm:passes.ttir.add_combine(pm)),
 ('reorder_broadcast',lambda pm:passes.ttir.add_reorder_broadcast(pm)),
 ('cse0',lambda pm:passes.common.add_cse(pm)),
 ('symbol_dce0',lambda pm:passes.common.add_symbol_dce(pm)),
 ('loop_unroll',lambda pm:passes.ttir.add_loop_unroll(pm)),
 ('convert_to_ttgpuir',lambda pm:passes.ttir.add_convert_to_ttgpuir(pm,'cuda:120',4,32,1)),
 ('coalesce',lambda pm:passes.ttgpuir.add_coalesce(pm)),
 ('f32_dot_tc',lambda pm:passes.ttgpuir.add_f32_dot_tc(pm,True)),
 ('plan_cta',lambda pm:nvidia.passes.ttnvgpuir.add_plan_cta(pm)),
 ('remove_layout_conversions1',lambda pm:passes.ttgpuir.add_remove_layout_conversions(pm)),
 ('optimize_thread_locality',lambda pm:passes.ttgpuir.add_optimize_thread_locality(pm)),
 ('accelerate_matmul',lambda pm:passes.ttgpuir.add_accelerate_matmul(pm)),
 ('remove_layout_conversions2',lambda pm:passes.ttgpuir.add_remove_layout_conversions(pm)),
 ('optimize_dot_operands1',lambda pm:passes.ttgpuir.add_optimize_dot_operands(pm,True)),
 ('optimize_descriptor_encoding',lambda pm:nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)),
 ('loop_aware_cse1',lambda pm:passes.ttir.add_loop_aware_cse(pm)),
 ('fuse_nested_loops',lambda pm:passes.ttgpuir.add_fuse_nested_loops(pm)),
 ('canonicalizer1',lambda pm:passes.common.add_canonicalizer(pm)),
 ('triton_licm',lambda pm:passes.ttir.add_triton_licm(pm)),
 ('optimize_accumulator_init',lambda pm:passes.ttgpuir.add_optimize_accumulator_init(pm)),
 ('hoist_tmem_alloc0',lambda pm:passes.ttgpuir.add_hoist_tmem_alloc(pm,False)),
 ('promote_lhs_to_tmem',lambda pm:nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)),
 ('assign_latencies',lambda pm:passes.ttgpuir.add_assign_latencies(pm,N)),
 ('schedule_loops',lambda pm:passes.ttgpuir.add_schedule_loops(pm)),
 ('warp_specialize',lambda pm:passes.ttgpuir.add_warp_specialize(pm,N)),
 ('pipeline',lambda pm:passes.ttgpuir.add_pipeline(pm,N,False)),
 ('optimize_partition_warps',lambda pm:passes.ttgpuir.add_optimize_partition_warps(pm)),
 ('combine_tensor_select_and_if',lambda pm:passes.ttgpuir.add_combine_tensor_select_and_if(pm)),
 ('hoist_tmem_alloc1',lambda pm:passes.ttgpuir.add_hoist_tmem_alloc(pm,True)),
 ('remove_tmem_tokens',lambda pm:nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)),
 ('canonicalizer2',lambda pm:passes.common.add_canonicalizer(pm)),
 ('loop_aware_cse2',lambda pm:passes.ttir.add_loop_aware_cse(pm)),
 ('prefetch',lambda pm:passes.ttgpuir.add_prefetch(pm)),
 ('optimize_dot_operands2',lambda pm:passes.ttgpuir.add_optimize_dot_operands(pm,True)),
 ('coalesce_async_copy',lambda pm:passes.ttgpuir.add_coalesce_async_copy(pm)),
 ('optimize_tmem_layouts',lambda pm:nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)),
 ('tma_lowering',lambda pm:nvidia.passes.ttnvgpuir.add_tma_lowering(pm)),
 ('remove_layout_conversions3',lambda pm:passes.ttgpuir.add_remove_layout_conversions(pm)),
 ('interleave_tmem',lambda pm:nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)),
 ('reduce_data_duplication',lambda pm:passes.ttgpuir.add_reduce_data_duplication(pm)),
 ('reorder_instructions',lambda pm:passes.ttgpuir.add_reorder_instructions(pm)),
 ('loop_aware_cse3',lambda pm:passes.ttir.add_loop_aware_cse(pm)),
 ('symbol_dce1',lambda pm:passes.common.add_symbol_dce(pm)),
 ('fence_insertion',lambda pm:nvidia.passes.ttnvgpuir.add_fence_insertion(pm,120)),
 ('lower_mma',lambda pm:nvidia.passes.ttnvgpuir.add_lower_mma(pm)),
 ('sccp',lambda pm:passes.common.add_sccp(pm)),
 ('cse1',lambda pm:passes.common.add_cse(pm)),
 ('canonicalizer3',lambda pm:passes.common.add_canonicalizer(pm)),
 ('llir_combine_tensor_select_and_if',lambda pm:passes.ttgpuir.add_combine_tensor_select_and_if(pm)),
 ('allocate_warp_groups',lambda pm:passes.ttgpuir.add_allocate_warp_groups(pm)),
 ('scf_to_cf',lambda pm:passes.convert.add_scf_to_cf(pm)),
 ('gluon_inliner',lambda pm:passes.gluon.add_inliner(pm)),
 ('allocate_shared_memory_nv',lambda pm:nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm,120,PTX)),
 ('allocate_tensor_memory',lambda pm:nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)),
 ('check_matmul_two_cta',lambda pm:nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)),
 ('allocate_global_scratch_memory',lambda pm:passes.ttgpuir.add_allocate_global_scratch_memory(pm)),
 ('proxy_fence_insertion',lambda pm:nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm,120)),
 ('to_llvmir',lambda pm:nvidia.passes.ttgpuir.add_to_llvmir(pm,120,PTX)),
 ('llir_canonicalizer0',lambda pm:passes.common.add_canonicalizer(pm)),
 ('llir_cse0',lambda pm:passes.common.add_cse(pm)),
 ('nvgpu_to_llvm',lambda pm:nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)),
 ('warp_specialize_to_llvm',lambda pm:nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)),
 ('llir_canonicalizer1',lambda pm:passes.common.add_canonicalizer(pm)),
 ('llir_cse1',lambda pm:passes.common.add_cse(pm)),
 ('llir_symbol_dce',lambda pm:passes.common.add_symbol_dce(pm)),
 ('nvvm_to_llvm',lambda pm:passes.convert.add_nvvm_to_llvm(pm)),
]
ctx=ir.context();ir.load_dialects(ctx);nvidia.load_dialects(ctx)
mod=ir.parse_mlir_module(str(a.ttir),ctx);mod.context=ctx
count=lambda t:{op:len(re.findall(r'\b'+re.escape(op)+r'\b',t)) for op in ('scf.if','scf.while','scf.for','arith.select','tt.reduce','cf.cond_br','llvm.cond_br')}
report={'ttir':str(a.ttir),'num_stages':N,'passes':[]}
sites=[t for t in a.track.split(',') if t]
def track(text):
    # Resolve '#locN = loc(...)' aliases transitively; count op lines whose loc
    # alias (or inline loc) contains each tracked "file":line site.
    alias={};body=[]
    for line in text.splitlines():
        m=re.match(r'^(#loc\d*) = (loc\(.*)$',line)
        if m:alias[m.group(1)]=m.group(2)
        else:body.append(line)
    memo={}
    def expand(name,depth=0):
        if name in memo:return memo[name]
        s=alias.get(name,'');memo[name]=s
        if depth<50:
            for ref in set(re.findall(r'#loc\d*\b',s)):s+=expand(ref,depth+1)
        memo[name]=s;return s
    out={}
    for site in sites:
        f,l=site.rsplit(':',1);needle=f'{f}":{l}:'
        hit={n for n in alias if needle in expand(n)}
        c=0
        for line in body:
            m=re.search(r'loc\((#loc\d*)\)\s*$',line)
            if m and m.group(1) in hit:c+=1
            elif needle in line:c+=1
        out[site]=c
    return out
def dump():(a.out/'bisect.json').write_text(json.dumps(report,indent=2)+'\n')
text=mod.str_nodebug();report['input']=count(text)
if sites:report['input_track']=track(mod.str())
dump()
for name,add in PASSES:
    if name in skip:report['passes'].append({'pass':name,'skipped':True});continue
    pm=ir.pass_manager(ctx);add(pm);t=time.monotonic()
    try:pm.run(mod,name);ok=True;err=None
    except Exception as e:ok=False;err=repr(e)[-1500:]
    row={'pass':name,'ok':ok,'s':round(time.monotonic()-t,3)}
    if ok:
        text=mod.str_nodebug();row.update(count(text))
        if sites:row['track']=track(mod.str())
        if name in save:(a.out/f'after_{name}.mlir').write_text(text)
    else:row['error']=err
    report['passes'].append(row);dump();print(json.dumps(row)[:600],flush=True)
    if not ok or name==a.stop_after:break
print('DONE',flush=True)
