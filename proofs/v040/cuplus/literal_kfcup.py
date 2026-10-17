"""Execute candidate CuP equations with literal REAL32/libm arithmetic.

This is a qualification backend, not a production fallback or an oracle.
The independent oracle remains the separately compiled pristine WRF caller.
Only the candidate source's arithmetic/control backend changes: every binary
REAL operation rounds to REAL32, and scalar intrinsics use the host libm ABI
used by gfortran. No reference outputs enter the equation executor.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import ctypes.util
import hashlib
import json
import operator
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import time

import numpy as np

from compare_kfcup_parcel import parse


_LIBM=ctypes.CDLL(ctypes.util.find_library('m'))
for _name in ('expf','logf','log10f','sqrtf','powf'):
    _fn=getattr(_LIBM,_name)
    _fn.restype=ctypes.c_float
    _fn.argtypes=[ctypes.c_float]*(2 if _name=='powf' else 1)
for _name in ('exp','log','log10','sqrt','pow'):
    _fn=getattr(_LIBM,_name)
    _fn.restype=ctypes.c_double
    _fn.argtypes=[ctypes.c_double]*(2 if _name=='pow' else 1)


def intrinsic(name,*values):
    dtype=np.float32 if name.endswith('f') else np.float64
    values=np.broadcast_arrays(*(np.asarray(x,dtype) for x in values))
    fn=getattr(_LIBM,name)
    result=np.fromiter((fn(*(float(a.flat[k]) for a in values)) for k in range(values[0].size)),
                       dtype=dtype,count=values[0].size).reshape(values[0].shape)
    return result[()] if result.ndim==0 else result


_OPS={'Add':operator.add,'Sub':operator.sub,'Mult':operator.mul,'Div':operator.truediv,
      'FloorDiv':operator.floordiv,'Mod':operator.mod,'BitAnd':operator.and_,
      'BitOr':operator.or_,'BitXor':operator.xor,'LShift':operator.lshift,'RShift':operator.rshift}


def is_double(value):
    # Module Python literals are weak WRF REAL constants. Only explicit
    # NumPy DOUBLE scalars/arrays (or the DOUBLE AST) promote arithmetic.
    return isinstance(value,(np.ndarray,np.generic)) and np.asarray(value).dtype==np.float64


def binary(kind,a,b):
    floating=any(np.asarray(x).dtype.kind=='f' for x in (a,b))
    dtype=np.float64 if any(is_double(x) for x in (a,b)) else np.float32
    if kind=='Pow':
        if isinstance(b,(int,np.integer)) and b in (2,3,4):
            a=np.asarray(a,dtype)
            aa=a*a
            return aa if b==2 else (aa*a if b==3 else aa*aa)
        return intrinsic('pow' if dtype==np.float64 else 'powf',a,b)
    if floating:
        a,b=np.asarray(a,dtype),np.asarray(b,dtype)
    return _OPS[kind](a,b)


def where(pred,a,b):
    kinds={np.asarray(x).dtype.kind for x in (a,b)}
    dtype=(np.float64 if any(is_double(x) for x in (a,b)) else np.float32) if 'f' in kinds else (np.bool_ if kinds=={'b'} else np.int32)
    return np.where(pred,np.asarray(a,dtype),np.asarray(b,dtype))


def get(a,index):
    if isinstance(a,(tuple,list,dict)):
        return a[index]
    index=int(index)
    if index<0:index+=len(a)
    return a[min(max(index,0),len(a)-1)]


def set_(a,index,value):
    out=a.copy()
    index=int(index)
    if index<0:index+=len(a)
    if 0<=index<len(a):out[index]=value
    return out


def tree_map(fn,*trees):
    one=trees[0]
    if isinstance(one,dict):return {key:tree_map(fn,*(tree[key] for tree in trees)) for key in one}
    if isinstance(one,tuple):return tuple(tree_map(fn,*(tree[k] for tree in trees)) for k in range(len(one)))
    if isinstance(one,list):return [tree_map(fn,*(tree[k] for tree in trees)) for k in range(len(one))]
    return fn(*trees)


def coerce(value):
    # np.float64 subclasses Python float: retain its explicit DOUBLE type.
    # Only untyped Python literals are WRF REAL/INTEGER loop initializers.
    if type(value) is bool:return np.bool_(value)
    if type(value) is float:return np.float32(value)
    if type(value) is int:return np.int32(value)
    return value


def fori(lower,upper,body,carry):
    carry=tree_map(coerce,carry)
    for k in range(int(lower),int(upper)):
        carry=tree_map(coerce,body(np.int32(k),carry))
    return carry


def while_(condition,body,carry):
    carry=tree_map(coerce,carry)
    trips=0
    while bool(condition(carry)):
        carry=tree_map(coerce,body(carry))
        trips+=1
        if trips>10000:raise RuntimeError('literal backend loop exceeds original column budget')
    return carry


class RealBackend:
    def __getattr__(self,name):
        if name in ('exp','log','log10','sqrt'):
            return lambda *values:intrinsic(name if any(is_double(x) for x in values) else name+'f',*values)
        return getattr(np,name)


class Literal(ast.NodeTransformer):
    def __init__(self,double=False,respect_with=False):
        self.double=double
        self.respect_with=respect_with

    def visit_With(self,node):
        original=self.double
        if self.respect_with:
            for item in node.items:
                call=item.context_expr
                if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                    and call.func.attr=='enable_x64' and call.args and isinstance(call.args[0],ast.Constant)):
                    self.double=bool(call.args[0].value)
        node=self.generic_visit(node)
        self.double=original
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,float):
            return ast.copy_location(ast.Call(ast.Name('_r64' if self.double else '_r32',ast.Load()),[node],[]),node)
        return node

    def visit_BinOp(self,node):
        node=self.generic_visit(node)
        return ast.copy_location(ast.Call(ast.Name('_binary',ast.Load()),
                                         [ast.Constant(type(node.op).__name__),node.left,node.right],[]),node)


def executor(source,scheme='kfcup'):
    real=RealBackend()
    fake=SimpleNamespace(lax=SimpleNamespace(
        select=lambda pred,yes,no:np.where(pred,yes,no),
        fori_loop=fori,while_loop=while_,cond=lambda pred,yes,no,arg:(yes if bool(pred) else no)(arg),
        bitcast_convert_type=lambda value,dtype:np.asarray(value).view(dtype)),
        tree_util=SimpleNamespace(tree_map=tree_map),enable_x64=lambda value:nullcontext())
    namespace=dict(np=np,jnp=real,jax=fake,_r32=np.float32,_r64=np.float64,_binary=binary,
                   _where=where,_get=get,_set=set_,_roll=lambda a,n:np.roll(a,int(n)),
                   _nint_positive=lambda x:np.floor(binary('Add',x,np.float32(.5))),
                   _real=lambda x:np.asarray(x,np.float32),
                   _gather_p=SimpleNamespace(bind=lambda a,i:a[np.asarray(i,np.int32)]))
    tables={}
    asset=source/'src/gpuwrf/physics/data/kfcup_real_tables.npz'
    with np.load(asset,allow_pickle=False) as raw:
        tables.update({key:np.array(raw[key]) for key in raw.files})
    namespace['tables']=SimpleNamespace(KFNT=250,KFNP=220,**{key:tables[key] for key in ('RDPR','RDTHK','PLUTOP')})
    wanted={
        'phys_kf_column.py':{'_closure_control'},
        'phys_kfcup_column.py':{'_cup_column','_cond'},
        'phys_kfcup_cps.py':{'_surface','_distribution','_empty','_cps_column','_cloud_fraction'},
        'cumulus_kfcup_support.py':{'find_cp','find_rs','cup_adjust_mass_flux'},
    }
    if scheme=='mskf':
        wanted={'phys_kf_column.py':{'_closure_control'},
                'cumulus_mskf_support.py':{'wrf_nint_i4','mskf_time_scale'},
                'phys_mskf_cmt.py':{'_cmt'},
                'phys_mskf_column.py':{'_mskf_column','_cond'}}
    for filename,names in wanted.items():
        folder='physics' if filename.startswith('cumulus_') else 'kernels'
        path=source/'src/gpuwrf'/folder/filename
        tree=ast.parse(path.read_text())
        nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and filename in ('phys_kfcup_column.py','phys_kfcup_cps.py','phys_mskf_column.py','phys_mskf_cmt.py'):
                nodes.append(node)
            if isinstance(node,ast.FunctionDef) and node.name in names:
                node.decorator_list=[]
                nodes.append(Literal(double=filename=='phys_mskf_cmt.py').visit(node))
        exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),str(path),'exec'),namespace)
    fn=namespace['_mskf_column' if scheme=='mskf' else '_cps_column']
    if scheme=='mskf':
        fn.harmonics=tuple(np.asarray(namespace[key]+(0.,)*6,np.float64) for key in ('_ALFA1','_ALFA2','_BSJ0','_BSJ1','_TAU'))
    return fn,tables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    import os
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    fn,tables=executor(args.source)
    oracle=json.loads(args.oracle.read_text())
    names=('T','QV','P','DZ','RHO','W0AVG','U','V','TH','Z','ZW','PI')
    records=[]
    provenance=dict(
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_receipt_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(args.source)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.source/'src/gpuwrf/kernels/phys_kf_column.py',
                         args.source/'src/gpuwrf/kernels/phys_kfcup_column.py',
                         args.source/'src/gpuwrf/kernels/phys_kfcup_cps.py',
                         args.source/'src/gpuwrf/physics/cumulus_kfcup_support.py',
                         args.source/'src/gpuwrf/physics/data/kfcup_real_tables.npz')},
        arithmetic='NumPy REAL32 after every binary operation; gfortran host scalar libm ABI; no FMA',
    )
    for row in oracle['records'][:args.limit]:
        if row['mode']!='plain':continue
        fields,scalars=parse(row['stdout'])
        n=len(fields['T'])
        profiles=tuple(np.pad(np.asarray(fields[key],np.float32),(1,64-n-1)) for key in names)
        ctrl=np.asarray([scalars.get('CTRL_DT',54.),scalars.get('CTRL_DX',9000.),
            scalars.get('CTRL_PBLH',1500.),scalars.get('CTRL_KPBL',6.),
            scalars.get('CTRL_HFX',float(row['controls'][0]) if row['controls'] else 0.),
            scalars.get('CTRL_QFX',1e-4),scalars.get('CTRL_XLAND',1.),scalars.get('CTRL_HT',0.)],np.float32)
        start=time.monotonic()
        with np.errstate(all='ignore'):
            actual=fn(profiles,ctrl,n,tables['THE0K'],tables['TTAB'],tables['QSTAB'],tables['ALU'],
                      num_bins=5,th_bin_size=np.float32(.1),r_bin_size=np.float32(1e-4),
                      min_deep_freq=np.float32(.1),min_shallow_freq=np.float32(.001),
                      warm_rain=False,f_qi=True,f_qs=True)
        differences={};count=0
        for key,value in actual.items():
            if key in fields:
                candidate=np.asarray(value)[1:n+1]
                truth=np.asarray(fields[key],np.float32)
            elif (lookup:='SHALL' if key=='ISHALL' else key) in scalars:
                candidate=np.asarray(value,np.float32)
                truth=np.asarray(scalars[lookup],np.float32)
            else:continue
            assert np.isfinite(candidate).all() and np.isfinite(truth).all(),key
            assert candidate.dtype==np.float32,key
            changed=np.count_nonzero(candidate.view(np.uint32)!=truth.view(np.uint32));count+=candidate.size
            if changed:differences[key]=dict(changed=int(changed),max_abs=float(np.max(np.abs(candidate-truth))))
        result=dict(case=row['case'],label=row['label'],compared_values=count,differences=differences,
                    elapsed_s=time.monotonic()-start)
        records.append(result)
        args.output.write_text(json.dumps(dict(source=str(args.source),records=records,provenance=provenance,
            scope='candidate literal backend vs original WRF, not a new oracle'),indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
