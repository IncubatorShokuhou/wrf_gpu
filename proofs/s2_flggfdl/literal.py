"""Evaluate the SAME port functions as unfused REAL4 + system libm *f calls.

Proof-only CPU facade: NumPy rounds each array operation to float32; scalar
transcendentals call libm expf/logf/sqrtf/sinf/cosf, like the native oracle.
This is the literal/no-FMA byte gate, not a second implementation of physics.
Production JAX functions and compiled tests remain separate evidence.
"""
from __future__ import annotations

import ctypes
import ast
import importlib.util
from pathlib import Path

import numpy as np


class RealArray(np.ndarray):
    @property
    def at(self):
        return _At(self)


class _At:
    def __init__(self,array,index=None):
        self.array,self.index=array,index

    def __getitem__(self,index):
        return _At(self.array,index)

    def set(self,value):
        out=self.array.copy()
        out[self.index]=value
        return out


def _wrap(value):
    if isinstance(value,np.ndarray):
        if value.dtype == np.float64:
            raise AssertionError("Literal REAL4 evaluator introduced a float64 array")
        return value.view(RealArray)
    if isinstance(value,(tuple,list)):
        return type(value)(_wrap(x) for x in value)
    return value


class RealMath:
    _libm=ctypes.CDLL("libm.so.6")

    def __getattr__(self,name):
        if name in ("exp","log","log10","sqrt","sin","cos","arcsin","arccos"):
            fn=getattr(self._libm,{"arcsin":"asin","arccos":"acos"}.get(name,name)+"f")
            fn.argtypes=[ctypes.c_float];fn.restype=ctypes.c_float
            def transcendental(a):
                a=np.asarray(a)
                assert a.dtype == np.float32
                values=np.fromiter((fn(float(x)) for x in a.flat),dtype=np.float32,count=a.size)
                return values.reshape(a.shape).view(RealArray)
            return transcendental
        attr=getattr(np,name)
        if not callable(attr) or isinstance(attr,type):
            return attr
        def call(*args,**kwargs):
            # Bare literals in WRF are REAL4. NumPy scalar where(0., x_f32)
            # otherwise treats 0. as a strong float64, unlike WRF literals.
            args=tuple(np.float32(x) if isinstance(x,float) else x for x in args)
            return _wrap(attr(*args,**kwargs))
        return call


class RealLoops:
    @staticmethod
    def div(left,right):
        left,right=np.asarray(left),np.asarray(right)
        if left.dtype.kind in 'iu' and right.dtype.kind in 'iu':
            quotient=np.abs(left)//np.abs(right)
            return np.where((left<0)!=(right<0),-quotient,quotient).astype(left.dtype)
        return left/right

    @staticmethod
    def while_loop(condition,body,carry):
        while bool(condition(carry)):
            carry=body(carry)
        return carry

    @staticmethod
    def cond(predicate,true,false,operand):
        return true(operand) if bool(predicate) else false(operand)

    @staticmethod
    def fori_loop(lower,upper,body,carry):
        for index in range(int(lower),int(upper)):
            carry=body(index,carry)
        return carry


class _ScalarStatementControl(ast.NodeTransformer):
    """Evaluate generated scalar statement predicates before array operands.

    JAX's predicated reference evaluates both where operands; native Fortran
    skips an inactive statement. This proof facade keeps the same arithmetic
    expressions on active paths and strict NumPy bounds checks there. It never
    clips an index or changes a physical operand.
    """
    def visit_BinOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.BitAnd):
            # Generated IF predicates are mask & condition. Native code does
            # not evaluate an IF condition after an enclosing RETURN/GOTO.
            thunk=ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=node.right)
            return ast.copy_location(ast.Call(func=ast.Name('_source_and',ast.Load()),args=[node.left,thunk],keywords=[]),node)
        return node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        value=node.value
        where=value
        indexed=False
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='set':
            if len(value.args)!=1:return node
            where=value.args[0];indexed=True
        if not (isinstance(where,ast.Call) and isinstance(where.func,ast.Attribute)
                and isinstance(where.func.value,ast.Name) and where.func.value.id=='jnp'
                and where.func.attr=='where' and len(where.args)==3):return node
        if indexed:
            at=value.func.value
            if not (isinstance(at,ast.Subscript) and isinstance(at.value,ast.Attribute) and at.value.attr=='at'):return node
            old=ast.Subscript(value=at.value.value,slice=at.slice,ctx=ast.Load())
            assert ast.unparse(old)==ast.unparse(where.args[2]),'Non-identity inactive array assignment'
            value.args[0]=where.args[1]
        else:
            assert len(node.targets)==1 and ast.unparse(node.targets[0])==ast.unparse(where.args[2]),'Non-identity inactive scalar assignment'
            # A whole-array native assignment broadcasts its scalar RHS.
            # Preserve the old target's shape/type when skipping eager where.
            node.value=ast.Call(func=ast.Name('_source_assign',ast.Load()),args=[where.args[2],where.args[1]],keywords=[])
        return ast.copy_location(ast.If(test=where.args[0],body=[node],orelse=[]),node)


def _source_and(left,right):
    if np.ndim(left)==0 and np.asarray(left).dtype.kind=='b' and not bool(left):
        return np.bool_(False)
    return left & right()


def _source_assign(old,value):
    if isinstance(old,np.ndarray):
        return np.broadcast_to(np.asarray(value,dtype=old.dtype),old.shape).copy().view(RealArray)
    if isinstance(old,np.generic):return old.dtype.type(value)
    return value


def load(source: Path,*,source_control=False):
    spec=importlib.util.spec_from_file_location("_literal_"+source.stem,source)
    module=importlib.util.module_from_spec(spec)
    if source_control:
        module._source_and=_source_and
        module._source_assign=_source_assign
        tree=_ScalarStatementControl().visit(ast.parse(source.read_text()))
        ast.fix_missing_locations(tree)
        exec(compile(tree,str(source),'exec'),vars(module))
    else:
        spec.loader.exec_module(module)
    module.jnp=RealMath()
    module.lax=RealLoops()
    if hasattr(module,'transfer'):
        module.transfer=load(Path(module.transfer.__file__))
    return module
