"""No-FMA scalar execution of the generated SSiB arithmetic for BYTE gates.

This validation-only backend evaluates every REAL binary operation separately,
uses the system libm REAL intrinsics used by GNU Fortran, and applies Fortran
mixed integer/REAL promotion (NumPy would otherwise promote REAL32+INTEGER32).
There are no JAX optimizations, GPU callbacks, or forecast entry points here.
"""
import ast
import ctypes
import ctypes.util
import types
import numpy as np

_LIBM=ctypes.CDLL(ctypes.util.find_library('m'))
_FUN={}
for _name in ('exp','log','sqrt','sin','cos','tan','asin','acos','atan','tanh','pow'):
    for _double in (False,True):
        fn=getattr(_LIBM,_name+('' if _double else 'f'))
        typ=ctypes.c_double if _double else ctypes.c_float
        fn.argtypes=[typ]*(2 if _name=='pow' else 1);fn.restype=typ
        _FUN[_name,_double]=fn


def _dtype(*args):
    dtypes=[np.asarray(a).dtype for a in args]
    if any(d==np.float64 for d in dtypes): return np.float64
    if any(d==np.float32 for d in dtypes): return np.float32
    if all(d==np.bool_ for d in dtypes): return np.bool_
    return np.int32


def _op(a,b,op):
    integer_exponent=np.issubdtype(np.asarray(b).dtype,np.integer)
    dtype=_dtype(a,b);a=np.asarray(a,dtype=dtype);b=np.asarray(b,dtype=dtype)
    if op=='Pow':
        if integer_exponent:
            exponent=int(b)
            if exponent<0: return _op(dtype(1),_op(a,-exponent,'Pow'),'Div')
            def power(n):
                if n==0:return dtype(1)
                if n==1:return a
                half=power(n//2);square=np.multiply(half,half,dtype=dtype)
                return np.multiply(square,a,dtype=dtype) if n%2 else square
            return power(exponent)
        return dtype(_FUN['pow',dtype==np.float64](float(a),float(b)))
    operations={'Add':np.add,'Sub':np.subtract,'Mult':np.multiply,'Div':np.divide,
                'BitAnd':np.bitwise_and,'BitOr':np.bitwise_or}
    return operations[op](a,b,dtype=dtype)


def _math(name,a):
    dtype=_dtype(a)
    if np.ndim(a): return np.asarray([_math(name,x) for x in a],dtype=dtype)
    return dtype(_FUN[name,dtype==np.float64](float(a)))


def _set(a,idx,value):
    out=np.array(a,copy=True);out[idx]=value;return out


def _minimum(*args):
    dtype=_dtype(*args);value=np.asarray(args[0],dtype=dtype)
    for a in args[1:]: value=np.minimum(value,np.asarray(a,dtype=dtype))
    return value


def _maximum(*args):
    dtype=_dtype(*args);value=np.asarray(args[0],dtype=dtype)
    for a in args[1:]: value=np.maximum(value,np.asarray(a,dtype=dtype))
    return value


class _Lax:
    @staticmethod
    def cond(predicate,true,false,arg):return true(arg) if bool(predicate) else false(arg)
    @staticmethod
    def while_loop(predicate,body,carry):
        count=0
        while bool(predicate(carry)):
            carry=body(carry);count+=1
            if count>2000:raise RuntimeError('literal column failed to converge')
        return carry


class _Transform(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='jnp':return ast.copy_location(ast.Name('np',ast.Load()),node)
        if node.id=='lax':return ast.copy_location(ast.Name('_Lax',ast.Load()),node)
        return node
    def visit_Assign(self,node):
        value=node.value; selected=None
        if isinstance(value,ast.Call):
            if isinstance(value.func,ast.Attribute) and value.func.attr=='where':selected=value
            elif isinstance(value.func,ast.Attribute) and value.func.attr=='set' and value.args:
                inner=value.args[0]
                if isinstance(inner,ast.Call) and isinstance(inner.func,ast.Attribute) and inner.func.attr=='where':selected=inner
        transformed=self.generic_visit(node)
        if selected is not None:
            return ast.copy_location(ast.If(self.visit(selected.args[0]),[transformed],[]),node)
        return transformed
    def visit_BinOp(self,node):
        if isinstance(node.op,(ast.BitAnd,ast.BitOr)):
            return ast.copy_location(ast.BoolOp(ast.And() if isinstance(node.op,ast.BitAnd) else ast.Or(),
                [self.visit(node.left),self.visit(node.right)]),node)
        return ast.copy_location(ast.Call(ast.Name('_op',ast.Load()),
            [self.visit(node.left),self.visit(node.right),ast.Constant(type(node.op).__name__)],[]),node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and node.func.attr=='set':
            sub=node.func.value
            if isinstance(sub,ast.Subscript) and isinstance(sub.value,ast.Attribute) and sub.value.attr=='at':
                return ast.copy_location(ast.Call(ast.Name('_set',ast.Load()),
                    [self.visit(sub.value.value),self.visit(sub.slice),self.visit(node.args[0])],[]),node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='jnp':
            names={'exp':'exp','log':'log','sqrt':'sqrt','sin':'sin','cos':'cos','tan':'tan','arcsin':'asin','arccos':'acos','arctan':'atan','tanh':'tanh'}
            if node.func.attr in names:
                return ast.copy_location(ast.Call(ast.Name('_math',ast.Load()),
                    [ast.Constant(names[node.func.attr]),self.visit(node.args[0])],[]),node)
        return self.generic_visit(node)


def load(source):
    tree=ast.parse(source.read_text())
    tree.body=[node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    tree=_Transform().visit(tree);ast.fix_missing_locations(tree)
    module=types.ModuleType('ssib_literal')
    module.__dict__.update(np=np,_op=_op,_math=_math,_set=_set,_Lax=_Lax)
    exec(compile(tree,str(source),'exec'),module.__dict__)
    module._min=_minimum;module._max=_maximum
    return module
