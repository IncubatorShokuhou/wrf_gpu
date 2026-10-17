"""Generate literal source-mapped MY2 point processes, never used at runtime.

Only the independent (i,k) loops are lifted to vectors. IF stores become masked
stores; the two depletion sweeps remain in their original order. Unsupported
syntax is fatal. Source line numbers make generated equations auditable.
"""
from pathlib import Path
import ast
import re
import hashlib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_milbrandt2mom.F')
FIELDS = 't q qc qr qi qn qg qh nc nr ny nn ng nh'.split()


def statements(lo, hi):
    pending = ''
    start = 0
    for n, raw in enumerate(SOURCE.read_text().splitlines(), 1):
        line = raw.split('!', 1)[0].strip().lower()
        if not line:
            continue
        if not pending:
            start = n
        pending += ' ' + line.lstrip('&').rstrip('&').strip()
        if line.endswith('&'):
            continue
        for stmt in pending.strip().split(';'):
            if stmt.strip() and lo <= start <= hi:
                yield start, stmt.strip()
        pending = ''
    assert not pending


class Expr(ast.NodeTransformer):
    def visit_BinOp(self, node):
        a,b=self.visit(node.left),self.visit(node.right)
        if isinstance(node.op,ast.Pow):
            return ast.Call(ast.Name('_pow',ast.Load()),[a,b],[])
        ops={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div'}
        if type(node.op) in ops:
            return ast.Call(ast.Name('_op',ast.Load()),[ast.Constant(ops[type(node.op)]),a,b],[])
        return ast.BinOp(a,node.op,b)

    def visit_Constant(self, node):
        if isinstance(node.value, float):
            return ast.Call(ast.Name('_r', ast.Load()), [node], [])
        return node

    def visit_BoolOp(self, node):
        vals = [self.visit(x) for x in node.values]
        result = vals[0]
        for val in vals[1:]:
            result = ast.BinOp(result, ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr(), val)
        return result

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return ast.Call(ast.Name('_not',ast.Load()),[self.visit(node.operand)],[])
        return self.generic_visit(node)

    def visit_Call(self, node):
        name = node.func.id
        args = [self.visit(x) for x in node.args]
        funcs = {'exp':'_exp','log':'_log','log10':'_log10', 'sqrt':'jnp.sqrt',
                 'abs':'jnp.abs','dble':'_d','float':'_r','dim':'_dim'}
        if name in ('min','max'):
            result = args[0]
            for x in args[1:]:
                result = ast.Call(ast.parse('jnp.minimum' if name == 'min' else 'jnp.maximum',mode='eval').body,[result,x],[])
            return result
        if name in funcs:
            return ast.Call(ast.parse(funcs[name],mode='eval').body,args,[])
        if name in ('dm_x','ilamda_x','des_of_ds','n_cooper','nos_thompson','nccnfnc','sxfnc','qsat','gamma'):
            return ast.Call(ast.Name('_'+name,ast.Load()),args,[])
        raise ValueError(f'unknown function {name}')


def expr(s):
    s = re.sub(r'\b([a-z]\w*)\(i,k\)', r'\1', s)
    s = re.sub(r'\.true\.', 'True', s)
    s = re.sub(r'\.false\.', 'False', s)
    for a,b in {'.and.':' and ','.or.':' or ','.not.':' not ','.eq.':'==','.ne.':'!=',
                '.lt.':'<','.le.':'<=','.gt.':'>','.ge.':'>=','/=':'!='}.items():
        s=s.replace(a,b)
    tree = ast.parse(s.strip(), mode='eval')
    return ast.unparse(ast.fix_missing_locations(Expr().visit(tree))).strip()


def translate(stmts, mask='True'):
    out=[]; assigned=set(); stack=[]; loops=[]; counter=0; indent=1
    def emit(s): out.append('    '*indent+s)
    for n,s in stmts:
        if s.startswith(('do k','do i')):
            loops.append(False)
            continue
        if s=='enddo':
            if loops.pop(): indent-=1
            continue
        if s.startswith('print') or s=='stop':
            continue
        if s.startswith('do niter'):
            loops.append(True); emit('for _depletion_pass in range(2):'); indent+=1; continue
        # Depletion END DO is distinguished below by its original source location.
        if s in ('endif','end if'):
            mask=stack.pop()[0]; continue
        m=re.match(r'else\s*if\s*\((.*)\)\s*then$',s)
        if m:
            parent,seen=stack[-1]
            counter+=1; new=f'_m{counter}'
            emit(f'{new} = ({parent}) & _not({seen}) & ({expr(m[1])})')
            stack[-1]=(parent,f'({seen}) | ({new})'); mask=new; continue
        if s=='else':
            parent,seen=stack[-1]; counter+=1; mask=f'_m{counter}'
            emit(f'{mask} = ({parent}) & _not({seen})'); continue
        if s.startswith('if'):
            # Find matching close-paren, rather than a greedy regexp for one-line IF.
            op=s.index('('); depth=1; end=op+1
            while depth:
                depth+=(s[end]=='(')-(s[end]==')'); end+=1
            cond=expr(s[op+1:end-1]); tail=s[end:].strip()
            counter+=1; new=f'_m{counter}'
            emit(f'{new} = ({mask}) & ({cond})')
            if tail=='then':
                stack.append((mask,new)); mask=new; continue
            effective=new; s=tail
        else:
            effective=mask
        if s.startswith('call check_values') or not s:
            continue
        m=re.fullmatch(r'([a-z]\w*)(?:\(i,k\))?\s*=\s*(.*)',s)
        if not m:
            raise ValueError((n,s))
        name,rhs=m.groups(); assigned.add(name)
        emit(f'# WRF:{n}')
        rhs=expr(rhs)
        cast='jnp.asarray' if name in ('activepoint','log1','log2','log3','log4','rainpresent') else '_r'
        if effective=='True': emit(f'{name} = {cast}({rhs})')
        else: emit(f'{name} = jnp.where({effective}, {cast}({rhs}), {name})')
    assert not stack, stack
    return out,assigned


def main():
    class MathExpr(Expr):
        def visit_Constant(self,node): return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='jnp' and node.func.attr in ('exp','log','log10'):
                node.func=ast.Name('_'+node.func.attr,ast.Load())
            return node
    mt=ast.parse((ROOT/'proofs/s2_milbrandt/math_template.py').read_text())
    mt.body=[MathExpr().visit(x) if isinstance(x,ast.FunctionDef) and x.name!='_gamma' else x for x in mt.body]
    mt.body.insert(3,ast.parse('from .rounding import _op, _pow, _exp, _log, _log10').body[0])
    (ROOT/'src/gpuwrf/physics/milbrandt2mom/math.py').write_text(ast.unparse(ast.fix_missing_locations(mt))+'\n')
    params=[]
    for n,s in statements(1020,1178):
        if 'parameter ::' not in s:
            continue
        s=s.split('::',1)[1].strip()
        for item in s.split(','):
            name,val=item.split('=',1)
            cast='bool' if 'logical' in SOURCE.read_text().splitlines()[n-1].lower() else '_r'
            params.append(f'    {name.strip()} = {cast}({expr(val)})  # WRF:{n}')
    init=list(statements(1257,1438))
    # Preserve branching via the same masked translator. All its inputs are constants.
    initout,initnames=translate(init)
    stages=[]; names=set()
    for ranges, mask in [([(1445,1531)], 'True'),
                         ([(1591,1602),(1610,2828)], 'point_mask'),
                         ([(2840,3128)], 'point_mask')]:
        stmts=[]
        for lo,hi in ranges: stmts.extend(statements(lo,hi))
        lines, assigned=translate(stmts,mask)
        stages.append(lines); names |= assigned
    header='''"""Generated literal Milbrandt-Yau local processes; regenerate with proofs/s2_milbrandt/generate_point.py.
Pristine source SHA256: SOURCE_SHA
"""
'''.replace('SOURCE_SHA',hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    paramnames={re.match(r'    (\w+)',s)[1] for s in params}
    constnames=(paramnames|initnames)-{'idt'}
    const=header+'''import numpy as jnp
from functools import lru_cache
from .math import _gamma

def _op(op,a,b):
    return {'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[op]()
def _pow(a,b): return a**b
_exp, _log, _log10 = jnp.exp, jnp.log, jnp.log10
_not = jnp.logical_not

@lru_cache(maxsize=2)
def constants(dtype='float32'):
    _r = jnp.dtype(dtype).type
    ccntype = 2
    dt = _r(1.)  # idt is NOT cached; computed from actual dt in local_processes
'''
    const += '\n'.join(params)+ '\n'
    const += '\n'.join('    '+x+' = _r(0.)' for x in sorted(initnames))+'\n'
    const += '\n'.join(initout)+'\n'
    const += '    return {\n'+ ''.join(f'        {x!r}: {x},\n' for x in sorted(constnames))+'    }\n'
    (ROOT/'src/gpuwrf/physics/milbrandt2mom/constants.py').write_text(const)
    code=header+'''import jax.numpy as jnp
from .math import (_dm_x, _ilamda_x, _des_of_ds, _n_cooper, _nos_thompson,
                   _nccnfnc, _sxfnc, _qsat, _dim)
from .rounding import _op, _pow, _exp, _log, _log10, real, _not

def local_processes(fields, pres, wz, dt, c, point_mask):
    t, q, qc, qr, qi, qn, qg, qh, nc, nr, ny, nn, ng, nh = fields
    _r = lambda x: real(x, t.dtype)
    _d = lambda x: jnp.asarray(x, jnp.float64)
    ccntype = 2
    icephase_on = warmphase_on = snow_on = autoconv_on = True
    dt = _r(dt)
    idt = _op('div',_r(1.),dt)
    de = _op('div',pres,_op('mul',_r(c['rgasd']),t))
    ide = _op('div',_r(1.),de)
    gamfact = jnp.sqrt(_op('div',_r(c['deo']),de))
    qsw = _qsat(t, pres, 0)
    qsi = _qsat(t, pres, 1)
    nc, nr, ny, nn, ng, nh = (_op('mul',x,de) for x in (nc, nr, ny, nn, ng, nh))
'''
    code+='\n'.join(f'    {x} = c[{x!r}]' for x in sorted(constnames))+'\n'
    notinit=set(FIELDS)|constnames|{'de','ide','qsw','qsi'}
    for x in sorted(names-notinit):
        code+=f'    {x} = '+ ('jnp.zeros_like(t, dtype=jnp.bool_)' if x in ('activepoint','log1','log2','log3','log4','rainpresent') else '_r(0.)')+'\n'
    code+='\n'.join(stages[0])+'\n'
    code+='    qc_in, qr_in, nc_in, nr_in = qc, qr, nc, nr\n'
    code+='\n'.join(stages[1]+stages[2])+'\n'
    code+='    return (t, q, qc, qr, qi, qn, qg, qh, nc, nr, ny, nn, ng, nh), de, ide\n'
    (ROOT/'src/gpuwrf/physics/milbrandt2mom/local.py').write_text(code)
    print('parameter count',len(params),'init',len(initnames),'point',len(names))

if __name__ == '__main__':
    main()
