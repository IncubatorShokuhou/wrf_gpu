"""Offline literal emitter for NTU's loop-free scalar microphysics routines.

Only this frozen source grammar is supported. Emitted Python is reviewable and
line-attributed; no Fortran parsing occurs during a forecast. Native Pallas and
the NumPy literal oracle share the expressions. Column orchestration is separate.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

SOURCE = Path(__file__).resolve().parents[2] / "data/wrf_pristine/WRF/phys/module_mp_ntu.F"
OUT = Path(__file__).resolve().parents[2] / "src/gpuwrf/physics/ntu"
ROUTINES = "solve_afac solve_afar solve_afai solve_afas solve_afag solve_afah icenu sedi_fall small_dt large_dt".split()
AEROSOL_ROUTINES = "rswhitby ccnwhitby activa deactiva".split()
MODIFIED = {}
# Masked CALLs skipped when no lane merges their outputs (support.masked_call). small_dt: on the
# heaviest real c88 columns 97-99 % of the small steps have no active cloudy level (t-ntucompile NP12).
SKIP_UNSELECTED = {"small_dt"}


def split(text, delimiter=","):
    depth = 0
    start = 0
    result = []
    for i,ch in enumerate(text):
        depth += (ch=="(")-(ch==")")
        if ch==delimiter and depth==0:
            result.append(text[start:i].strip());start=i+1
    return result+[text[start:].strip()]


def statements(source):
    pending = ""
    start = 0
    for n,line in enumerate(source.splitlines(),1):
        line=line.split("!",1)[0].strip().lower()
        if not line or line.startswith("#"):
            continue
        if not pending:
            start=n
        pending+=" "+line.lstrip("&").rstrip("&").strip()
        if line.endswith("&"):
            continue
        for stmt in pending.strip().split(";"):
            if stmt.strip():
                yield start,stmt.strip()
        pending=""


def parse(text):
    for a,b in ((".and."," and "),(".or."," or "),(".not."," not "),
                (".ge.",">="),(".gt.",">"),(".le.","<="),(".lt.","<"),
                (".eq.","=="),(".ne.","!="),(".true.","True"),(".false.","False")):
        text=text.replace(a,b)
    # Preserve DOUBLE literal identity instead of replacing its exponent blindly.
    text=re.sub(r"(?<![\w.])((?:\d+\.?\d*|\.\d+)[dD][+-]?\d+)",lambda m:"_dlit("+m[1].lower().replace("d","e")+")",text)
    return ast.parse(text.strip(),mode="eval").body


INTRINSICS={"exp":"xp.exp","log":"xp.log","sqrt":"xp.sqrt","abs":"xp.abs","sin":"xp.sin",
            "cos":"xp.cos","tanh":"xp.tanh","log10":"xp.log10","acos":"xp.arccos",
            "max":"_max","min":"_min","int":"_int","nint":"_nint","real":"r",
            "sngl":"_real","dble":"_double","real":"_real", "derf":"_erf","erf":"_erf",
            "dexp":"xp.exp","dlog":"xp.log","dsqrt":"xp.sqrt","dabs":"xp.abs"}


class Expression(ast.NodeTransformer):
    def __init__(self,arrays,integer_names=(),local_arrays=(),level_axes=None):
        self.arrays=arrays
        self.integer_names=set(integer_names)
        self.local_arrays=set(local_arrays)
        self.level_axes=level_axes or {}

    def visit_Name(self,node):
        if node.id in ("True","False"):
            return node
        return ast.Subscript(value=ast.Name(id="v",ctx=ast.Load()),slice=ast.Constant(node.id),ctx=ast.Load())

    def visit_Constant(self,node):
        if type(node.value) is float:
            return ast.Call(func=ast.Name(id="r",ctx=ast.Load()),args=[node],keywords=[])
        return node

    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Pow):
            return ast.Call(func=ast.parse('xp.power',mode='eval').body,
                            args=[self.visit(node.left),self.visit(node.right)],keywords=[])
        return self.generic_visit(node)

    def visit_BoolOp(self,node):
        name="logical_and" if isinstance(node.op,ast.And) else "logical_or"
        args=[self.visit(x) for x in node.values]
        result=args[0]
        for arg in args[1:]:
            result=ast.Call(func=ast.Attribute(value=ast.Name(id="xp",ctx=ast.Load()),attr=name,ctx=ast.Load()),args=[result,arg],keywords=[])
        return result

    def visit_UnaryOp(self,node):
        if isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant):
            return self.visit(ast.Constant(-node.operand.value))
        if isinstance(node.op,ast.Not):
            return ast.Call(func=ast.parse("xp.logical_not",mode="eval").body,args=[self.visit(node.operand)],keywords=[])
        return self.generic_visit(node)

    def visit_Call(self,node):
        name=node.func.id
        if name=="_dlit":
            return ast.Call(func=ast.Name(id="_double",ctx=ast.Load()),args=node.args,keywords=[])
        if name in self.arrays:
            lower=self.arrays[name]
            assert len(node.args)==len(lower),(name,node.args,lower)
            indices=[ast.BinOp(left=self.visit(x),op=ast.Sub(),right=ast.Constant(lo)) for i,(x,lo) in enumerate(zip(node.args,lower)) if i not in self.level_axes.get(name,())]
            if not indices:return self.visit_Name(ast.Name(id=name,ctx=ast.Load()))
            index=indices[0] if len(indices)==1 else ast.Tuple(elts=indices,ctx=ast.Load())
            return ast.Call(func=ast.Name(id="_get_local" if name in self.local_arrays else "_get",ctx=ast.Load()),args=[self.visit_Name(ast.Name(id=name,ctx=ast.Load())),index],keywords=[])
        if name in INTRINSICS:
            if name=='max' and len(node.args)==2:
                for i,arg in enumerate(node.args):
                    if isinstance(arg,ast.Constant) and arg.value==0:
                        # Original gfortran MAX(0., NaN) with a literal floor
                        # returns zero (unlike MAX(a,b) with variable operands).
                        return ast.Call(func=ast.Name(id='_max0',ctx=ast.Load()),
                                        args=[self.visit(node.args[1-i])],keywords=[])
            return ast.Call(func=ast.parse(INTRINSICS[name],mode="eval").body,args=[self.visit(x) for x in node.args],keywords=[])
        if name in {"gamln","gamin","gammp","polysvp","dlnx","dlnx2","dpdf"}:
            return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(x) for x in node.args],keywords=[ast.keyword(arg="xp",value=ast.Name(id="xp",ctx=ast.Load()))])
        raise ValueError((name,ast.unparse(node)))


def expression(text,arrays,ints=(),local_arrays=(),level_axes=None):
    return ast.unparse(ast.fix_missing_locations(Expression(arrays,ints,local_arrays,level_axes).visit(parse(text))))


def declarations(block):
    types={};arrays={};params=[];data=[];body=[]
    for n,s in block:
        if s.startswith("implicit"):
            continue
        if re.match(r"^(real|integer|logical|double precision)\b",s) and "::" in s:
            lhs,rhs=s.split("::",1)
            kind=lhs.split(",")[0].strip()
            dim=re.search(r"dimension\((.*?)\)",lhs)
            for entry in split(rhs):
                name=entry.split("=")[0].strip()
                own=re.match(r"(\w+)\((.*)\)",name)
                if own:
                    name=own[1]
                types[name]=kind
                dims=split(own[2]) if own else split(dim[1]) if dim else None
                if dims:
                    arrays[name]=tuple(int(d.split(":")[0]) if ":" in d and d.split(":")[0].isdigit() else 1 for d in dims)
                if "parameter" in lhs:
                    target,value=entry.split("=",1);params.append((n,target.strip(),value.strip()))
            continue
        if s.startswith("data "):
            match=re.fullmatch(r"data\s+([^/]+)/([^/]+)/",s)
            assert match,s
            names=split(match[1]);values=split(match[2])
            if len(names)==1 and names[0] in arrays:
                data.append((n,names[0],values))
            else:
                assert len(names)==len(values),s
                params.extend((n,k,x) for k,x in zip(names,values))
            continue
        body.append((n,s))
    return types,arrays,params,data,body


def generate(name,header,block,globals_,global_arrays,signatures,*,level=False):
    args=split(header[header.index("(")+1:header.rindex(")")])
    types,arrays,params,data,body=declarations(block)
    local_arrays=dict(arrays)
    level_axes={};extents={}
    if level:
        for _,stmt in block:
            if '::' not in stmt:continue
            left,right=stmt.split('::',1);dim=re.search(r'dimension\((.*?)\)',left)
            if dim:
                dims=split(dim[1])
                for entry in split(right):
                    key=entry.split('=')[0].strip()
                    level_axes[key]=tuple(i for i,d in enumerate(dims) if 'kts' in d or 'kte' in d)
                    extents[key]=[d for i,d in enumerate(dims) if i not in level_axes[key]]
    arrays=dict(global_arrays)|local_arrays
    ints={k for k,t in types.items() if t=="integer"}
    bools={k for k,t in types.items() if t=="logical"}
    doubles={k for k,t in types.items() if t=="double precision"}
    loops={re.match(r'do\s+(\w+)\s*=',s)[1] for _,s in body if re.match(r'do\s+(\w+)\s*=',s)}
    expr=lambda s:expression(s,arrays,ints,local_arrays,level_axes)
    assigned=set()
    output=[];stack=[];counter=0;mask="active" if level else "True";indent=1
    static_indices={'iv0','iv1','iv2','iv3','iv4','iv','im','j','i','it','kts','kte'} if level else set()
    static_names=(set(globals_)-set(types)) | loops | static_indices | ({'iae'} if name in AEROSOL_ROUTINES else set())
    def is_static(condition):
        return {x.id for x in ast.walk(parse(condition)) if isinstance(x,ast.Name)} <= static_names
    emit=lambda s:output.append("    "*indent+s)
    def assign(lhs,rhs,n,active=None):
        node=parse(lhs)
        key=node.id if isinstance(node,ast.Name) else node.func.id
        assigned.add(key)
        dtype="np.int32" if key in ints else "bool" if key in bools else "np.float64" if key in doubles else "ref.dtype"
        selected=f"xp.logical_and({mask if active is None else active}, alive)"
        if key in static_indices:
            emit(f"v[{key!r}] = {rhs}  # WRF:{n}; static aerosol index")
        elif isinstance(node,ast.Call) and key in level_axes and len(node.args)==len(level_axes[key]):
            emit(f"v[{key!r}] = xp.where({selected}, xp.asarray({rhs}, dtype={dtype}), v[{key!r}])  # WRF:{n}")
        elif isinstance(node,ast.Call):
            assert key in local_arrays,(name,n,lhs)
            index=", ".join(expr(ast.unparse(a))+f"-({lo})" for i,(a,lo) in enumerate(zip(node.args,local_arrays[key])) if i not in level_axes.get(key,()))
            emit(f"v[{key!r}] = _set_local(v[{key!r}], ({index},), xp.asarray({rhs}, dtype={dtype}), {selected}, xp=xp)  # WRF:{n}")
        elif key in local_arrays:
            emit(f"v[{key!r}] = put_value(v[{key!r}], xp.asarray({rhs}, dtype={dtype}), {selected}, xp=xp)  # WRF:{n}")
        else:
            emit(f"v[{key!r}] = xp.where({selected}, xp.asarray({rhs}, dtype={dtype}), v[{key!r}])  # WRF:{n}")
    def action(s,n,active=None):
        if s.startswith("call "):
            node=parse(s[5:]);callee=node.func.id
            def address(arg):
                if isinstance(arg,ast.Call) and arg.func.id in arrays:
                    indices=[x for i,x in enumerate(arg.args) if i not in level_axes.get(arg.func.id,())]
                    assert isinstance(indices[0],ast.Constant) and indices[0].value==1
                    if len(indices)==1:return f"v[{arg.func.id!r}]"
                    assert len(indices)==2
                    return f"get_column(v[{arg.func.id!r}], {expr(ast.unparse(indices[1]))}-1, xp=xp)"
                return expr(ast.unparse(arg))
            if callee=='find_rc0':
                assert len(node.args)==7
                emit("rc_out = find_rc0("+", ".join(address(node.args[i]) for i in (0,1,2,3,5,6))+", xp=xp)")
                assign(ast.unparse(node.args[4]),'rc_out',n,active)
                return
            assert callee in signatures,(name,n,callee)
            cargs=signatures[callee]
            assert len(cargs)==len(node.args),(callee,cargs,node.args)
            array_formals={'rswhitby':{'zccn','cnmod','cnstd'},'ccnwhitby':{'zccn','cnmod','cnstd'},
                           'activa':{'abcd','zccns'},'deactiva':{'zccns'}}.get(callee,set())
            ins=[]
            for key,arg in zip(cargs,node.args):
                if key in array_formals:
                    ins.append(address(arg))
                else:ins.append(expr(ast.unparse(arg)))
            inputs=f"dict(zip({tuple(cargs)!r}, ("+", ".join(ins)+",)))"
            if callee in SKIP_UNSELECTED:
                emit(f"call_out = masked_call(xp.logical_and({mask if active is None else active}, alive), {callee}, {inputs}, xp=xp, c=c)")
            else:emit(f"call_out = {callee}({inputs}, xp=xp, c=c)")
            for key,arg in zip(cargs,node.args):
                if key not in MODIFIED[callee]:continue
                if isinstance(arg,ast.Call) and arg.func.id in global_arrays and arg.func.id not in local_arrays:
                    continue # Original global aerosol tables are read-only inputs.
                assign(ast.unparse(arg),f"call_out[{key!r}]",n,active)
        elif s=='return':
            emit(f"alive = xp.logical_and(alive, xp.logical_not({mask if active is None else active}))")
        else:
            assert '=' in s,(name,n,s)
            lhs,rhs=s.split("=",1);assign(lhs.strip(),expr(rhs),n,active)
    for n,s in body:
        if s in ('enddo','end do'):
            kind,mask,_=stack.pop();assert kind in ('for','level')
            if kind=='for':indent-=1
            continue
        if s.startswith('do '):
            match=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',s);assert match,(name,n,s)
            key,bounds=match[1],split(match[2]);assert len(bounds) in (2,3)
            start,stop=bounds[:2];step=bounds[2] if len(bounds)==3 else '1'
            if level and key=='k':
                stack.append(('level',mask,None));continue
            emit(f"for loop_{key} in range(int({expr(start)}), int({expr(stop)})+1, int({expr(step)})):")
            indent+=1;emit(f"v[{key!r}] = np.int32(loop_{key})");stack.append(('for',mask,None));continue
        if s in ("endif","end if"):
            kind,mask,_=stack.pop()
            if kind=='static':indent-=1
            continue
        if s.startswith("else"):
            kind,parent,taken=stack[-1]
            if kind=='static':
                indent-=1
                if s=='else':emit('else:')
                else:
                    match=re.fullmatch(r"else\s*if\s*\((.*)\)\s*then|elseif\s*\((.*)\)\s*then",s);assert match,(n,s)
                    condition=match[1] or match[2];assert is_static(condition),(name,n,s)
                    emit('elif '+expr(condition).replace('xp.','np.')+':')
                indent+=1;continue
            counter+=1;mask=f"m{counter}"
            if s=="else":
                emit(f"{mask} = xp.logical_and({parent}, xp.logical_not({taken}))")
            else:
                match=re.fullmatch(r"else\s*if\s*\((.*)\)\s*then|elseif\s*\((.*)\)\s*then",s);assert match,(n,s)
                cond=expr(match[1] or match[2])
                emit(f"{mask} = xp.logical_and(xp.logical_and({parent}, xp.logical_not({taken})), {cond})")
                emit(f"{taken} = xp.logical_or({taken}, {mask})")
            continue
        if re.match(r"if\s*\(",s):
            depth=0;end=None;start=s.index("(")
            for pos,ch in enumerate(s[start:],start):
                depth+=(ch=="(")-(ch==")")
                if depth==0:end=pos;break
            assert end is not None,(n,s)
            condition=s[start+1:end];cond=expr(condition);rest=s[end+1:].strip()
            if is_static(condition):
                emit('if '+cond.replace('xp.','np.')+':');indent+=1
                if rest=='then':stack.append(('static',mask,None))
                else:action(rest,n);indent-=1
                continue
            counter+=1;child=f"m{counter}";emit(f"{child} = xp.logical_and({mask}, {cond})")
            if rest=="then":
                taken=f"t{counter}";emit(f"{taken} = {child}");stack.append(("if",mask,taken));mask=child
            else:
                action(rest,n,child)
            continue
        action(s,n)
    assert not stack,name
    lines=[f"def {name}(v, *, xp=np, c=None"+(", active=True" if level else "")+"):", "    v = dict(v)",
           f"    ref = v[{args[0]!r}]", "    r = np.dtype(ref.dtype).type", "    c = constants(ref.dtype) if c is None else c", "    for key, value in c.items(): v.setdefault(key, value)",
           "    _max = lambda *x: maximum(*x, xp=xp)", "    _min = lambda *x: minimum(*x, xp=xp)",
           "    _int = lambda x: xp.asarray(x, dtype=np.int32)",
           "    _nint = lambda x: nint(x, xp=xp)", "    _double = lambda x: xp.asarray(x, dtype=np.float64)",
           "    _max0 = lambda x: xp.fmax(r(0), x)", "    _real = lambda x: xp.asarray(x, dtype=ref.dtype)",
           "    _erf = lambda x: erf(x, xp=xp)","    _get_local = lambda x, i: get_local(x, i, xp=xp)", "    alive = True"]
    for key,kind in types.items():
        if key not in args:
            zero="False" if kind=="logical" else "np.int32(0)" if kind=="integer" else "np.float64(0)" if kind=="double precision" else "r(0)"
            if level and key not in static_indices:
                dtype="bool" if kind=='logical' else "np.int32" if kind=='integer' else 'ref.dtype'
                extra=', '.join(f"extent(int({expr(x)}))" for x in extents.get(key,()))
                shape="ref.shape"+(f" + ({extra},)" if extra else '')
                zero=f"xp.zeros({shape}, dtype={dtype})"
            lines.append(f"    if {key!r} not in v: v[{key!r}] = {zero}" if level else f"    v.setdefault({key!r}, {zero})")
    for n,k,x in params:
        lines.append(f"    v[{k!r}] = {expr(x)}  # WRF:{n}")
    returned=sorted(set(types)-static_indices) if level else args
    lines+=output+["    return {"+", ".join(f"{k!r}: v[{k!r}]" for k in returned)+"}",""]
    return "\n".join(lines)


def main():
    source=SOURCE.read_text();all_=list(statements(source))
    blocks={};current=None
    for n,s in all_:
        if s.startswith("subroutine "):
            name=s.split()[1].split("(")[0];blocks[name]=[(n,s)];current=name
        elif s.startswith("end subroutine"):
            current=None
        elif current:
            blocks[current].append((n,s))
    signatures={name:split(blocks[name][0][1].split("(",1)[1].rsplit(")",1)[0]) for name in ROUTINES+AEROSOL_ROUTINES}
    # Preserve Fortran by-reference mutation without writing unchanged arguments
    # back into the caller. In particular IAE/short-loop indices stay static.
    modified={name:set() for name in signatures}
    calls=[]
    for name in signatures:
        for _,s in declarations(blocks[name][1:])[-1]:
            if re.match(r'if\s*\(',s) and not s.endswith('then'):
                depth=0
                for i,ch in enumerate(s[s.index('('):],s.index('(')):
                    depth+=(ch=='(')-(ch==')')
                    if depth==0:s=s[i+1:].strip();break
            if re.match(r'^\w+(?:\([^)]*\))?\s*=',s):
                target=s.split('=')[0].strip().split('(')[0]
                if target in signatures[name]:modified[name].add(target)
            elif s.startswith('call '):
                call=parse(s[5:]);calls.append((name,call))
    changed=True
    while changed:
        changed=False
        for name,call in calls:
            callee=call.func.id
            outputs=[4] if callee=='find_rc0' else [i for i,k in enumerate(signatures[callee]) if k in modified[callee]]
            for i in outputs:
                arg=call.args[i];target=arg.id if isinstance(arg,ast.Name) else arg.func.id
                if target in signatures[name] and target not in modified[name]:modified[name].add(target);changed=True
    MODIFIED.update(modified)
    types,arrays,params,data,_=declarations([(n,s) for n,s in all_ if n<255])
    common=["\"\"\"Original NTU module constants, emitted offline; host-only dtype rounding.\"\"\"", "from functools import lru_cache", "import numpy as np", "from .support import _max, _min, _int, _get", "", "@lru_cache(maxsize=2)", "def constants(dtype):", "    r = np.dtype(dtype).type", "    xp = np", "    _max0 = lambda x: xp.fmax(r(0), x)", "    v = {}"]
    for n,k,x in params:
        common.append(f"    v[{k!r}] = {expression(x,arrays)}  # WRF:{n}")
    for n,k,values in data:
        common.append(f"    v[{k!r}] = np.asarray(["+", ".join(expression(x,arrays) for x in values)+f"], dtype=r)  # WRF:{n}")
        common.append(f"    v[{k!r}].flags.writeable = False")
    # AERO_CONST's CCNTY=2 REAL constants. General aerosol configuration is
    # supplied by the column wrapper; keeping it explicit avoids global mutation.
    common += ["    v.update(dict(zip(('dnc0','dnc1','dnc2','dnc3'), map(r, (-48575.888,26634.802,-4862.961,295.7886)))))", "    return v", ""]
    text='"""Literal NTU scalar routines; generated by proofs/s2_ntu/port_scalar.py.\nPristine source SHA256 '+hashlib.sha256(source.encode()).hexdigest()+'\n"""\nimport numpy as np\nfrom .constants import constants\nfrom .support import maximum, minimum, nint, _get, get_local, _set_local, get_column, put_value\nfrom ._native import extent\nfrom .math import gamln, gamin, gammp, polysvp, erf, dlnx, dlnx2, dpdf, find_rc0\n\n'
    for name in ROUTINES:
        text+=generate(name,blocks[name][0][1],blocks[name][1:],types,arrays,signatures)+"\n"
    aerosol=text[:text.index('def solve_afac')]
    for name in AEROSOL_ROUTINES:
        aerosol+=generate(name,blocks[name][0][1],blocks[name][1:],types,arrays,signatures)+"\n"
    micro=blocks['ntu_micro']
    decl=[(n,s) for n,s in micro[1:] if n<1853]
    phases=text[:text.index('def solve_afac')]+"from .support import masked_call\nfrom .scalar_generated import *\nfrom .aerosol_generated import *\n\n"
    ranges={'initialize':(1853,2199),'environment':(2201,2227),
            'reset_small':(2233,2243),'advance_small':(2246,2328),
            'after_small':(2334,2379),'coalescence':(2382,2407),
            'before_sedimentation':(2411,2565),'finish':(2607,2684)}
    for name,(first,last) in ranges.items():
        chosen=[(n,s) for n,s in micro[1:] if first<=n<=last]
        phases+=generate(name,'subroutine '+name+'(tk3d)',decl+chosen,types,arrays,signatures,level=True)+'\n'
    ast.parse(text);ast.parse("\n".join(common))
    OUT.mkdir(exist_ok=True)
    (OUT/"constants.py").write_text("\n".join(common))
    (OUT/"scalar_generated.py").write_text(text.rstrip()+"\n")
    (OUT/"aerosol_generated.py").write_text(aerosol.rstrip()+"\n")
    ast.parse(phases)
    (OUT/"phases_generated.py").write_text(phases.rstrip()+"\n")
    print(f"{len(ROUTINES)} routines -> {len(text.splitlines())} source-attributed lines")


if __name__=="__main__":main()
