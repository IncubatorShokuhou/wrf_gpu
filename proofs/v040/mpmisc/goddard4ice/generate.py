"""Restricted, fail-closed transcription of pristine GSFC 4-ice equations.

Only this one source is supported. Column-local saticel levels are vectorized;
Fortran IF ownership becomes explicit masks, with assignment REAL rounding.
The generator is a development tool, never imported by the forecast.
"""
import argparse
import ast
import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_gsfcgce_4ice_nuwrf.F')
OUT = ROOT / 'src/gpuwrf/physics/goddard4ice'

def statements(text):
    pending = ''
    start = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split('!')[0].strip().lower()
        if not line or line.startswith('#'):
            continue
        if not pending:
            start = lineno
        pending += ' ' + line.strip('&').strip()
        if line.endswith('&'):
            continue
        for value in pending.strip().split(';'):
            if value.strip():
                yield start, value.strip()
        pending = ''
    assert not pending

def split(text):
    result, pos, depth = [], 0, 0
    for i, c in enumerate(text):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            result.append(text[pos:i].strip()); pos = i + 1
    result.append(text[pos:].strip())
    return result

TOKENS = re.compile(r'\.[a-z]+\.|(?:\d+\.?\d*|\.\d+)(?:[ed][+-]?\d+)?|[a-z_]\w*|\*\*|<=|>=|==|/=|[()+*/,:<>=-]')
PRECEDENCE = {'.or.':1, '.and.':2, '.eq.':3, '.ne.':3, '.lt.':3, '.gt.':3, '.le.':3, '.ge.':3, '==':3, '/=':3, '<':3, '>':3, '<=':3, '>=':3, '+':4, '-':4, '*':5, '/':5, '**':7}
OPS = {'.or.':'|', '.and.':'&', '.eq.':'==', '.ne.':'!=', '.lt.':'<', '.gt.':'>', '.le.':'<=', '.ge.':'>=', '/=':'!='}
INTRINSICS = {'sqrt':'sqrt', 'exp':'exp', 'log':'log', 'alog':'log', 'alog10':'log10', 'acos':'arccos', 'atan':'arctan', 'abs':'abs', 'sin':'sin'}
RESERVED = {'as':'as_', 'lambda':'lambda_', 'del':'del_'}
def field(name):
    return 'w.' + RESERVED.get(name,name)

class Expression:
    def __init__(self, text, tables=(), vector=False):
        text = re.sub(r'(\.[a-z]+\.)', r' \1 ', text)
        self.tokens = TOKENS.findall(text)
        compact = re.sub(r'\s+','',text)
        assert ''.join(self.tokens) == compact, (text, self.tokens)
        self.at = 0; self.tables = tables; self.vector=vector
    def pop(self):
        value=self.tokens[self.at];self.at+=1;return value
    def parse(self, minimum=0):
        tok=self.pop()
        if tok == '(':
            value=self.parse(); assert self.pop()==')'
        elif tok in ('-', '+', '.not.'):
            operand=self.parse(6)
            value=(operand[:2]+'-'+operand[2:] if tok=='-' and operand.startswith(('F(','D(')) and re.fullmatch(r'[FD]\([.0-9ed+\-]+\)',operand) else f'({"~" if tok==".not." else tok}{operand})')
        elif tok in ('.true.','.false.'):
            value='True' if tok=='.true.' else 'False'
        elif re.match(r'\d|\.',tok):
            value = ('D(' + tok.replace('d','e') + ')' if 'd' in tok else 'F(' + tok + ')' if '.' in tok or 'e' in tok else tok)
        else:
            if self.at<len(self.tokens) and self.tokens[self.at]=='(':
                self.pop(); args=[]
                if self.tokens[self.at]!=')':
                    while True:
                        if self.tokens[self.at]==':':self.pop();args.append(':')
                        else:args.append(self.parse())
                        if self.tokens[self.at]!=',':break
                        self.pop()
                assert self.pop()==')'
                if tok in INTRINSICS:value=f'jnp.{INTRINSICS[tok]}('+', '.join(args)+')'
                elif tok in ('max','max0','amax1','min','min0','amin1'):
                    fn='maximum' if 'max' in tok else 'minimum'
                    value=args[0]
                    for arg in args[1:]:value=f'jnp.{fn}({value}, {arg})'
                elif tok in ('int','ifix'):value=f'jnp.asarray({args[0]}, jnp.int32)'
                elif tok=='real':value=f'F({args[0]})'
                elif tok=='dble':value=f'D({args[0]})'
                elif tok=='mod':value=f'({args[0]} % {args[1]})'
                elif tok=='gammagce':value=f'gammagce({args[0]})'
                elif tok=='gamma_toshi':value=f'gamma_toshi({args[0]})'
                elif tok=='eff_rad':value=f'eff_rad({args[0]})'
                elif tok=='present':value='True'
                elif tok=='wrf_dm_on_monitor':value='False'
                elif tok in self.tables:
                    value=f'table_lookup("{tok}", {args[0]})'
                elif self.vector:
                    if tok=='rho' and args==[field('i'),'1']:value=field('rho_first')
                    elif tok=='ics': value=field('ics'+str(args[-1]))
                    elif len(args)==2 and args[1]=='(w.k + 1)':value=f'shift_up({field(tok)}, "{tok}")'
                    else:value=field(tok)
                else:
                    value=field(tok)
            else:value=field(tok)
        while self.at<len(self.tokens):
            op=self.tokens[self.at]; prec=PRECEDENCE.get(op,-1)
            if prec<minimum:break
            self.pop();right=self.parse(prec if op=='**' else prec+1)
            value=(f'ipow({value}, {right})' if op=='**' and right.isdigit() else f'({value} {OPS.get(op,op)} {right})')
        return value
    def result(self):
        value=self.parse();assert self.at==len(self.tokens),(self.tokens[self.at:],value);return value

def condition(line):
    pos=line.index('('); depth=1;end=pos+1
    while depth:
        if line[end]=='(':depth+=1
        if line[end]==')':depth-=1
        end+=1
    return line[pos+1:end-1],line[end:].strip()

TABLES={'aa1','aa2','rn12a','rn12b','rn13','rn25a','bergcon1','bergcon2','bergcon3','bergcon4'}

def generate_function(name, body, inputs, output, vector=False):
    lines=[f'def {name}('+', '.join(inputs)+(', diagnostics=None, _physical_nz=None' if name=='saticel_s' else '')+'):', '    w = _Work(**C)']
    for var in inputs:
        lines.append(f'    {field(var)} = {var}')
    if name=='saticel_s':
        lines += ['    w.k = jnp.arange(1, ptwrf.shape[0] + 1)', '    w.kte = ptwrf.shape[0] if _physical_nz is None else _physical_nz', '    w.kts = 1', '    w.rho_first = first_density(rho_mks) * F(0.001)', '    if diagnostics is not None:', '        w.values.update(diagnostics)']
    masks=['True']; branch=[]; counter=0
    skip_depth=0
    declarations=True
    for number,line in body:
        if re.match(r'^(implicit|integer|logical|real|character|dimension|data|save|use)\b',line):
            if 'parameter' in line and '::' in line:
                for decl in split(line.split('::',1)[1]):
                    lhs,rhs=decl.split('=',1); lines.append(f'    {field(lhs.strip())} = {Expression(rhs,TABLES,vector).result()}')
            continue
        if re.match(r'^(print|write|call wrf_error_fatal)',line):continue
        if line in ('end subroutine '+name,'end function '+name):continue
        if line.startswith(('do ', 'do\t')) or line=='do' or line.startswith(('j_loop:', 'i_loop:')):
            if not vector and not name=='consat_s':raise ValueError((name,number,line))
            continue
        if line.startswith(('enddo','end do')) or re.match(r'\d+ continue$',line):continue
        if line.startswith(('endif','end if')):
            masks.pop();branch.pop();continue
        iselse=bool(re.match(r'else\s*if\b|elseif\b',line))
        if line=='else':
            masks[-1]=f'({masks[-2]} & ~{branch[-1]})';continue
        if line.startswith(('if ','if(')) or iselse:
            expr,rest=condition(line)
            pred=Expression(expr,TABLES,vector).result()
            if rest=='then':
                counter+=1;label=f'm{counter}';lines.append(f'    {label} = {pred}  # WRF:{number}')
                if iselse:
                    masks[-1]=f'({masks[-2]} & ~{branch[-1]} & {label})'
                    branch[-1]=f'({branch[-1]} | {label})'
                else:
                    masks.append(f'({masks[-1]} & {label})');branch.append(label)
                continue
            assert not iselse,(number,line)
            active=f'({masks[-1]} & {pred})';line=rest
        else:active=masks[-1]
        if line.startswith('call '):
            fname=line[5:line.index('(')].strip(); args=split(line[line.index('(')+1:-1])
            assert fname in ('sgmap','vqrqi'),(number,line)
            lhs=args[-1]; rhs=fname+'('+', '.join(Expression(a,TABLES,vector).result() for a in args[:-1])+')'
        else:
            assert '=' in line and not line.startswith('return'),(name,number,line)
            lhs,rhs=line.split('=',1)
            rhs=Expression(rhs,TABLES,vector).result()
        lhs=lhs.strip();var=lhs.split('(')[0].strip()
        if var=='ics':var='ics'+lhs[lhs.rindex(',')+1:-1].strip()
        if name=='saticel_s' and number==4000:
            # Keep the original REAL quotient rounding before SSI's sign test.
            rhs='(_saturation_ratio(w.qv + w.qb0, w.qsi) - F(1.))'
        lines.append(f'    w.put({active}, {RESERVED.get(var,var)}={rhs})  # WRF:{number}')
    assert len(masks)==1,(name,masks)
    lines.append('    return '+output)
    return '\n'.join(lines)+'\n'


def preprocess(source):
    result=[];stack=[];active=True
    for line in source.splitlines():
        stripped=line.strip()
        if stripped.startswith('#if'):
            stack.append((active,False));active=False
            result.append('');continue
        if stripped.startswith('#else'):
            outer,old=stack[-1];active=outer and not old;stack[-1]=(outer,True)
            result.append('');continue
        if stripped.startswith('#endif'):
            active=stack.pop()[0];result.append('');continue
        result.append(line if active else '')
    assert not stack
    return '\n'.join(result)


def main():
    source=SOURCE.read_text()
    expected='79b782f6832ce45c0a13f463841d1820cabc4de0049ac60b5c2a23d24389cb8b'
    if hashlib.sha256(source.encode()).hexdigest()!=expected:
        raise ValueError('Pristine Goddard4ice source changed; re-audit before regeneration')
    processed=preprocess(source)
    items=list(statements(processed))
    blocks={}
    for i,(n,line) in enumerate(items):
        match=re.match(r'(?:real )?(?:subroutine|function) (\w+)\s*\(',line)
        if match:
            name=match[1]
            for end in range(i+1,len(items)):
                if items[end][1].startswith(('end subroutine','end function')):break
            blocks[name]=items[i+1:end]
    header='''# Generated by proofs/v040/mpmisc/goddard4ice/generate.py.
# All equations originate in pristine module_mp_gsfcgce_4ice_nuwrf.F.
# Source sha256: '''+hashlib.sha256(source.encode()).hexdigest()+'''
from .support import C, F, D, _Work, gammagce, gamma_toshi, eff_rad, ipow, table_lookup, shift_up, first_density
import jax.numpy as jnp

def _saturation_ratio(numerator, denominator):
    """WRF REAL division, rounded before SSI's sign test (WRF:4000)."""
    return numerator / denominator

'''
    allcode=header
    allcode+=generate_function('sgmap',blocks['sgmap'],['isg','qcs','qcg','qcg2','qch','qch2','r00','tairc'],'w.ftnsg',True)+'\n'
    allcode+=generate_function('vqrqi',blocks['vqrqi'],['isg','r00','fv','qri','ql','tair'],'w.ww1',True)+'\n'
    inputs=['dt','dx','ptwrf','qvwrf','qlwrf','qrwrf','qiwrf','qswrf','qgwrf','qhwrf','rho_mks','pi_mks','p0_mks','w_mks','xland','itimestep']
    outfields=dict(zip(['theta','qv','qc','qr','qi','qs','qg','qh'],['ptwrf','qvwrf','qlwrf','qrwrf','qiwrf','qswrf','qgwrf','qhwrf']))
    for key in ['re_cloud_gsfc','re_rain_gsfc','re_ice_gsfc','re_snow_gsfc','re_graupel_gsfc','re_hail_gsfc'] + ['phys'+x for x in 'cedsmf'] + ['acphys'+x for x in 'cedsmf']:outfields[key]=key
    out='{'+', '.join(repr(k)+': '+field(v) for k,v in outfields.items())+'}'
    allcode+=generate_function('saticel_s',blocks['saticel_s'],inputs,out,True)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'equations.py').write_text(allcode)
    # Scalar/31-element constant setup remains a proof-time host calculation.
    constants=generate_function('consat_s',blocks['consat_s'],['itaobraun'],'w.values',True)
    (ROOT/'proofs/v040/mpmisc/goddard4ice/constant_equations.py').write_text(constants)
    print('generated',len(allcode.splitlines()),'lines')
if __name__=='__main__':main()
