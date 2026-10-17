"""Offline GFDL LWR88/FST88 family transcription; native qualification required."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from transcribe_gfdl import Emitter,read_routine,base

ROUTINES=('lwr88','fst88','e1e290','spa88','e290','e2spec','e3v88')
CO2=('co251','cdt51','cdt58','c2d51','c2d58','co258','stemp','gtemp',
     'co231','co238','c2d31','c2d38','cdt31','cdt38','co271','co278',
     'c2d71','c2d78','cdt71','cdt78','co2m51','co2m58','cdtm51','cdtm58',
     'c2dm51','c2dm58')
FLAT={'t1':'table1','t2':'table2','t4':'table3','em1v':'em1','em1vw':'em1wde','em3v':'em3'}
TABLES=('source','dsrce','ab15wd','sko2d','skc1r','sko3r')


def main(source,output,receipt):
    base.MODULE['nbly']=('i',None,'15')
    # Truthy shape marks a table operand, including a scalar DATA value. The
    # exact shapes and native REAL bytes are checked on the oracle provider.
    for name in (*CO2,*FLAT,*TABLES):base.MODULE[name]=('r',('table',),None)
    records=[];rendered=[]
    for name in ROUTINES:
        routine=read_routine(source,name)
        text=Emitter(routine).render()
        unresolved=routine.variables-set(routine.types)-set(routine.params)-set(base.MODULE)
        if unresolved:raise ValueError((name,'unresolved native module variables',sorted(unresolved)))
        ast.parse(text)
        rendered.append(text)
        records.append(dict(name=name,arguments=routine.args,shapes=routine.shapes,types=routine.types,
                            source_statements=len(list(base.descend(routine.body)))))
    header='''"""Generated GFDL LWR88 family column reference, qualification pending.
Regenerate with proofs/s2_flggfdl/transcribe_gfdl_lw.py. No runtime source parsing.
Native source SHA256: SOURCE_SHA
The CO2 provider is grid-dependent; native fixture tables are oracle inputs,
not an operational substitute for generic CO2 initialization.
"""
import numpy as np
import jax.numpy as jnp
from jax import lax
from gpuwrf.physics.ra_gfdleta import _host_tables

def _min(*args):
    value=args[0]
    for x in args[1:]:value=jnp.minimum(value,x)
    return value

def _max(*args):
    value=args[0]
    for x in args[1:]:value=jnp.maximum(value,x)
    return value

def _sign(a,b):return jnp.copysign(jnp.abs(a),b)
def _nint(x):return jnp.where(x>=0.,jnp.floor(x+.5),jnp.ceil(x-.5)).astype(jnp.int32)

def reference_factory(nlayers,co2_tables):
    _NLAYERS=int(nlayers)
    if _NLAYERS<3:raise ValueError('WRF radiation needs at least three layers')
    _TABLES=dict(_host_tables())
    for key,value in co2_tables.items():
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('CO2 provider must contain finite host REAL4 tables, never traced values')
        _TABLES[key]=value
    for key,native in FLAT_ALIASES.items():_TABLES[key]=_TABLES[native].T.reshape(-1)
    for key in REQUIRED_CO2:
        if key not in co2_tables:raise ValueError('Missing native CO2 table '+key)
'''.replace('SOURCE_SHA',hashlib.sha256(source.read_bytes()).hexdigest())
    header=header.replace('FLAT_ALIASES',repr(FLAT)).replace('REQUIRED_CO2',repr(CO2))
    generated=header+'\n\n'.join('\n'.join('    '+line for line in text.splitlines()) for text in rendered)+'\n    return _lwr88\n'
    ast.parse(generated);output.write_text(generated)
    receipt.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),routines=records,
        generated_lines=len(generated.splitlines()),required_co2_tables=CO2,
        scope='offline source transcription only; native table provider/oracle+compiled qualification pending'),indent=2)+'\n')
    print('LWR88 family emitted',len(generated.splitlines()),'lines; qualification pending')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','receipt'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.source,a.output,a.receipt)
