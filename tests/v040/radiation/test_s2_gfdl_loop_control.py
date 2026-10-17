"""Native labelled DO-terminal semantics used throughout RADTN cloud loops."""
import importlib.util
from pathlib import Path
import sys
import numpy as np


def test_goto_terminal_continues_next_fortran_iteration(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root/'proofs/s2_flggfdl'))
    from transcribe_gfdl import read_routine,Emitter
    from literal import load,_wrap
    source=tmp_path/'loop.F90'
    source.write_text('''subroutine loop_goto(values)
real,intent(out)::values(3)
integer::k
values=0.
do 255 k=1,3
if(k.eq.1)go to 255
values(k)=real(k)
255 continue
end subroutine loop_goto
''')
    routine=read_routine(source,'loop_goto')
    generated=tmp_path/'loop_reference.py'
    generated.write_text('import jax.numpy as jnp\nfrom jax import lax\n'+Emitter(routine).render()+'\n')
    evaluator=load(generated,source_control=True)
    result=evaluator._loop_goto(_wrap(np.zeros(3,np.float32)))[0]
    # A jump to the terminal skips the current store, then the native DO
    # control advances K. It must not disable every subsequent iteration.
    assert result.dtype==np.float32
    assert np.array_equal(result,np.asarray([0.,2.,3.],np.float32))
