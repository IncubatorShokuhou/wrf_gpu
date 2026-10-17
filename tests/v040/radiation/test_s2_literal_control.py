"""Proof evaluator must skip inactive statements and retain active bounds errors."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('active,index,expected',[(True,0,1.),(False,1,0.),(True,1,None)])
def test_scalar_statement_control(tmp_path,active,index,expected):
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('literal_control',root/'proofs/s2_flggfdl/literal.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source=tmp_path/'control_fixture.py'
    source.write_text('''import jax.numpy as jnp
def evaluate(active,index):
    values=jnp.zeros((1,),dtype=jnp.float32)
    active=active & (values[index]>=0.)
    values=values.at[index].set(jnp.where(active,values[index]+1.,values[index]))
    return values
''')
    evaluator=helper.load(source,source_control=True)
    if expected is None:
        with pytest.raises(IndexError):evaluator.evaluate(active,index)
    else:
        value=evaluator.evaluate(active,index)
        assert value.dtype==np.float32 and np.array_equal(value,np.asarray([expected],np.float32))
