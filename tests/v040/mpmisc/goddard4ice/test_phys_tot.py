"""Only the added sum is executed; original microphysics science is reused."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent.parent
FIELDS=('physc','physe','physd','physs','physm','physf')


def test_original_phys_tot_literal_association_byte():
    reference=HERE/'goddard4ice_phys_tot.npz'
    metadata=json.loads(reference.with_suffix('.json').read_text())
    original=HERE/'goddard4ice_columns.npz'
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==metadata['reference_sha256']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==metadata['original_operand_fixture_sha256']
    assert metadata['byte_operands']==4320 and metadata['byte_phys_tot_values']==720
    tree=ast.parse((ROOT/'src/gpuwrf/physics/goddard4ice/column.py').read_text())
    assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name)
                         and t.value.id=='result' and isinstance(t.slice,ast.Constant)
                         and t.slice.value=='phys_tot' for t in node.targets)]
    assert len(assignments)==1, 'Original PHYS_TOT history output is missing'
    expression=compile(ast.Expression(assignments[0].value),'phys_tot_output','eval')
    # Independently compiled Fortran executes the verbatim original driver
    # assignment on already oracle-qualified six REAL4 rate operands. Test
    # only the new output expression, without rerunning the physical column.
    with np.load(reference,allow_pickle=False) as truth,np.load(original,allow_pickle=False) as operands:
        assert len(truth.files)==18
        for key in truth.files:
            prefix=key[:-len('phys_tot')]
            values={name:operands[prefix+name] for name in FIELDS}
            actual=eval(expression,{},dict(outputs=values))
            assert actual.dtype==np.float32 and np.isfinite(actual).all(),key
            assert actual.shape==truth[key].shape and actual.tobytes()==truth[key].tobytes(),key
