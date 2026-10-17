"""Load the manager-approved, immutable WRF-operand R4 budget packet."""
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


def finite_reference(data, context='reference'):
    """Reject corrupted truth, including the REAL8 rounding-spread operands."""
    if isinstance(data, Mapping):
        for name,value in data.items():finite_reference(value,context+'.'+str(name))
    else:
        value=np.asarray(data)
        if value.dtype.kind in 'bifcu':
            assert np.isfinite(value).all(),f'nonfinite {context}'


def comparison(actual,reference,raw_limit,extra=0.,*,exact=False):
    """Unchanged gates and raw counts, after all operands/bounds are admitted."""
    for name,value in [('candidate',actual),('reference',reference),
                       ('raw limit',raw_limit),('conditioning term',extra)]:
        assert np.isfinite(value).all(),f'nonfinite {name}'
    assert np.all(np.asarray(raw_limit)>=0),'negative raw limit'
    assert np.all(np.asarray(extra)>=0),'negative conditioning term'
    limit=raw_limit+extra
    assert np.isfinite(limit).all(),'nonfinite final limit'
    assert np.all(np.asarray(limit)>=0),'negative final limit'
    error=np.abs(actual-reference)
    assert np.isfinite(error).all(),'nonfinite comparison error'
    if exact:
        ratio=error  # KPBL has a valid zero bound; never divide by that bound.
    else:
        assert np.all(np.asarray(limit)>0),'zero non-exact limit'
        ratio=error/limit
    assert np.isfinite(ratio).all(),'nonfinite comparison ratio'
    return dict(max_abs=float(error.max()),max_ratio=float(ratio.max()),
                raw_violations=int(np.sum(error>raw_limit)),
                violations=int(np.sum(error>limit)))


def load_budget(path):
    if path is None:return {}
    path=Path(path)
    sealed=Path(__file__).resolve().parent/'evidence/ROUND01_APPROVAL.json'
    approval=json.loads(sealed.read_text())
    assert approval['decision']=='APPROVED'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==approval['budget_npz_sha256']
    values={name:value for name,value in np.load(path).items()}
    finite_reference(values,'conditioning terms')
    assert all(np.all(value>=0) for value in values.values()),'negative conditioning term'
    return values


def column_budget(budgets,arm,field,step,actual_shape,reference_records):
    key=arm+'__'+field
    if key not in budgets:return 0.
    index=step if reference_records==7 else step-1
    result=budgets[key][:,index]
    if result.shape!=actual_shape and result.ndim==2 and result.shape[-1]==9 and actual_shape[-1]==8:
        result=result[:,:8]
    assert result.shape==actual_shape,(key,result.shape,actual_shape)
    return result
