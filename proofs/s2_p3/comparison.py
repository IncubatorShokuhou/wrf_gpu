"""Fail-closed comparisons for both P3 proof CLIs (never model code)."""
import numpy as np

from generate_columns import FIELDS

RTOL = 32 * np.finfo(np.float32).eps
SURFACE_FIELDS = ('rainnc', 'rainncv', 'sr', 'snownc', 'snowncv')
FIXTURE_FIELDS = (*FIELDS, *SURFACE_FIELDS)
SECOND_ICE = ('qi2', 'ni2', 'qir2', 'bir2', 'vm2', 'di2', 'rhoi2')


def tolerance(name):
    if name in ('th', 'th_old'):
        return 0., 3.e-4
    if name in ('nc', 'ni', 'ni2', 'nr'):
        return RTOL, 1.e-3
    if name in ('qv', 'qv_old', 'qc', 'qr', 'qi', 'qi2', 'qir', 'qir2'):
        return RTOL, 2.e-11
    if name in ('bir', 'bir2', 'qzi'):
        return RTOL, 1.e-19
    if name.startswith('rhoi'):
        return RTOL, 2.e-3
    if name in ('ze',):
        return 0., 3.e-4
    if name in ('rainncv','snowncv','sr'):
        return RTOL, 2.e-6
    return RTOL, 2.e-10 if name.startswith(('re_', 'di')) else 1.e-6


class ComparisonError(ValueError):
    def __init__(self, message, count=1, nonfinite=0):
        super().__init__(message)
        self.count = count
        self.nonfinite = nonfinite


def checked_array(value, label, shape, dtype=np.dtype('float32')):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != dtype:
        raise ComparisonError(f'{label}: expected {shape}/{dtype}, got {value.shape}/{value.dtype}')
    count = int(np.count_nonzero(~np.isfinite(value)))
    if count:
        raise ComparisonError(f'{label}: {count} non-finite values', count, count)
    return value


def output_fields(mp):
    excluded = {'p', 'exner', 'dz', 'w', 'rainnc', 'snownc'}
    if mp != 52:
        excluded.update(SECOND_ICE)
    if mp != 53:
        excluded.add('qzi')
    return tuple(name for name in FIXTURE_FIELDS if name not in excluded)


def checked_inventory(arrays, label, *, mp=None, reference=None):
    """Check *all* fields, including held inputs and unused category slots."""
    expected = set(FIXTURE_FIELDS if mp is None else output_fields(mp))
    if set(arrays) != expected:
        missing, extra = expected - set(arrays), set(arrays) - expected
        raise ComparisonError(f'{label}: missing={sorted(missing)}, extra={sorted(extra)}',
                              len(missing) + len(extra))
    for name in sorted(expected):
        if reference is None:
            # Frozen original-wrapper corpus: 8 columns, 12 levels, frames 0..3.
            shape = (8, 4) if name in SURFACE_FIELDS else (8, 12, 4)
        else:
            shape = reference[name].shape[:-1]
        checked_array(arrays[name], f'{label}.{name}', shape)


def checked_limit(value, label):
    value = np.asarray(value)
    count = int(np.count_nonzero(~np.isfinite(value)))
    if count:
        raise ComparisonError(f'{label}: {count} non-finite bound values', count, count)
    count = int(np.count_nonzero(value < 0))
    if count:
        raise ComparisonError(f'{label}: {count} negative bound values', count)
    return value


def compare_field(name, value, reference, *, limit=None, literal=False):
    """Finiteness and structure precede subtraction, equality and reductions."""
    reference = np.asarray(reference)
    checked_array(reference, f'reference.{name}', reference.shape)
    value = checked_array(value, f'candidate.{name}', reference.shape, reference.dtype)
    rtol, atol = tolerance(name)
    ref = reference if literal else reference.astype(np.float64)
    strict_limit = checked_limit(atol + rtol * np.abs(ref), f'{name}.strict_limit')
    limit = strict_limit if limit is None else checked_limit(limit, f'{name}.limit')
    if limit.shape != reference.shape:
        raise ComparisonError(f'{name}.limit: shape {limit.shape} != {reference.shape}')
    error = np.abs(value.astype(np.float64) - reference.astype(np.float64))
    return dict(finite=True, max_abs=float(error.max()),
                bad=int(np.count_nonzero(error > limit)),
                strict_bad=int(np.count_nonzero(error > strict_limit)),
                different=int(np.count_nonzero(value != reference)),
                BYTE=value.tobytes() == reference.tobytes(), n=reference.size,
                rtol=float(rtol), atol=float(atol))
