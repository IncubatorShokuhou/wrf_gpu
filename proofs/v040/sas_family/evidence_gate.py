"""Finite, counted evidence comparisons. No tolerance policy lives here."""
import numpy as np


def require_finite(**operands):
    bad = {name: int(np.count_nonzero(~np.isfinite(value)))
           for name, value in operands.items()}
    bad = {name: count for name, count in bad.items() if count}
    if bad:
        raise ValueError(f'nonfinite evidence operands; failed_values={sum(bad.values())}: {bad}')


def score_field(actual, reference, bound_factory, *, operands=()):
    """Validate operands before evaluating a bound, and count every bad cell.

    A corrupt auxiliary operand invalidates the field rather than allowing a
    derived infinite bound to excuse corrupt evidence. Invalid diagnostics use
    None, never NaN/Inf or a misleading zero failure count.
    """
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape:
        return dict(pass_=False, failed_values=max(1, a.size, b.size), bad_values=[],
                    max_abs=None, byte_fraction=None, reason='shape mismatch')
    invalid = ~np.isfinite(a) | ~np.isfinite(b)
    auxiliary_bad = sum(int(np.count_nonzero(~np.isfinite(x))) for x in operands)
    if auxiliary_bad:
        invalid = np.ones(a.shape, dtype=bool)
    if np.any(invalid):
        return dict(pass_=False, failed_values=max(1, int(invalid.sum())),
                    bad_values=np.argwhere(invalid).tolist(), max_abs=None,
                    byte_fraction=None, reason='nonfinite candidate/reference/operand')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        limit = np.broadcast_to(np.asarray(bound_factory()), a.shape)
        delta = np.abs(a.astype(np.float64)-b)
    invalid = ~np.isfinite(limit) | (limit < 0.) | ~np.isfinite(delta)
    bad = invalid | (delta > limit)
    return dict(pass_=not bool(bad.any()), failed_values=int(bad.sum()),
                bad_values=np.argwhere(bad).tolist(),
                max_abs=None if invalid.any() else float(delta.max()),
                byte_fraction=float(np.mean(a == b)),
                reason='nonfinite/negative derived bound or difference' if invalid.any() else '')


def field_score(*args, **kwargs):
    result = score_field(*args, **kwargs)
    result['pass'] = result.pop('pass_')
    return result


def promoted_fields(actual, expected, *, exact=()):
    results = {}
    for name, reference in expected.items():
        b = np.asarray(reference)
        def bound():
            if name in exact:
                return np.zeros_like(b)
            scale = np.max(np.abs(b), axis=1, keepdims=True) if b.ndim == 2 else np.abs(b)
            return 1.e-9*scale
        results[name] = field_score(actual[name], b, bound)
    return results
