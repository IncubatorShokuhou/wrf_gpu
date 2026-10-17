"""Host restart contracts for independent WRF ALB and packed up-feedback.

No timestep work or new carry leaves. The native identity binds the complete
initialized field list; the portable formats retain that list in their header.
"""
import numpy as np

VERSION = 1
REQUIRED_FIELDS = ('TH2', 'SWDNB', 'SWUPB', 'SWDNT', 'SWUPT',
    'LWDNB', 'LWUPB', 'LWDNT', 'LWUPT', 'SWDNBC', 'SWUPBC',
    'SWDNTC', 'SWUPTC', 'LWDNBC', 'LWUPBC', 'LWDNTC', 'LWUPTC')


def configuration(base, history, volume_shape, surface_shape, *, required=False):
    """Freeze the initialized schema, refusing incomplete active groups."""
    alb = getattr(base, 'alb', None)
    if alb is None and history is None and not required:
        return None  # Legacy/default inactive restarts gain no metadata.
    names = getattr(history, 'names', ())
    config = dict(version=VERSION, active=True, volume_shape=list(volume_shape),
                  surface_shape=list(surface_shape), history_fields=list(names))
    validate(base, history, volume_shape, surface_shape, config, required=True)
    return config


def validate(base, history, volume_shape, surface_shape, config, *, required=False, expected=None):
    """Check names/shapes/REAL storage before device restoration.

    Only array metadata is inspected here. Existing checkpoint boundary guards
    handle finiteness, without introducing a second full host copy or GPU read.
    """
    alb = getattr(base, 'alb', None)
    if config is None:
        if required or expected is not None or alb is not None or history is not None:
            raise ValueError('two-way restart lacks ALB/feedback history schema (E78)')
        return
    keys = {'version', 'active', 'volume_shape', 'surface_shape', 'history_fields'}
    if (not isinstance(config, dict) or set(config) != keys
            or type(config['version']) is not int or config['version'] != VERSION
            or config['active'] is not True):
        raise ValueError('two-way restart schema/version/selection differs (E78)')
    if expected is not None and config != expected:
        raise ValueError('two-way restart initialized field schema differs (E78)')
    for key, shape, rank in (('volume_shape', volume_shape, 3), ('surface_shape', surface_shape, 2)):
        recorded = config[key]
        if (not isinstance(recorded, list) or len(recorded) != rank
                or any(type(n) is not int or n <= 0 for n in recorded)
                or tuple(recorded) != tuple(shape)):
            raise ValueError(f'two-way restart {key} differs from State (E78)')
    from gpuwrf.runtime.history_accumulators import PackedFields
    from gpuwrf.coupling.feedback_registry import FEEDBACK_FIELDS
    names = config['history_fields']
    if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or not set(REQUIRED_FIELDS) <= set(names)
            or any(name not in FEEDBACK_FIELDS for name in names)):
        raise ValueError('two-way restart required initialized history fields differ (E78)')
    if (alb is None or tuple(getattr(alb, 'shape', ())) != tuple(volume_shape)
            or getattr(alb, 'dtype', None) != np.dtype('float32')):
        raise ValueError('two-way restart ALB must have mass-volume shape and WRF REAL dtype (E78)')
    if (not isinstance(history, PackedFields) or history.names != tuple(names)
            or tuple(getattr(history.data, 'shape', ())) != (len(names), *surface_shape)
            or getattr(history.data, 'dtype', None) != np.dtype('float32')):
        raise ValueError('two-way restart packed history names/shape/WRF REAL dtype differ (E78)')


def validate_payload(payload, *, required=False, expected=None):
    extras, state = payload.get('extra_fields', {}), payload['state_fields']
    validate(extras.get('base_state'), extras.get('feedback_history'),
             state['qv'].shape, state['t_skin'].shape, payload.get('feedback_schema'),
             required=required, expected=expected)
