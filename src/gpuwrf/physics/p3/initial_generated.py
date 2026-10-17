"""Generated literal P3 level-local processes; see proofs/s2_p3/port_processes.py."""
from .process_support import *  # noqa: F403


@_with_double
def initial(v, tables, *, ncat, log_predictnc, log_3momentice, observer=None):
    v = dict(v)
    iparam = 3
    scpf_on = log_predictssat = False
    niter_mui = 5
    log_hmossopon = ncat > 1
    alive = v['level_active']
    level_alive = alive
    v.setdefault('acn', _initial(v['qv'], 'acn', ncat))
    v.setdefault('birim', _initial(v['qv'], 'birim', ncat))
    v.setdefault('dum', _initial(v['qv'], 'dum', ncat))
    v.setdefault('inv_rho', _initial(v['qv'], 'inv_rho', ncat))
    v.setdefault('log_hydrometeorspresent', _initial(v['qv'], 'log_hydrometeorspresent', ncat))
    v.setdefault('log_nucleationpossible', _initial(v['qv'], 'log_nucleationpossible', ncat))
    v.setdefault('nc', _initial(v['qv'], 'nc', ncat))
    v.setdefault('nitot', _initial(v['qv'], 'nitot', ncat))
    v.setdefault('nr', _initial(v['qv'], 'nr', ncat))
    v.setdefault('qc', _initial(v['qv'], 'qc', ncat))
    v.setdefault('qirim', _initial(v['qv'], 'qirim', ncat))
    v.setdefault('qitot', _initial(v['qv'], 'qitot', ncat))
    v.setdefault('qr', _initial(v['qv'], 'qr', ncat))
    v.setdefault('qv', _initial(v['qv'], 'qv', ncat))
    v.setdefault('qvi', _initial(v['qv'], 'qvi', ncat))
    v.setdefault('qvs', _initial(v['qv'], 'qvs', ncat))
    v.setdefault('rho', _initial(v['qv'], 'rho', ncat))
    v.setdefault('rhofaci', _initial(v['qv'], 'rhofaci', ncat))
    v.setdefault('rhofacr', _initial(v['qv'], 'rhofacr', ncat))
    v.setdefault('ssat', _initial(v['qv'], 'ssat', ncat))
    v.setdefault('sup', _initial(v['qv'], 'sup', ncat))
    v.setdefault('supi', _initial(v['qv'], 'supi', ncat))
    v.setdefault('th', _initial(v['qv'], 'th', ncat))
    v.setdefault('xlf', _initial(v['qv'], 'xlf', ncat))
    v.setdefault('xxls', _initial(v['qv'], 'xxls', ncat))
    v.setdefault('xxlv', _initial(v['qv'], 'xxlv', ncat))
    assert 'g' in v, 'P3 required WRF input/constant missing: g'
    assert 'inv_cp' in v, 'P3 required WRF input/constant missing: inv_cp'
    assert 'invexn' in v, 'P3 required WRF input/constant missing: invexn'
    assert 'it' in v, 'P3 required WRF input/constant missing: it'
    assert 'nccnst' in v, 'P3 required WRF input/constant missing: nccnst'
    assert 'pres' in v, 'P3 required WRF input/constant missing: pres'
    assert 'qsmall' in v, 'P3 required WRF input/constant missing: qsmall'
    assert 'qv_old' in v, 'P3 required WRF input/constant missing: qv_old'
    assert 'rd' in v, 'P3 required WRF input/constant missing: rd'
    assert 'rhosui' in v, 'P3 required WRF input/constant missing: rhosui'
    assert 'rhosur' in v, 'P3 required WRF input/constant missing: rhosur'
    assert 'rhow' in v, 'P3 required WRF input/constant missing: rhow'
    assert 't' in v, 'P3 required WRF input/constant missing: t'
    assert 't_old' in v, 'P3 required WRF input/constant missing: t_old'
    _put(v, 'rho', None, v['pres'] / (v['rd'] * v['t']), alive)  # WRF:2323
    _put(v, 'inv_rho', None, _r(1.0) / v['rho'], alive)  # WRF:2324
    _put(v, 'xxlv', None, _r(3148400.0) - _r(2370.0) * _r(273.15), alive)  # WRF:2325
    _put(v, 'xxls', None, v['xxlv'] + _r(333700.0), alive)  # WRF:2326
    _put(v, 'xlf', None, v['xxls'] - v['xxlv'], alive)  # WRF:2327
    _put(v, 'qvs', None, qv_sat(_max(v['t_old'], _r(1.0)), v['pres'], 0), alive)  # WRF:2329
    _put(v, 'qvi', None, qv_sat(_max(v['t_old'], _r(1.0)), v['pres'], 1), alive)  # WRF:2330
    mask_1 = jnp.logical_and(alive, jnp.logical_or(not log_predictssat, v['it'] <= 1))
    taken_1 = mask_1
    _put(v, 'ssat', None, v['qv_old'] - v['qvs'], mask_1)  # WRF:2334
    _put(v, 'sup', None, v['qv_old'] / v['qvs'] - _r(1.0), mask_1)  # WRF:2335
    _put(v, 'supi', None, v['qv_old'] / v['qvi'] - _r(1.0), mask_1)  # WRF:2336
    mask_2 = jnp.logical_and(jnp.logical_and(alive, jnp.logical_not(taken_1)), jnp.logical_and(log_predictssat, v['it'] > 1))
    taken_1 = jnp.logical_or(taken_1, mask_2)
    _put(v, 'sup', None, v['ssat'] / v['qvs'], mask_2)  # WRF:2339
    _put(v, 'supi', None, (v['ssat'] + v['qvs'] - v['qvi']) / v['qvi'], mask_2)  # WRF:2340
    _put(v, 'rhofacr', None, (v['rhosur'] * v['inv_rho']) ** _r(0.54), alive)  # WRF:2343
    _put(v, 'rhofaci', None, (v['rhosui'] * v['inv_rho']) ** _r(0.54), alive)  # WRF:2344
    _put(v, 'dum', None, _r(1.496e-06) * v['t'] ** _r(1.5) / (v['t'] + _r(120.0)), alive)  # WRF:2345
    _put(v, 'acn', None, v['g'] * v['rhow'] / (_r(18.0) * v['dum']), alive)  # WRF:2346
    if not log_predictnc:
        _put(v, 'nc', None, v['nccnst'] * v['inv_rho'], alive)  # WRF:2350
    mask_3 = jnp.logical_and(alive, jnp.logical_or(jnp.logical_and(v['t'] < _r(273.15), v['supi'] >= _r(-0.05)), jnp.logical_and(jnp.logical_and(v['t'] >= _r(273.15), v['sup'] >= _r(-0.05)), not scpf_on)))
    _put(v, 'log_nucleationpossible', None, True, mask_3)  # WRF:2358
    mask_4 = jnp.logical_and(alive, jnp.logical_or(v['qc'] < v['qsmall'], jnp.logical_and(v['qc'] < _r(1e-08), v['sup'] < _r(-0.1))))
    taken_4 = mask_4
    _put(v, 'qv', None, v['qv'] + v['qc'], mask_4)  # WRF:2366
    _put(v, 'th', None, v['th'] - v['invexn'] * v['qc'] * v['xxlv'] * v['inv_cp'], mask_4)  # WRF:2367
    _put(v, 'qc', None, _r(0.0), mask_4)  # WRF:2368
    _put(v, 'nc', None, _r(0.0), mask_4)  # WRF:2369
    mask_5 = jnp.logical_and(alive, jnp.logical_not(taken_4))
    _put(v, 'log_hydrometeorspresent', None, True, mask_5)  # WRF:2371
    mask_6 = jnp.logical_and(alive, jnp.logical_or(v['qr'] < v['qsmall'], jnp.logical_and(v['qr'] < _r(1e-08), v['sup'] < _r(-0.1))))
    taken_6 = mask_6
    _put(v, 'qv', None, v['qv'] + v['qr'], mask_6)  # WRF:2375
    _put(v, 'th', None, v['th'] - v['invexn'] * v['qr'] * v['xxlv'] * v['inv_cp'], mask_6)  # WRF:2376
    _put(v, 'qr', None, _r(0.0), mask_6)  # WRF:2377
    _put(v, 'nr', None, _r(0.0), mask_6)  # WRF:2378
    mask_7 = jnp.logical_and(alive, jnp.logical_not(taken_6))
    _put(v, 'log_hydrometeorspresent', None, True, mask_7)  # WRF:2380
    for iice in range(1, (ncat) + (1 if (1) > 0 else -1), 1):
        mask_8 = jnp.logical_and(alive, jnp.logical_or(_get(v['qitot'], (iice - 1,)) < v['qsmall'], jnp.logical_and(_get(v['qitot'], (iice - 1,)) < _r(1e-08), v['supi'] < _r(-0.1))))
        taken_8 = mask_8
        _put(v, 'qv', None, v['qv'] + _get(v['qitot'], (iice - 1,)), mask_8)  # WRF:2386
        _put(v, 'th', None, v['th'] - v['invexn'] * _get(v['qitot'], (iice - 1,)) * v['xxls'] * v['inv_cp'], mask_8)  # WRF:2387
        _put(v, 'qitot', ((iice) - 1,), _r(0.0), mask_8)  # WRF:2388
        _put(v, 'nitot', ((iice) - 1,), _r(0.0), mask_8)  # WRF:2389
        _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_8)  # WRF:2390
        _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_8)  # WRF:2391
        mask_9 = jnp.logical_and(alive, jnp.logical_not(taken_8))
        _put(v, 'log_hydrometeorspresent', None, True, mask_9)  # WRF:2393
        mask_10 = jnp.logical_and(alive, jnp.logical_and(jnp.logical_and(_get(v['qitot'], (iice - 1,)) >= v['qsmall'], _get(v['qitot'], (iice - 1,)) < _r(1e-08)), v['t'] >= _r(273.15)))
        taken_10 = mask_10
        _put(v, 'qr', None, v['qr'] + _get(v['qitot'], (iice - 1,)), mask_10)  # WRF:2398
        _put(v, 'nr', None, v['nr'] + _get(v['nitot'], (iice - 1,)), mask_10)  # WRF:2399
        _put(v, 'th', None, v['th'] - v['invexn'] * _get(v['qitot'], (iice - 1,)) * v['xlf'] * v['inv_cp'], mask_10)  # WRF:2400
        _put(v, 'qitot', ((iice) - 1,), _r(0.0), mask_10)  # WRF:2401
        _put(v, 'nitot', ((iice) - 1,), _r(0.0), mask_10)  # WRF:2402
        _put(v, 'qirim', ((iice) - 1,), _r(0.0), mask_10)  # WRF:2403
        _put(v, 'birim', ((iice) - 1,), _r(0.0), mask_10)  # WRF:2404
    return v
