"""WRF P3 column driver: level-local processes and adaptive sedimentation.

The vertical axis is last. Ice categories use an additional final axis inside
the implementation. All wrappers retain old theta/vapor, radii, and the WRF
sqrt(N*Z) transport representation. No host transfers occur inside this step.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .constants import CONSTANTS
from .final_generated import final
from .initial_generated import initial
from .processes_generated import processes
from . import lookup as lut
from .properties import bulk_rime_density, compute_mu_3moment
from .process_support import (_r, _int, _cloud, _rain, gamma, g_of_mu,
                              _call_find_lookuptable_indices_3, _any, _put, _get, _expand, NATIVE_REAL)
from .tables import load_tables
from .qzi import decode as qzi_decode, encode as qzi_encode


def _ice_properties(q, n, r, b, z, tables, three):
    r, b, rho = bulk_rime_density(q, r, b)
    # Only the active WRF branch uses these coordinates; placeholder operands
    # keep inactive SIMD lanes valid without changing their outputs.
    c = lut.ice_coordinates(jnp.maximum(q, _r(1.e-14)), jnp.maximum(n, _r(1.e-16)), r, rho)
    mu = None
    density = _r(200)
    if three:
        z = jnp.maximum(z, _r(1.e-35))
        moment3 = _r(6) / (_r(200) * _r(CONSTANTS['pi'])) * q
        for _ in range(5):
            shape = compute_mu_3moment(n, jnp.maximum(moment3, _r(1.e-19)), z)
            mu = lut.mu_coordinates(q, shape)
            density = lut.ice_lookup(tables['ice'], c, 12, mu)
            moment3 = _r(6) / (density * _r(CONSTANTS['pi'])) * q
    def get(channel):
        return lut.ice_lookup(tables['ice'], c, channel, mu)
    return r, b, z, get, density


def _warm_lookup(table, mu, lam):
    i, j, _, a, b, _ = _call_find_lookuptable_indices_3(0, 0, 0, 0, 0, 0, mu, lam)
    i, j = i - 1, j - 1
    lo = table[i, j] + (a - _r(i + 1)) * (table[i + 1, j] - table[i, j])
    hi = table[i, j + 1] + (a - _r(i + 1)) * (table[i + 1, j + 1] - table[i, j + 1])
    return lo + (b - _r(j + 1)) * (hi - lo)


def _sediment(v, tables, *, kind, category=0, predict_nc=True, three=False, columns_active):
    """Original P3 adaptive substeps, with column-specific top/bottom/dt."""
    cloud, rain = kind == 'cloud', kind == 'rain'
    qname, nname = ('qc', 'nc') if cloud else ('qr', 'nr') if rain else ('qitot', 'nitot')
    q = v[qname] if cloud or rain else _get(v[qname], (category,))
    n = v[nname] if cloud or rain else _get(v[nname], (category,))
    r = jnp.zeros_like(q) if cloud or rain else _get(v['qirim'], (category,))
    b = jnp.zeros_like(q) if cloud or rain else _get(v['birim'], (category,))
    z = jnp.zeros_like(q) if cloud or rain or not three else _get(v['zitot'], (category,))
    levels = jnp.arange(q.shape[-1], dtype=jnp.int32)
    exists = q >= _r(1.e-14)
    present = _any(exists, axis=-1) & columns_active
    top = jnp.max(jnp.where(exists, levels, 0), axis=-1)
    bottom = jnp.min(jnp.where(exists, levels, q.shape[-1]-1), axis=-1)
    dt = v['dt']
    left = jnp.where(present, jnp.broadcast_to(dt, present.shape), _r(0))
    accum = jnp.zeros_like(left)
    rho, invrho, invdz = v['rho'], v['inv_rho'], v['inv_dzq']

    def body(carry):
        q, n, r, b, z, left, bottom, accum = carry
        active_column = left > _r(1.e-4)
        span = (levels >= bottom[..., None]) & (levels <= top[..., None]) & _expand(active_column, 1)
        active = span & ((q > _r(1.e-14)) if cloud or rain else (q >= _r(1.e-14)))
        if cloud:
            nn, mu, _, lam, _, _, _, _ = _cloud(q, n, rho, _r(1))
            n = jnp.where(active, nn, n)
            lam = jnp.where(active, lam, _r(1))
            power = 1 / lam ** _r(CONSTANTS['bcn'])
            vq = v['acn'] * gamma(_r(4) + _r(CONSTANTS['bcn']) + mu) * power / gamma(mu + _r(4))
            if predict_nc:
                vn = v['acn'] * gamma(_r(1) + _r(CONSTANTS['bcn']) + mu) * power / gamma(mu + _r(1))
            else:
                vn = jnp.zeros_like(q)
            vz = vq
        elif rain:
            nn, mu, lam, _, _ = _rain(q, jnp.maximum(n, _r(1.e-16)), _r(1))
            n = jnp.where(active, nn, n)
            lam = jnp.where(active, lam, _r(1))
            vq = _warm_lookup(tables['mass_speed'], mu, lam) * v['rhofacr']
            vn = _warm_lookup(tables['number_speed'], mu, lam) * v['rhofacr']
            vz = vq
        else:
            nn = jnp.maximum(n, _r(1.e-16))
            rr, bb, zz, get, _ = _ice_properties(q, nn, r, b, z, tables, three)
            nn = jnp.minimum(nn, get(7) * q)
            nn = jnp.maximum(nn, get(8) * q)
            n = jnp.where(active, nn, n)
            r, b = jnp.where(active, rr, r), jnp.where(active, bb, b)
            z = jnp.where(active, zz, z)
            vq, vn = get(2) * v['rhofaci'], get(1) * v['rhofaci']
            vz = get(13) * v['rhofaci'] if three else vq
        vq, vn, vz = (jnp.where(active, velocity, _r(0)) for velocity in (vq, vn, vz))
        courant = jnp.max(vz * left[..., None] * invdz, axis=-1)
        steps = _int(courant + _r(1))
        subdt = jnp.minimum(left, left / _r(steps))
        low = jnp.maximum(bottom - 1, 0)
        update = (levels >= low[..., None]) & (levels <= top[..., None]) & _expand(active_column, 1)
        def flux(value, velocity):
            return jnp.where(update, velocity * value * rho, _r(0))
        fq, fn, fr, fb, fz = flux(q, vq), flux(n, vn), flux(r, vq), flux(b, vq), flux(z, vz)
        surface_flux = (jnp.sum(jnp.where(levels == 0, fq, _r(0)), axis=-1)
                        if NATIVE_REAL.get() else fq[..., 0])
        accum = jnp.where(active_column & (bottom == 0), accum + surface_flux * subdt, accum)
        def advance(value, f):
            if NATIVE_REAL.get():
                from gpuwrf.kernels.phys_thompson_sedimentation import _gather
                above = jnp.where(levels+1 < q.shape[-1], _gather.bind(f, (levels+1) % q.shape[-1]), _r(0))
            else:
                above = jnp.concatenate((f[..., 1:], jnp.zeros_like(f[..., :1])), axis=-1)
            div = (above - f) * invdz
            return jnp.where(update, value + div * subdt[..., None] * invrho, value)
        q = advance(q, fq)
        if not cloud or predict_nc:
            n = advance(n, fn)
        if not cloud and not rain:
            r, b = advance(r, fr), advance(b, fb)
            if three:
                z = advance(z, fz)
        left = left - subdt
        bottom = jnp.maximum(bottom - 1, 0)
        return q, n, r, b, z, left, bottom, accum

    carry = jax.lax.while_loop(lambda x: _any(x[5] > _r(1.e-4)), body,
                               (q, n, r, b, z, left, bottom, accum))
    q, n, r, b, z, _, _, accum = carry
    if cloud or rain:
        v[qname], v[nname] = q, n
    else:
        mask = jnp.ones_like(q, dtype=jnp.bool_)
        _put(v, qname, (category,), q, mask)
        _put(v, nname, (category,), n, mask)
        _put(v, 'qirim', (category,), r, mask)
        _put(v, 'birim', (category,), b, mask)
        if three:
            _put(v, 'zitot', (category,), z, mask)
    precip = accum * _r(CONSTANTS['inv_rhow']) * v['odt']
    return v, precip


@partial(jax.jit, static_argnames=('mp', 'capture_stages'))
def step(fields, dt, timestep, *, mp=50, capture_stages=False):
    """One WRF P3 wrapper call, with REAL4 fields shaped (..., nz).

    Required keys: th,qv,qc,qr,nr,qi,ni,qir,bir,p,dz,th_old,qv_old.
    nc for mp51–53; second ice fields for mp52; advected qzi for mp53.
    Returned rainncv/snowncv are amounts (mm), matching the WRF wrapper.
    """
    return _step_impl(fields, dt, timestep, selected_tables(mp), mp=mp, capture_stages=capture_stages)


def selected_tables(mp):
    tables = {k: _r(a) for k, a in load_tables('3mom' if mp == 53 else '2mom').items()}
    tables.update({k: _r(a) for k, a in load_tables('warm').items()})
    if mp == 52:
        tables.update({k: _r(a) for k, a in load_tables('ice_ice').items()})
    return tables


def _step_impl(fields, dt, timestep, tables, *, mp, capture_stages=False, physical_constants=None):
    if mp not in (50, 51, 52, 53):
        raise ValueError('P3 mp must be 50,51,52,53')
    three, predict = mp == 53, mp != 50
    ncat = 2 if mp == 52 else 1
    captured = {}
    def capture(tag, values):
        if capture_stages:
            names = ('th','qv','qc','nc','qr','nr','t','rho','qvs','qvi','sup','supi',
                     'qitot','nitot','qirim','birim','diag_effc','diag_effi','diag_di','diag_vmi','diag_rhoi')
            if tag.startswith('r'):
                names = ('qc','qr','qv','nc','nr','t','rho','qvs','qvi','qccon','qcnuc','qcevp',
                         'qcacc','qcaut','qrcon','qrevp','qcon_satadj','qccol','qchetc','qcheti','qcshd')
            captured[tag] = {name: values[name] for name in names}
    v = ({key: _r(value) for key, value in CONSTANTS.items()}
         if physical_constants is None else dict(physical_constants))
    v.update({key: _r(value) for key, value in fields.items()})
    # WRF wrapper arguments and module PARAMETERs, separate from p3_init's
    # mutable constants. Never silently initialize a missing physical input to 0.
    v.update(clbfact_dep=_r(1), clbfact_sub=_r(1), mu_i_max=_r(20),
             isize=_int(50), iisize=_int(25), rimsize=_int(4), densize=_int(5),
             zsize=_int(11), rcollsize=_int(30),
             dnu=_r((0.,-.557,-.430,-.307,-.186,-.067,.050,.167,.282,.397,.512,.626,.739,.853,.966,.966)))
    v['revap_table'] = tables['ventilation']
    v['vn_table'], v['vm_table'] = tables['number_speed'], tables['mass_speed']
    v['pres'], v['dt'], v['it'] = v['p'], _r(dt), _int(timestep)
    v['odt'] = _r(1) / v['dt']
    shape = v['qv'].shape
    v.setdefault('nc', jnp.zeros_like(v['qv']))
    for target, source in (('qitot','qi'),('nitot','ni'),('qirim','qir'),('birim','bir')):
        v[target] = jnp.stack([v[source]] + ([v[source+'2']] if ncat == 2 else []), axis=-1)
    v['zitot'] = jnp.zeros_like(v['qitot'])
    if three:
        ni = v['ni']
        v['zitot'] = qzi_decode(v['qzi'], ni)[..., None]
    exner = (v['pres'] * _r(1.e-5)) ** (v['rd'] * v['inv_cp'])
    v['invexn'], v['t'], v['t_old'] = 1 / exner, v['th'] * exner, v['th_old'] * exner
    v['qv'] = jnp.maximum(v['qv'], _r(0))
    v['inv_dzq'] = 1 / v['dz']
    v['level_active'] = jnp.ones(shape, dtype=jnp.bool_)
    for name in ('diam_ice', 'rimefraction', 'rimevolume', 'diag_vmi', 'diag_di', 'diag_rhoi'):
        v[name] = jnp.zeros(shape + (ncat,), dtype=jnp.float32)
    v['diag_effi'] = jnp.full(shape + (ncat,), _r(25.e-6), dtype=jnp.float32)
    v['diag_effc'] = jnp.full(shape, _r(10.e-6), dtype=jnp.float32)
    v['diag_ze'] = jnp.full(shape, _r(-99), dtype=jnp.float32)
    v['ze_ice'], v['ze_rain'] = jnp.full(shape, _r(1.e-22)), jnp.full(shape, _r(1.e-22))
    v['deltad_init'] = _r(500.e-6 if ncat == 2 else 999.)
    for name in ('scf','iscf','spf','ispf'):
        v[name] = jnp.ones_like(v['qv'])
    v['spf_clr'] = jnp.zeros_like(v['qv'])
    v['log_nucleationpossible'] = v['log_hydrometeorspresent'] = jnp.zeros(shape, dtype=jnp.bool_)
    v = initial(v, tables, ncat=ncat, log_predictnc=predict, log_3momentice=three)
    capture('1', v)
    if three:
        v['zitot'] = jnp.where(v['qitot'] < v['qsmall'], _r(0), v['zitot'])
    active = _any(v['log_nucleationpossible'] | v['log_hydrometeorspresent'], axis=-1)
    v['qv_cld'], v['qv_clr'] = v['qv'], jnp.zeros_like(v['qv'])
    v['level_active'] = jnp.broadcast_to(_expand(active, 1), shape)
    v['log_hydrometeorspresent'] = jnp.zeros(shape, dtype=jnp.bool_)
    v = processes(v, tables, ncat=ncat, log_predictnc=predict, log_3momentice=three,
                  observer=capture if capture_stages else None)
    capture('2', v)
    hydrometeors = _any(v['log_hydrometeorspresent'], axis=-1) & active
    if three:
        v['zitot'] = jnp.where(v['qitot'] < v['qsmall'], _r(0), v['zitot'])
    v['qv_cld'] = v['qv']
    v, cloud_precip = _sediment(v, tables, kind='cloud', predict_nc=predict, columns_active=hydrometeors)
    v, rain_precip = _sediment(v, tables, kind='rain', columns_active=hydrometeors)
    solid = jnp.zeros(shape[:-1], dtype=jnp.float32)
    for cat in range(ncat):
        v, ice_precip = _sediment(v, tables, kind='ice', category=cat, three=three, columns_active=hydrometeors)
        solid = solid + ice_precip
    capture('3', v)
    v['qv_cld'] = v['qv']
    v['level_active'] = jnp.broadcast_to(_expand(hydrometeors, 1), shape)
    v = final(v, tables, ncat=ncat, log_predictnc=predict, log_3momentice=three)
    capture('5', v)
    if three:
        v['zitot'] = jnp.where(v['qitot'] < v['qsmall'], _r(0), v['zitot'])
    out = {name: v[name] for name in ('th','qv','qc','qr','nr')}
    out.update(th_old=v['th'], qv_old=v['qv'], re_cloud=v['diag_effc'],
               ze=v['diag_ze'], nc=v['nc'] if predict else _r(fields.get('nc', jnp.zeros_like(v['nc']))))
    for target, source in (('qi','qitot'),('ni','nitot'),('qir','qirim'),('bir','birim'),
                           ('vm','diag_vmi'),('di','diag_di'),('rhoi','diag_rhoi')):
        out[target] = _get(v[source], (0,))
        if ncat == 2:
            out[target+'2'] = _get(v[source], (1,))
    if ncat == 2:
        total = _get(v['qitot'], (0,)) + _get(v['qitot'], (1,))
        # WRF wrapper weights by mass/projected area (a harmonic radius).
        q0, q1 = _get(v['qitot'], (0,)), _get(v['qitot'], (1,))
        area0 = jnp.where(q0 >= _r(1.e-14), q0 / _get(v['diag_effi'], (0,)), _r(0))
        area1 = jnp.where(q1 >= _r(1.e-14), q1 / _get(v['diag_effi'], (1,)), _r(0))
        has_ice = (q0 >= _r(1.e-14)) | (q1 >= _r(1.e-14))
        out['re_ice'] = jnp.where(has_ice, total / jnp.where(has_ice, area0 + area1, _r(1)), _r(25.e-6))
    else:
        out['re_ice'] = _get(v['diag_effi'], (0,))
    if three:
        z, ni = _get(v['zitot'], (0,)), _get(v['nitot'], (0,))
        out['qzi'] = qzi_encode(z, ni)
    liquid = cloud_precip + rain_precip
    scale = _r(1000) * v['dt']
    out['rainncv'] = (liquid + solid) * scale
    out['snowncv'] = solid * scale
    out['sr'] = solid / (liquid + solid + _r(1.e-12))
    return (out, captured) if capture_stages else out
