"""Generated GFDL LWR88 family column reference, qualification pending.
Regenerate with proofs/s2_flggfdl/transcribe_gfdl_lw.py. No runtime source parsing.
Native source SHA256: b6b2e9e006eff0b61f637a30e31a70f21b3c2bd8275182acb3c49aaf3fac0b4a
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
    for key,native in {'t1': 'table1', 't2': 'table2', 't4': 'table3', 'em1v': 'em1', 'em1vw': 'em1wde', 'em3v': 'em3'}.items():_TABLES[key]=_TABLES[native].T.reshape(-1)
    for key in ('co251', 'cdt51', 'cdt58', 'c2d51', 'c2d58', 'co258', 'stemp', 'gtemp', 'co231', 'co238', 'c2d31', 'c2d38', 'cdt31', 'cdt38', 'co271', 'co278', 'c2d71', 'c2d78', 'cdt71', 'cdt78', 'co2m51', 'co2m58', 'cdtm51', 'cdtm58', 'c2dm51', 'c2dm58'):
        if key not in co2_tables:raise ValueError('Missing native CO2 table '+key)
    def _lwr88(heatra, grnflx, topflx, press, temp, rh2o, qo3, cldfac, camt, nclds, ktop, kbtm, bo3rnd, ao3rnd, apcm, bpcm, atpcm, btpcm, acomb, bcomb, betacm, zero, one, h18e3, p0inv, h6p08108, diffctr, ginv, h3m4, betinw, rath2omw, gp0inv, p0, p0xzp8, p0xzp2, h3m3, h1m3, h1m2, h25e2, b0, b2, b1, b3, haf, ten, hp1, four, hm1ez, radcon, quartr, two, hm6666m2, hmp66667, hmp5, hp166666, h41666m2, radcon1, h16e1, h28e1, h44194m2, h1p41819, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(heatra).dtype == jnp.dtype("float32") else jnp.float64
        v = {'heatra': heatra, 'grnflx': grnflx, 'topflx': topflx, 'press': press, 'temp': temp, 'rh2o': rh2o, 'qo3': qo3, 'cldfac': cldfac, 'camt': camt, 'nclds': nclds, 'ktop': ktop, 'kbtm': kbtm, 'bo3rnd': bo3rnd, 'ao3rnd': ao3rnd, 'apcm': apcm, 'bpcm': bpcm, 'atpcm': atpcm, 'btpcm': btpcm, 'acomb': acomb, 'bcomb': bcomb, 'betacm': betacm, 'zero': zero, 'one': one, 'h18e3': h18e3, 'p0inv': p0inv, 'h6p08108': h6p08108, 'diffctr': diffctr, 'ginv': ginv, 'h3m4': h3m4, 'betinw': betinw, 'rath2omw': rath2omw, 'gp0inv': gp0inv, 'p0': p0, 'p0xzp8': p0xzp8, 'p0xzp2': p0xzp2, 'h3m3': h3m3, 'h1m3': h1m3, 'h1m2': h1m2, 'h25e2': h25e2, 'b0': b0, 'b2': b2, 'b1': b1, 'b3': b3, 'haf': haf, 'ten': ten, 'hp1': hp1, 'four': four, 'hm1ez': hm1ez, 'radcon': radcon, 'quartr': quartr, 'two': two, 'hm6666m2': hm6666m2, 'hmp66667': hmp66667, 'hmp5': hmp5, 'hp166666': hp166666, 'h41666m2': h41666m2, 'radcon1': radcon1, 'h16e1': h16e1, 'h28e1': h28e1, 'h44194m2': h44194m2, 'h1p41819': h1p41819, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['a1'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['a2'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['cntval'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co21'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co2m2d'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['co2md'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['co2mr'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['co2nbl'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['co2r'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co2r1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co2r2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co2sp1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co2sp2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['d2cd21'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['d2cd22'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['d2cdt2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dco2d1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dco2d2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dco2dt'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['delp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['delp2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['dift'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['empl'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['emx1'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['emx2'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['k'] = jnp.int32(0)
        v['kp'] = jnp.int32(0)
        v['l'] = jnp.int32(0)
        v['ll'] = jnp.int32(0)
        v['llp1'] = jnp.int32(0)
        v['lm1'] = jnp.int32(0)
        v['lm2'] = jnp.int32(0)
        v['lm3'] = jnp.int32(0)
        v['lp1'] = jnp.int32(0)
        v['lp2'] = jnp.int32(0)
        v['lp3'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['p'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['qh2o'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['t'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tdav'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['texpsl'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tlsqu'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['toto3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['totphi'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['totvo2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tphio3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tstdav'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['var1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['var2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['var3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['var4'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['vsum1'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['vsum2'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['vsum3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['vv'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        # WRF:2492 l=kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:2493 lp1=l+1
        v['lp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(1))), v['lp1'])
        # WRF:2493 lp2=l+2
        v['lp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(2))), v['lp2'])
        # WRF:2493 lp3=l+3
        v['lp3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(3))), v['lp3'])
        # WRF:2493 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:2494 lm1=l-1
        v['lm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(1))), v['lm1'])
        # WRF:2494 lm2=l-2
        v['lm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(2))), v['lm2'])
        # WRF:2494 lm3=l-3
        v['lm3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(3))), v['lm3'])
        # WRF:2494 ll = 2*l
        v['ll'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((jnp.int32(2) * v['l'])), v['ll'])
        # WRF:2495 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:2495 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:2498 do k=2,l
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p4*(jnp.int32(1))), v['k'])
            # WRF:2499 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:2500 p(i,k)=haf*(press(i,k-1)+press(i,k))
                v['p'] = v['p'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['press'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['press'][(v['i'])-1, (v['k'])-1]))), v['p'][(v['i'])-1, (v['k'])-1]))
                # WRF:2501 t(i,k)=haf*(temp(i,k-1)+temp(i,k))
                v['t'] = v['t'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['temp'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['temp'][(v['i'])-1, (v['k'])-1]))), v['t'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==103, 0, v['_go'])
                # WRF:2502 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2503 do i=myis,myie
        _p9 = ((jnp.bool_(True)) & (v['_go']==0))
        _p10 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p11(_p12,v):
            v = dict(v)
            v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p12*(jnp.int32(1))), v['i'])
            # WRF:2504 p(i,1)=zero
            v['p'] = v['p'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['p'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2505 p(i,lp1)=press(i,lp1)
            v['p'] = v['p'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(v['press'][(v['i'])-1, (v['lp1'])-1]), v['p'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:2506 t(i,1)=temp(i,1)
            v['t'] = v['t'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(v['temp'][(v['i'])-1, (jnp.int32(1))-1]), v['t'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2507 t(i,lp1)=temp(i,lp1)
            v['t'] = v['t'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(v['temp'][(v['i'])-1, (v['lp1'])-1]), v['t'][(v['i'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==105, 0, v['_go'])
            # WRF:2508 continue
            return v
        v = lax.fori_loop(0,_p10,_p11,v)
        v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p10*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2509 do k=1,l
        _p13 = ((jnp.bool_(True)) & (v['_go']==0))
        _p14 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p15(_p16,v):
            v = dict(v)
            v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p16*(jnp.int32(1))), v['k'])
            # WRF:2510 do i=myis,myie
            _p17 = (((_p13 & (v['_go']==0))) & (v['_go']==0))
            _p18 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p19(_p20,v):
                v = dict(v)
                v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p20*(jnp.int32(1))), v['i'])
                # WRF:2511 delp2(i,k)=p(i,k+1)-p(i,k)
                v['delp2'] = v['delp2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['p'][(v['i'])-1, (v['k'])-1])), v['delp2'][(v['i'])-1, (v['k'])-1]))
                # WRF:2512 delp(i,k)=one/delp2(i,k)
                v['delp'] = v['delp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['delp2'][(v['i'])-1, (v['k'])-1])), v['delp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==107, 0, v['_go'])
                # WRF:2513 continue
                return v
            v = lax.fori_loop(0,_p18,_p19,v)
            v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p18*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p14,_p15,v)
        v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p14*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2516 do k=1,lp1
        _p21 = ((jnp.bool_(True)) & (v['_go']==0))
        _p22 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p23(_p24,v):
            v = dict(v)
            v['k'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p24*(jnp.int32(1))), v['k'])
            # WRF:2517 do i=myis,myie
            _p25 = (((_p21 & (v['_go']==0))) & (v['_go']==0))
            _p26 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p27(_p28,v):
                v = dict(v)
                v['i'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((v['myis'])+_p28*(jnp.int32(1))), v['i'])
                # WRF:2518 texpsl(i,k)=h18e3/temp(i,k)-h6p08108
                v['texpsl'] = v['texpsl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p25 & (v['_go']==0))) & (v['_go']==0)), _r(((v['h18e3'] / v['temp'][(v['i'])-1, (v['k'])-1]) - v['h6p08108'])), v['texpsl'][(v['i'])-1, (v['k'])-1]))
                # WRF:2520 texpsl(i,k)=exp(texpsl(i,k))
                v['texpsl'] = v['texpsl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p25 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp(v['texpsl'][(v['i'])-1, (v['k'])-1])), v['texpsl'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==125, 0, v['_go'])
                # WRF:2521 continue
                return v
            v = lax.fori_loop(0,_p26,_p27,v)
            v['i'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((v['myis'])+_p26*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p22,_p23,v)
        v['k'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p22*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2529 do k=1,l
        _p29 = ((jnp.bool_(True)) & (v['_go']==0))
        _p30 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p31(_p32,v):
            v = dict(v)
            v['k'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p32*(jnp.int32(1))), v['k'])
            # WRF:2530 do i=myis,myie
            _p33 = (((_p29 & (v['_go']==0))) & (v['_go']==0))
            _p34 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p35(_p36,v):
                v = dict(v)
                v['i'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p36*(jnp.int32(1))), v['i'])
                # WRF:2531 qh2o(i,k)=rh2o(i,k)*diffctr
                v['qh2o'] = v['qh2o'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r((v['rh2o'][(v['i'])-1, (v['k'])-1] * v['diffctr'])), v['qh2o'][(v['i'])-1, (v['k'])-1]))
                # WRF:2534 vv(i,k)=haf*(p(i,k+1)+p(i,k))*p0inv
                v['vv'] = v['vv'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r(((v['haf'] * (v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + v['p'][(v['i'])-1, (v['k'])-1])) * v['p0inv'])), v['vv'][(v['i'])-1, (v['k'])-1]))
                # WRF:2535 var1(i,k)=delp2(i,k)*qh2o(i,k)*ginv
                v['var1'] = v['var1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r(((v['delp2'][(v['i'])-1, (v['k'])-1] * v['qh2o'][(v['i'])-1, (v['k'])-1]) * v['ginv'])), v['var1'][(v['i'])-1, (v['k'])-1]))
                # WRF:2536 var3(i,k)=delp2(i,k)*qo3(i,k)*diffctr*ginv
                v['var3'] = v['var3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r((((v['delp2'][(v['i'])-1, (v['k'])-1] * v['qo3'][(v['i'])-1, (v['k'])-1]) * v['diffctr']) * v['ginv'])), v['var3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2537 var2(i,k)=var1(i,k)*(vv(i,k)+h3m4)
                v['var2'] = v['var2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * (v['vv'][(v['i'])-1, (v['k'])-1] + v['h3m4']))), v['var2'][(v['i'])-1, (v['k'])-1]))
                # WRF:2538 var4(i,k)=var3(i,k)*(vv(i,k)+h3m3)
                v['var4'] = v['var4'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r((v['var3'][(v['i'])-1, (v['k'])-1] * (v['vv'][(v['i'])-1, (v['k'])-1] + v['h3m3']))), v['var4'][(v['i'])-1, (v['k'])-1]))
                # WRF:2546 cntval(i,k)=texpsl(i,k)*rh2o(i,k)*var2(i,k)*betinw/ (rh2o(i,k)+rath2omw)
                v['cntval'] = v['cntval'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r(((((v['texpsl'][(v['i'])-1, (v['k'])-1] * v['rh2o'][(v['i'])-1, (v['k'])-1]) * v['var2'][(v['i'])-1, (v['k'])-1]) * v['betinw']) / (v['rh2o'][(v['i'])-1, (v['k'])-1] + v['rath2omw']))), v['cntval'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==131, 0, v['_go'])
                # WRF:2548 continue
                return v
            v = lax.fori_loop(0,_p34,_p35,v)
            v['i'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p34*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p30,_p31,v)
        v['k'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p30*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2550 do i=myis,myie
        _p37 = ((jnp.bool_(True)) & (v['_go']==0))
        _p38 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p39(_p40,v):
            v = dict(v)
            v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p40*(jnp.int32(1))), v['i'])
            # WRF:2551 totphi(i,1)=zero
            v['totphi'] = v['totphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['totphi'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2552 toto3(i,1)=zero
            v['toto3'] = v['toto3'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['toto3'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2553 tphio3(i,1)=zero
            v['tphio3'] = v['tphio3'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['tphio3'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2554 totvo2(i,1)=zero
            v['totvo2'] = v['totvo2'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['totvo2'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==201, 0, v['_go'])
            # WRF:2555 continue
            return v
        v = lax.fori_loop(0,_p38,_p39,v)
        v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p38*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2556 do k=2,lp1
        _p41 = ((jnp.bool_(True)) & (v['_go']==0))
        _p42 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p43(_p44,v):
            v = dict(v)
            v['k'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p44*(jnp.int32(1))), v['k'])
            # WRF:2557 do i=myis,myie
            _p45 = (((_p41 & (v['_go']==0))) & (v['_go']==0))
            _p46 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p47(_p48,v):
                v = dict(v)
                v['i'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((v['myis'])+_p48*(jnp.int32(1))), v['i'])
                # WRF:2558 totphi(i,k)=totphi(i,k-1)+var2(i,k-1)
                v['totphi'] = v['totphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p45 & (v['_go']==0))) & (v['_go']==0)), _r((v['totphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['var2'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])), v['totphi'][(v['i'])-1, (v['k'])-1]))
                # WRF:2559 toto3(i,k)=toto3(i,k-1)+var3(i,k-1)
                v['toto3'] = v['toto3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p45 & (v['_go']==0))) & (v['_go']==0)), _r((v['toto3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['var3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])), v['toto3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2560 tphio3(i,k)=tphio3(i,k-1)+var4(i,k-1)
                v['tphio3'] = v['tphio3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p45 & (v['_go']==0))) & (v['_go']==0)), _r((v['tphio3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['var4'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])), v['tphio3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2561 totvo2(i,k)=totvo2(i,k-1)+cntval(i,k-1)
                v['totvo2'] = v['totvo2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p45 & (v['_go']==0))) & (v['_go']==0)), _r((v['totvo2'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['cntval'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])), v['totvo2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==203, 0, v['_go'])
                # WRF:2562 continue
                return v
            v = lax.fori_loop(0,_p46,_p47,v)
            v['i'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((v['myis'])+_p46*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p42,_p43,v)
        v['k'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p42*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2568 do i=myis,myie
        _p49 = ((jnp.bool_(True)) & (v['_go']==0))
        _p50 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p51(_p52,v):
            v = dict(v)
            v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p52*(jnp.int32(1))), v['i'])
            # WRF:2569 emx1(i)=qh2o(i,l)*press(i,l)*(press(i,l)-p(i,l))*gp0inv
            v['emx1'] = v['emx1'].at[(v['i'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((((v['qh2o'][(v['i'])-1, (v['l'])-1] * v['press'][(v['i'])-1, (v['l'])-1]) * (v['press'][(v['i'])-1, (v['l'])-1] - v['p'][(v['i'])-1, (v['l'])-1])) * v['gp0inv'])), v['emx1'][(v['i'])-1]))
            # WRF:2570 emx2(i)=qh2o(i,l)*press(i,l)*(p(i,lp1)-press(i,l))*gp0inv
            v['emx2'] = v['emx2'].at[(v['i'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((((v['qh2o'][(v['i'])-1, (v['l'])-1] * v['press'][(v['i'])-1, (v['l'])-1]) * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) * v['gp0inv'])), v['emx2'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==801, 0, v['_go'])
            # WRF:2571 continue
            return v
        v = lax.fori_loop(0,_p50,_p51,v)
        v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p50*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2574 do k=1,l
        _p53 = ((jnp.bool_(True)) & (v['_go']==0))
        _p54 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p55(_p56,v):
            v = dict(v)
            v['k'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p56*(jnp.int32(1))), v['k'])
            # WRF:2575 do i=myis,myie
            _p57 = (((_p53 & (v['_go']==0))) & (v['_go']==0))
            _p58 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p59(_p60,v):
                v = dict(v)
                v['i'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((v['myis'])+_p60*(jnp.int32(1))), v['i'])
                # WRF:2576 empl(i,k+1)=qh2o(i,k)*p(i,k+1)*(p(i,k+1)-press(i,k))*gp0inv
                v['empl'] = v['empl'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((((v['qh2o'][(v['i'])-1, (v['k'])-1] * v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * (v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['press'][(v['i'])-1, (v['k'])-1])) * v['gp0inv'])), v['empl'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==811, 0, v['_go'])
                # WRF:2577 continue
                return v
            v = lax.fori_loop(0,_p58,_p59,v)
            v['i'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((v['myis'])+_p58*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p54,_p55,v)
        v['k'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p54*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2578 do k=1,lm1
        _p61 = ((jnp.bool_(True)) & (v['_go']==0))
        _p62 = jnp.maximum(jnp.int32(0), ((v['lm1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p63(_p64,v):
            v = dict(v)
            v['k'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p64*(jnp.int32(1))), v['k'])
            # WRF:2579 do i=myis,myie
            _p65 = (((_p61 & (v['_go']==0))) & (v['_go']==0))
            _p66 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p67(_p68,v):
                v = dict(v)
                v['i'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((v['myis'])+_p68*(jnp.int32(1))), v['i'])
                # WRF:2580 empl(i,lp2+k-1)=qh2o(i,k+1)*p(i,k+1)*(press(i,k+1)-p(i,k+1)) *gp0inv
                v['empl'] = v['empl'].at[(v['i'])-1, (((v['lp2'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r((((v['qh2o'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * (v['press'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])) * v['gp0inv'])), v['empl'][(v['i'])-1, (((v['lp2'] + v['k']) - jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==812, 0, v['_go'])
                # WRF:2582 continue
                return v
            v = lax.fori_loop(0,_p66,_p67,v)
            v['i'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((v['myis'])+_p66*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p62,_p63,v)
        v['k'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p62*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2583 do i=myis,myie
        _p69 = ((jnp.bool_(True)) & (v['_go']==0))
        _p70 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p71(_p72,v):
            v = dict(v)
            v['i'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['myis'])+_p72*(jnp.int32(1))), v['i'])
            # WRF:2584 empl(i,1)=var2(i,l)
            v['empl'] = v['empl'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p69 & (v['_go']==0))) & (v['_go']==0)), _r(v['var2'][(v['i'])-1, (v['l'])-1]), v['empl'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2585 empl(i,llp1)=empl(i,ll)
            v['empl'] = v['empl'].at[(v['i'])-1, (v['llp1'])-1].set(jnp.where((((_p69 & (v['_go']==0))) & (v['_go']==0)), _r(v['empl'][(v['i'])-1, (v['ll'])-1]), v['empl'][(v['i'])-1, (v['llp1'])-1]))
            v['_go'] = jnp.where(v['_go']==821, 0, v['_go'])
            # WRF:2586 continue
            return v
        v = lax.fori_loop(0,_p70,_p71,v)
        v['i'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['myis'])+_p70*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2590 do i=myis,myie
        _p73 = ((jnp.bool_(True)) & (v['_go']==0))
        _p74 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p75(_p76,v):
            v = dict(v)
            v['i'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((v['myis'])+_p76*(jnp.int32(1))), v['i'])
            # WRF:2591 tstdav(i,1)=zero
            v['tstdav'] = v['tstdav'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['tstdav'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2592 tdav(i,1)=zero
            v['tdav'] = v['tdav'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['tdav'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==161, 0, v['_go'])
            # WRF:2593 continue
            return v
        v = lax.fori_loop(0,_p74,_p75,v)
        v['i'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((v['myis'])+_p74*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2594 do k=1,lp1
        _p77 = ((jnp.bool_(True)) & (v['_go']==0))
        _p78 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p79(_p80,v):
            v = dict(v)
            v['k'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p80*(jnp.int32(1))), v['k'])
            # WRF:2595 do i=myis,myie
            _p81 = (((_p77 & (v['_go']==0))) & (v['_go']==0))
            _p82 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p83(_p84,v):
                v = dict(v)
                v['i'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((v['myis'])+_p84*(jnp.int32(1))), v['i'])
                # WRF:2596 vsum3(i,k)=temp(i,k)-stemp(k)
                v['vsum3'] = v['vsum3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p81 & (v['_go']==0))) & (v['_go']==0)), _r((v['temp'][(v['i'])-1, (v['k'])-1] - _r(_TABLES['stemp'])[(v['k'])-1])), v['vsum3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==162, 0, v['_go'])
                # WRF:2597 continue
                return v
            v = lax.fori_loop(0,_p82,_p83,v)
            v['i'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((v['myis'])+_p82*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p78,_p79,v)
        v['k'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p78*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2598 do k=1,l
        _p85 = ((jnp.bool_(True)) & (v['_go']==0))
        _p86 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p87(_p88,v):
            v = dict(v)
            v['k'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p88*(jnp.int32(1))), v['k'])
            # WRF:2599 do i=myis,myie
            _p89 = (((_p85 & (v['_go']==0))) & (v['_go']==0))
            _p90 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p91(_p92,v):
                v = dict(v)
                v['i'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((v['myis'])+_p92*(jnp.int32(1))), v['i'])
                # WRF:2600 vsum2(i)=gtemp(k)*delp2(i,k)
                v['vsum2'] = v['vsum2'].at[(v['i'])-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r((_r(_TABLES['gtemp'])[(v['k'])-1] * v['delp2'][(v['i'])-1, (v['k'])-1])), v['vsum2'][(v['i'])-1]))
                # WRF:2601 vsum1(i)=vsum2(i)*vsum3(i,k)
                v['vsum1'] = v['vsum1'].at[(v['i'])-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r((v['vsum2'][(v['i'])-1] * v['vsum3'][(v['i'])-1, (v['k'])-1])), v['vsum1'][(v['i'])-1]))
                # WRF:2602 tstdav(i,k+1)=tstdav(i,k)+vsum2(i)
                v['tstdav'] = v['tstdav'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r((v['tstdav'][(v['i'])-1, (v['k'])-1] + v['vsum2'][(v['i'])-1])), v['tstdav'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:2603 tdav(i,k+1)=tdav(i,k)+vsum1(i)
                v['tdav'] = v['tdav'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r((v['tdav'][(v['i'])-1, (v['k'])-1] + v['vsum1'][(v['i'])-1])), v['tdav'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==165, 0, v['_go'])
                # WRF:2604 continue
                return v
            v = lax.fori_loop(0,_p90,_p91,v)
            v['i'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((v['myis'])+_p90*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==163, 0, v['_go'])
            # WRF:2605 continue
            return v
        v = lax.fori_loop(0,_p86,_p87,v)
        v['k'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p86*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2608 do i=myis,myie
        _p93 = ((jnp.bool_(True)) & (v['_go']==0))
        _p94 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p95(_p96,v):
            v = dict(v)
            v['i'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((v['myis'])+_p96*(jnp.int32(1))), v['i'])
            # WRF:2609 a1(i)=(press(i,lp1)-p0xzp8)/p0xzp2
            v['a1'] = v['a1'].at[(v['i'])-1].set(jnp.where((((_p93 & (v['_go']==0))) & (v['_go']==0)), _r(((v['press'][(v['i'])-1, (v['lp1'])-1] - v['p0xzp8']) / v['p0xzp2'])), v['a1'][(v['i'])-1]))
            # WRF:2610 a2(i)=(p0-press(i,lp1))/p0xzp2
            v['a2'] = v['a2'].at[(v['i'])-1].set(jnp.where((((_p93 & (v['_go']==0))) & (v['_go']==0)), _r(((v['p0'] - v['press'][(v['i'])-1, (v['lp1'])-1]) / v['p0xzp2'])), v['a2'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==171, 0, v['_go'])
            # WRF:2611 continue
            return v
        v = lax.fori_loop(0,_p94,_p95,v)
        v['i'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((v['myis'])+_p94*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2616 do k=1,lp1
        _p97 = ((jnp.bool_(True)) & (v['_go']==0))
        _p98 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p99(_p100,v):
            v = dict(v)
            v['k'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p100*(jnp.int32(1))), v['k'])
            # WRF:2617 do i=myis,myie
            _p101 = (((_p97 & (v['_go']==0))) & (v['_go']==0))
            _p102 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p103(_p104,v):
                v = dict(v)
                v['i'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((v['myis'])+_p104*(jnp.int32(1))), v['i'])
                # WRF:2618 co2r1(i,k)=a1(i)*co231(k)+a2(i)*co238(k)
                v['co2r1'] = v['co2r1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co231'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co238'])[(v['k'])-1]))), v['co2r1'][(v['i'])-1, (v['k'])-1]))
                # WRF:2619 d2cd21(i,k)=h1m3*(a1(i)*c2d31(k)+a2(i)*c2d38(k))
                v['d2cd21'] = v['d2cd21'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d31'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d38'])[(v['k'])-1])))), v['d2cd21'][(v['i'])-1, (v['k'])-1]))
                # WRF:2620 dco2d1(i,k)=h1m2*(a1(i)*cdt31(k)+a2(i)*cdt38(k))
                v['dco2d1'] = v['dco2d1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt31'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt38'])[(v['k'])-1])))), v['dco2d1'][(v['i'])-1, (v['k'])-1]))
                # WRF:2621 co2r2(i,k)=a1(i)*co271(k)+a2(i)*co278(k)
                v['co2r2'] = v['co2r2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co271'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co278'])[(v['k'])-1]))), v['co2r2'][(v['i'])-1, (v['k'])-1]))
                # WRF:2622 d2cd22(i,k)=h1m3*(a1(i)*c2d71(k)+a2(i)*c2d78(k))
                v['d2cd22'] = v['d2cd22'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d71'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d78'])[(v['k'])-1])))), v['d2cd22'][(v['i'])-1, (v['k'])-1]))
                # WRF:2623 dco2d2(i,k)=h1m2*(a1(i)*cdt71(k)+a2(i)*cdt78(k))
                v['dco2d2'] = v['dco2d2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt71'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt78'])[(v['k'])-1])))), v['dco2d2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==184, 0, v['_go'])
                # WRF:2624 continue
                return v
            v = lax.fori_loop(0,_p102,_p103,v)
            v['i'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((v['myis'])+_p102*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p98,_p99,v)
        v['k'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p98*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2625 do k=1,l
        _p105 = ((jnp.bool_(True)) & (v['_go']==0))
        _p106 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p107(_p108,v):
            v = dict(v)
            v['k'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p108*(jnp.int32(1))), v['k'])
            # WRF:2626 do i=myis,myie
            _p109 = (((_p105 & (v['_go']==0))) & (v['_go']==0))
            _p110 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p111(_p112,v):
                v = dict(v)
                v['i'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((v['myis'])+_p112*(jnp.int32(1))), v['i'])
                # WRF:2627 co2mr(i,k)=a1(i)*co2m51(k)+a2(i)*co2m58(k)
                v['co2mr'] = v['co2mr'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p109 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co2m51'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co2m58'])[(v['k'])-1]))), v['co2mr'][(v['i'])-1, (v['k'])-1]))
                # WRF:2628 co2md(i,k)=h1m2*(a1(i)*cdtm51(k)+a2(i)*cdtm58(k))
                v['co2md'] = v['co2md'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p109 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdtm51'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdtm58'])[(v['k'])-1])))), v['co2md'][(v['i'])-1, (v['k'])-1]))
                # WRF:2629 co2m2d(i,k)=h1m3*(a1(i)*c2dm51(k)+a2(i)*c2dm58(k))
                v['co2m2d'] = v['co2m2d'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p109 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2dm51'])[(v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2dm58'])[(v['k'])-1])))), v['co2m2d'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==190, 0, v['_go'])
                # WRF:2630 continue
                return v
            v = lax.fori_loop(0,_p110,_p111,v)
            v['i'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((v['myis'])+_p110*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p106,_p107,v)
        v['k'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p106*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2636 do kp=2,lp1
        _p113 = ((jnp.bool_(True)) & (v['_go']==0))
        _p114 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p115(_p116,v):
            v = dict(v)
            v['kp'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p116*(jnp.int32(1))), v['kp'])
            # WRF:2637 do i=myis,myie
            _p117 = (((_p113 & (v['_go']==0))) & (v['_go']==0))
            _p118 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p119(_p120,v):
                v = dict(v)
                v['i'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((v['myis'])+_p120*(jnp.int32(1))), v['i'])
                # WRF:2638 dift(i,kp)=tdav(i,kp)/tstdav(i,kp)
                v['dift'] = v['dift'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p117 & (v['_go']==0))) & (v['_go']==0)), _r((v['tdav'][(v['i'])-1, (v['kp'])-1] / v['tstdav'][(v['i'])-1, (v['kp'])-1])), v['dift'][(v['i'])-1, (v['kp'])-1]))
                v['_go'] = jnp.where(v['_go']==211, 0, v['_go'])
                # WRF:2639 continue
                return v
            v = lax.fori_loop(0,_p118,_p119,v)
            v['i'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((v['myis'])+_p118*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p114,_p115,v)
        v['kp'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p114*(jnp.int32(1))), v['kp'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2640 do i=myis,myie
        _p121 = ((jnp.bool_(True)) & (v['_go']==0))
        _p122 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p123(_p124,v):
            v = dict(v)
            v['i'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((v['myis'])+_p124*(jnp.int32(1))), v['i'])
            # WRF:2641 co21(i,1,1)=1.0
            v['co21'] = v['co21'].at[(v['i'])-1, (jnp.int32(1))-1, (jnp.int32(1))-1].set(jnp.where((((_p121 & (v['_go']==0))) & (v['_go']==0)), _r(_r(1.0)), v['co21'][(v['i'])-1, (jnp.int32(1))-1, (jnp.int32(1))-1]))
            # WRF:2642 co2sp1(i,1)=1.0
            v['co2sp1'] = v['co2sp1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p121 & (v['_go']==0))) & (v['_go']==0)), _r(_r(1.0)), v['co2sp1'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2643 co2sp2(i,1)=1.0
            v['co2sp2'] = v['co2sp2'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p121 & (v['_go']==0))) & (v['_go']==0)), _r(_r(1.0)), v['co2sp2'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==212, 0, v['_go'])
            # WRF:2644 continue
            return v
        v = lax.fori_loop(0,_p122,_p123,v)
        v['i'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((v['myis'])+_p122*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2645 do kp=2,lp1
        _p125 = ((jnp.bool_(True)) & (v['_go']==0))
        _p126 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p127(_p128,v):
            v = dict(v)
            v['kp'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p128*(jnp.int32(1))), v['kp'])
            # WRF:2646 do i=myis,myie
            _p129 = (((_p125 & (v['_go']==0))) & (v['_go']==0))
            _p130 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p131(_p132,v):
                v = dict(v)
                v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p132*(jnp.int32(1))), v['i'])
                # WRF:2648 co2r(i,kp)=a1(i)*co251(kp,1)+a2(i)*co258(kp,1)
                v['co2r'] = v['co2r'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co251'])[(v['kp'])-1, (jnp.int32(1))-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co258'])[(v['kp'])-1, (jnp.int32(1))-1]))), v['co2r'][(v['i'])-1, (v['kp'])-1]))
                # WRF:2649 dco2dt(i,kp)=h1m2*(a1(i)*cdt51(kp,1)+a2(i)*cdt58(kp,1))
                v['dco2dt'] = v['dco2dt'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt51'])[(v['kp'])-1, (jnp.int32(1))-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt58'])[(v['kp'])-1, (jnp.int32(1))-1])))), v['dco2dt'][(v['i'])-1, (v['kp'])-1]))
                # WRF:2650 d2cdt2(i,kp)=h1m3*(a1(i)*c2d51(kp,1)+a2(i)*c2d58(kp,1))
                v['d2cdt2'] = v['d2cdt2'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d51'])[(v['kp'])-1, (jnp.int32(1))-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d58'])[(v['kp'])-1, (jnp.int32(1))-1])))), v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))
                # WRF:2651 co21(i,kp,1)=co2r(i,kp)+dift(i,kp)*(dco2dt(i,kp)+ haf*dift(i,kp)*d2cdt2(i,kp))
                v['co21'] = v['co21'].at[(v['i'])-1, (v['kp'])-1, (jnp.int32(1))-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r'][(v['i'])-1, (v['kp'])-1] + (v['dift'][(v['i'])-1, (v['kp'])-1] * (v['dco2dt'][(v['i'])-1, (v['kp'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['kp'])-1]) * v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))))), v['co21'][(v['i'])-1, (v['kp'])-1, (jnp.int32(1))-1]))
                # WRF:2655 co2r(i,kp)=a1(i)*co251(1,kp)+a2(i)*co258(1,kp)
                v['co2r'] = v['co2r'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co251'])[(jnp.int32(1))-1, (v['kp'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co258'])[(jnp.int32(1))-1, (v['kp'])-1]))), v['co2r'][(v['i'])-1, (v['kp'])-1]))
                # WRF:2656 dco2dt(i,kp)=h1m2*(a1(i)*cdt51(1,kp)+a2(i)*cdt58(1,kp))
                v['dco2dt'] = v['dco2dt'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt51'])[(jnp.int32(1))-1, (v['kp'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt58'])[(jnp.int32(1))-1, (v['kp'])-1])))), v['dco2dt'][(v['i'])-1, (v['kp'])-1]))
                # WRF:2657 d2cdt2(i,kp)=h1m3*(a1(i)*c2d51(1,kp)+a2(i)*c2d58(1,kp))
                v['d2cdt2'] = v['d2cdt2'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d51'])[(jnp.int32(1))-1, (v['kp'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d58'])[(jnp.int32(1))-1, (v['kp'])-1])))), v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))
                # WRF:2658 co21(i,1,kp)=co2r(i,kp)+dift(i,kp)*(dco2dt(i,kp)+ haf*dift(i,kp)*d2cdt2(i,kp))
                v['co21'] = v['co21'].at[(v['i'])-1, (jnp.int32(1))-1, (v['kp'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r'][(v['i'])-1, (v['kp'])-1] + (v['dift'][(v['i'])-1, (v['kp'])-1] * (v['dco2dt'][(v['i'])-1, (v['kp'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['kp'])-1]) * v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))))), v['co21'][(v['i'])-1, (jnp.int32(1))-1, (v['kp'])-1]))
                v['_go'] = jnp.where(v['_go']==215, 0, v['_go'])
                # WRF:2660 continue
                return v
            v = lax.fori_loop(0,_p130,_p131,v)
            v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p130*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p126,_p127,v)
        v['kp'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p126*(jnp.int32(1))), v['kp'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2664 do k=2,lp1
        _p133 = ((jnp.bool_(True)) & (v['_go']==0))
        _p134 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p135(_p136,v):
            v = dict(v)
            v['k'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p136*(jnp.int32(1))), v['k'])
            # WRF:2665 do i=myis,myie
            _p137 = (((_p133 & (v['_go']==0))) & (v['_go']==0))
            _p138 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p139(_p140,v):
                v = dict(v)
                v['i'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((v['myis'])+_p140*(jnp.int32(1))), v['i'])
                # WRF:2666 co2sp1(i,k)=co2r1(i,k)+dift(i,k)*(dco2d1(i,k)+haf*dift(i,k)* d2cd21(i,k))
                v['co2sp1'] = v['co2sp1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r1'][(v['i'])-1, (v['k'])-1] + (v['dift'][(v['i'])-1, (v['k'])-1] * (v['dco2d1'][(v['i'])-1, (v['k'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['k'])-1]) * v['d2cd21'][(v['i'])-1, (v['k'])-1]))))), v['co2sp1'][(v['i'])-1, (v['k'])-1]))
                # WRF:2668 co2sp2(i,k)=co2r2(i,k)+dift(i,k)*(dco2d2(i,k)+haf*dift(i,k)* d2cd22(i,k))
                v['co2sp2'] = v['co2sp2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r2'][(v['i'])-1, (v['k'])-1] + (v['dift'][(v['i'])-1, (v['k'])-1] * (v['dco2d2'][(v['i'])-1, (v['k'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['k'])-1]) * v['d2cd22'][(v['i'])-1, (v['k'])-1]))))), v['co2sp2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==250, 0, v['_go'])
                # WRF:2670 continue
                return v
            v = lax.fori_loop(0,_p138,_p139,v)
            v['i'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((v['myis'])+_p138*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p134,_p135,v)
        v['k'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p134*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2673 do k=2,l
        _p141 = ((jnp.bool_(True)) & (v['_go']==0))
        _p142 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p143(_p144,v):
            v = dict(v)
            v['k'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p144*(jnp.int32(1))), v['k'])
            # WRF:2674 do kp=k+1,lp1
            _p145 = (((_p141 & (v['_go']==0))) & (v['_go']==0))
            _p146 = jnp.maximum(jnp.int32(0), ((v['lp1'])-((v['k'] + jnp.int32(1))))//(jnp.int32(1))+jnp.int32(1))
            def _p147(_p148,v):
                v = dict(v)
                v['kp'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p148*(jnp.int32(1))), v['kp'])
                # WRF:2675 do i=myis,myie
                _p149 = (((_p145 & (v['_go']==0))) & (v['_go']==0))
                _p150 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p151(_p152,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((v['myis'])+_p152*(jnp.int32(1))), v['i'])
                    # WRF:2676 dift(i,kp)=(tdav(i,kp)-tdav(i,k))/ (tstdav(i,kp)-tstdav(i,k))
                    v['dift'] = v['dift'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tdav'][(v['i'])-1, (v['kp'])-1] - v['tdav'][(v['i'])-1, (v['k'])-1]) / (v['tstdav'][(v['i'])-1, (v['kp'])-1] - v['tstdav'][(v['i'])-1, (v['k'])-1]))), v['dift'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2678 co2r(i,kp)=a1(i)*co251(kp,k)+a2(i)*co258(kp,k)
                    v['co2r'] = v['co2r'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co251'])[(v['kp'])-1, (v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co258'])[(v['kp'])-1, (v['k'])-1]))), v['co2r'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2679 dco2dt(i,kp)=h1m2*(a1(i)*cdt51(kp,k)+a2(i)*cdt58(kp,k))
                    v['dco2dt'] = v['dco2dt'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt51'])[(v['kp'])-1, (v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt58'])[(v['kp'])-1, (v['k'])-1])))), v['dco2dt'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2680 d2cdt2(i,kp)=h1m3*(a1(i)*c2d51(kp,k)+a2(i)*c2d58(kp,k))
                    v['d2cdt2'] = v['d2cdt2'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d51'])[(v['kp'])-1, (v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d58'])[(v['kp'])-1, (v['k'])-1])))), v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2681 co21(i,kp,k)=co2r(i,kp)+dift(i,kp)*(dco2dt(i,kp)+ haf*dift(i,kp)*d2cdt2(i,kp))
                    v['co21'] = v['co21'].at[(v['i'])-1, (v['kp'])-1, (v['k'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r'][(v['i'])-1, (v['kp'])-1] + (v['dift'][(v['i'])-1, (v['kp'])-1] * (v['dco2dt'][(v['i'])-1, (v['kp'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['kp'])-1]) * v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))))), v['co21'][(v['i'])-1, (v['kp'])-1, (v['k'])-1]))
                    # WRF:2683 co2r(i,kp)=a1(i)*co251(k,kp)+a2(i)*co258(k,kp)
                    v['co2r'] = v['co2r'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co251'])[(v['k'])-1, (v['kp'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co258'])[(v['k'])-1, (v['kp'])-1]))), v['co2r'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2684 dco2dt(i,kp)=h1m2*(a1(i)*cdt51(k,kp)+a2(i)*cdt58(k,kp))
                    v['dco2dt'] = v['dco2dt'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt51'])[(v['k'])-1, (v['kp'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt58'])[(v['k'])-1, (v['kp'])-1])))), v['dco2dt'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2685 d2cdt2(i,kp)=h1m3*(a1(i)*c2d51(k,kp)+a2(i)*c2d58(k,kp))
                    v['d2cdt2'] = v['d2cdt2'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d51'])[(v['k'])-1, (v['kp'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d58'])[(v['k'])-1, (v['kp'])-1])))), v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))
                    # WRF:2686 co21(i,k,kp)=co2r(i,kp)+dift(i,kp)*(dco2dt(i,kp)+ haf*dift(i,kp)*d2cdt2(i,kp))
                    v['co21'] = v['co21'].at[(v['i'])-1, (v['k'])-1, (v['kp'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r'][(v['i'])-1, (v['kp'])-1] + (v['dift'][(v['i'])-1, (v['kp'])-1] * (v['dco2dt'][(v['i'])-1, (v['kp'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['kp'])-1]) * v['d2cdt2'][(v['i'])-1, (v['kp'])-1]))))), v['co21'][(v['i'])-1, (v['k'])-1, (v['kp'])-1]))
                    v['_go'] = jnp.where(v['_go']==222, 0, v['_go'])
                    # WRF:2688 continue
                    return v
                v = lax.fori_loop(0,_p150,_p151,v)
                v['i'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((v['myis'])+_p150*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p146,_p147,v)
            v['kp'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p146*(jnp.int32(1))), v['kp'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==220, 0, v['_go'])
            # WRF:2689 continue
            return v
        v = lax.fori_loop(0,_p142,_p143,v)
        v['k'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p142*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2691 do k=2,lp1
        _p153 = ((jnp.bool_(True)) & (v['_go']==0))
        _p154 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p155(_p156,v):
            v = dict(v)
            v['k'] = jnp.where((_p153 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p156*(jnp.int32(1))), v['k'])
            # WRF:2692 do i=myis,myie
            _p157 = (((_p153 & (v['_go']==0))) & (v['_go']==0))
            _p158 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p159(_p160,v):
                v = dict(v)
                v['i'] = jnp.where((_p157 & (v['_go']==0)), jnp.int32((v['myis'])+_p160*(jnp.int32(1))), v['i'])
                # WRF:2693 dift(i,k)=haf*(vsum3(i,k)+vsum3(i,k-1))
                v['dift'] = v['dift'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['vsum3'][(v['i'])-1, (v['k'])-1] + v['vsum3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1]))), v['dift'][(v['i'])-1, (v['k'])-1]))
                # WRF:2694 co2r(i,k)=a1(i)*co251(k,k)+a2(i)*co258(k,k)
                v['co2r'] = v['co2r'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r(((v['a1'][(v['i'])-1] * _r(_TABLES['co251'])[(v['k'])-1, (v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['co258'])[(v['k'])-1, (v['k'])-1]))), v['co2r'][(v['i'])-1, (v['k'])-1]))
                # WRF:2695 dco2dt(i,k)=h1m2*(a1(i)*cdt51(k,k)+a2(i)*cdt58(k,k))
                v['dco2dt'] = v['dco2dt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m2'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['cdt51'])[(v['k'])-1, (v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['cdt58'])[(v['k'])-1, (v['k'])-1])))), v['dco2dt'][(v['i'])-1, (v['k'])-1]))
                # WRF:2696 d2cdt2(i,k)=h1m3*(a1(i)*c2d51(k,k)+a2(i)*c2d58(k,k))
                v['d2cdt2'] = v['d2cdt2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r((v['h1m3'] * ((v['a1'][(v['i'])-1] * _r(_TABLES['c2d51'])[(v['k'])-1, (v['k'])-1]) + (v['a2'][(v['i'])-1] * _r(_TABLES['c2d58'])[(v['k'])-1, (v['k'])-1])))), v['d2cdt2'][(v['i'])-1, (v['k'])-1]))
                # WRF:2697 co21(i,k,k)=co2r(i,k)+dift(i,k)*(dco2dt(i,k)+ haf*dift(i,k)*d2cdt2(i,k))
                v['co21'] = v['co21'].at[(v['i'])-1, (v['k'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2r'][(v['i'])-1, (v['k'])-1] + (v['dift'][(v['i'])-1, (v['k'])-1] * (v['dco2dt'][(v['i'])-1, (v['k'])-1] + ((v['haf'] * v['dift'][(v['i'])-1, (v['k'])-1]) * v['d2cdt2'][(v['i'])-1, (v['k'])-1]))))), v['co21'][(v['i'])-1, (v['k'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==206, 0, v['_go'])
                # WRF:2699 continue
                return v
            v = lax.fori_loop(0,_p158,_p159,v)
            v['i'] = jnp.where((_p157 & (v['_go']==0)), jnp.int32((v['myis'])+_p158*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p154,_p155,v)
        v['k'] = jnp.where((_p153 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p154*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2701 do k=1,l
        _p161 = ((jnp.bool_(True)) & (v['_go']==0))
        _p162 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p163(_p164,v):
            v = dict(v)
            v['k'] = jnp.where((_p161 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p164*(jnp.int32(1))), v['k'])
            # WRF:2702 do i=myis,myie
            _p165 = (((_p161 & (v['_go']==0))) & (v['_go']==0))
            _p166 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p167(_p168,v):
                v = dict(v)
                v['i'] = jnp.where((_p165 & (v['_go']==0)), jnp.int32((v['myis'])+_p168*(jnp.int32(1))), v['i'])
                # WRF:2703 co2nbl(i,k)=co2mr(i,k)+vsum3(i,k)*(co2md(i,k)+haf* vsum3(i,k)*co2m2d(i,k))
                v['co2nbl'] = v['co2nbl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p165 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2mr'][(v['i'])-1, (v['k'])-1] + (v['vsum3'][(v['i'])-1, (v['k'])-1] * (v['co2md'][(v['i'])-1, (v['k'])-1] + ((v['haf'] * v['vsum3'][(v['i'])-1, (v['k'])-1]) * v['co2m2d'][(v['i'])-1, (v['k'])-1]))))), v['co2nbl'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==260, 0, v['_go'])
                # WRF:2705 continue
                return v
            v = lax.fori_loop(0,_p166,_p167,v)
            v['i'] = jnp.where((_p165 & (v['_go']==0)), jnp.int32((v['myis'])+_p166*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p162,_p163,v)
        v['k'] = jnp.where((_p161 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p162*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2707 do k=1,lp1
        _p169 = ((jnp.bool_(True)) & (v['_go']==0))
        _p170 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p171(_p172,v):
            v = dict(v)
            v['k'] = jnp.where((_p169 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p172*(jnp.int32(1))), v['k'])
            # WRF:2708 do i=myis,myie
            _p173 = (((_p169 & (v['_go']==0))) & (v['_go']==0))
            _p174 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p175(_p176,v):
                v = dict(v)
                v['i'] = jnp.where((_p173 & (v['_go']==0)), jnp.int32((v['myis'])+_p176*(jnp.int32(1))), v['i'])
                # WRF:2709 __if__ t(i,k).le.h25e2
                _p177 = (((_p173 & (v['_go']==0))) & (v['_go']==0)) & ((v['t'][(v['i'])-1, (v['k'])-1] <= v['h25e2']))
                _p178 = (((_p173 & (v['_go']==0))) & (v['_go']==0)) & ~((v['t'][(v['i'])-1, (v['k'])-1] <= v['h25e2']))
                # WRF:2710 tlsqu(i,k)=b0+(t(i,k)-h25e2)* (b1+(t(i,k)-h25e2)* (b2+b3*(t(i,k)-h25e2)))
                v['tlsqu'] = v['tlsqu'].at[(v['i'])-1, (v['k'])-1].set(jnp.where(((_p177) & (v['_go']==0)), _r((v['b0'] + ((v['t'][(v['i'])-1, (v['k'])-1] - v['h25e2']) * (v['b1'] + ((v['t'][(v['i'])-1, (v['k'])-1] - v['h25e2']) * (v['b2'] + (v['b3'] * (v['t'][(v['i'])-1, (v['k'])-1] - v['h25e2'])))))))), v['tlsqu'][(v['i'])-1, (v['k'])-1]))
                # WRF:2714 tlsqu(i,k)=b0
                v['tlsqu'] = v['tlsqu'].at[(v['i'])-1, (v['k'])-1].set(jnp.where(((_p178) & (v['_go']==0)), _r(v['b0']), v['tlsqu'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==264, 0, v['_go'])
                # WRF:2716 continue
                return v
            v = lax.fori_loop(0,_p174,_p175,v)
            v['i'] = jnp.where((_p173 & (v['_go']==0)), jnp.int32((v['myis'])+_p174*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p170,_p171,v)
        v['k'] = jnp.where((_p169 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p170*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2718 do k=1,lp1
        _p179 = ((jnp.bool_(True)) & (v['_go']==0))
        _p180 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p181(_p182,v):
            v = dict(v)
            v['k'] = jnp.where((_p179 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p182*(jnp.int32(1))), v['k'])
            # WRF:2719 do kp=1,lp1
            _p183 = (((_p179 & (v['_go']==0))) & (v['_go']==0))
            _p184 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p185(_p186,v):
                v = dict(v)
                v['kp'] = jnp.where((_p183 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p186*(jnp.int32(1))), v['kp'])
                # WRF:2720 do i=myis,myie
                _p187 = (((_p183 & (v['_go']==0))) & (v['_go']==0))
                _p188 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p189(_p190,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p187 & (v['_go']==0)), jnp.int32((v['myis'])+_p190*(jnp.int32(1))), v['i'])
                    # WRF:2721 co21(i,kp,k)=co21(i,kp,k)*(one-tlsqu(i,kp))+tlsqu(i,kp)
                    v['co21'] = v['co21'].at[(v['i'])-1, (v['kp'])-1, (v['k'])-1].set(jnp.where((((_p187 & (v['_go']==0))) & (v['_go']==0)), _r(((v['co21'][(v['i'])-1, (v['kp'])-1, (v['k'])-1] * (v['one'] - v['tlsqu'][(v['i'])-1, (v['kp'])-1])) + v['tlsqu'][(v['i'])-1, (v['kp'])-1])), v['co21'][(v['i'])-1, (v['kp'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==282, 0, v['_go'])
                    # WRF:2722 continue
                    return v
                v = lax.fori_loop(0,_p188,_p189,v)
                v['i'] = jnp.where((_p187 & (v['_go']==0)), jnp.int32((v['myis'])+_p188*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p184,_p185,v)
            v['kp'] = jnp.where((_p183 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p184*(jnp.int32(1))), v['kp'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==280, 0, v['_go'])
            # WRF:2723 continue
            return v
        v = lax.fori_loop(0,_p180,_p181,v)
        v['k'] = jnp.where((_p179 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p180*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2724 do k=1,lp1
        _p191 = ((jnp.bool_(True)) & (v['_go']==0))
        _p192 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p193(_p194,v):
            v = dict(v)
            v['k'] = jnp.where((_p191 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p194*(jnp.int32(1))), v['k'])
            # WRF:2725 do i=myis,myie
            _p195 = (((_p191 & (v['_go']==0))) & (v['_go']==0))
            _p196 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p197(_p198,v):
                v = dict(v)
                v['i'] = jnp.where((_p195 & (v['_go']==0)), jnp.int32((v['myis'])+_p198*(jnp.int32(1))), v['i'])
                # WRF:2726 co2sp1(i,k)=co2sp1(i,k)*(one-tlsqu(i,1))+tlsqu(i,1)
                v['co2sp1'] = v['co2sp1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p195 & (v['_go']==0))) & (v['_go']==0)), _r(((v['co2sp1'][(v['i'])-1, (v['k'])-1] * (v['one'] - v['tlsqu'][(v['i'])-1, (jnp.int32(1))-1])) + v['tlsqu'][(v['i'])-1, (jnp.int32(1))-1])), v['co2sp1'][(v['i'])-1, (v['k'])-1]))
                # WRF:2727 co2sp2(i,k)=co2sp2(i,k)*(one-tlsqu(i,1))+tlsqu(i,1)
                v['co2sp2'] = v['co2sp2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p195 & (v['_go']==0))) & (v['_go']==0)), _r(((v['co2sp2'][(v['i'])-1, (v['k'])-1] * (v['one'] - v['tlsqu'][(v['i'])-1, (jnp.int32(1))-1])) + v['tlsqu'][(v['i'])-1, (jnp.int32(1))-1])), v['co2sp2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==286, 0, v['_go'])
                # WRF:2728 continue
                return v
            v = lax.fori_loop(0,_p196,_p197,v)
            v['i'] = jnp.where((_p195 & (v['_go']==0)), jnp.int32((v['myis'])+_p196*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==284, 0, v['_go'])
            # WRF:2729 continue
            return v
        v = lax.fori_loop(0,_p192,_p193,v)
        v['k'] = jnp.where((_p191 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p192*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2730 do k=1,l
        _p199 = ((jnp.bool_(True)) & (v['_go']==0))
        _p200 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p201(_p202,v):
            v = dict(v)
            v['k'] = jnp.where((_p199 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p202*(jnp.int32(1))), v['k'])
            # WRF:2731 do i=myis,myie
            _p203 = (((_p199 & (v['_go']==0))) & (v['_go']==0))
            _p204 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p205(_p206,v):
                v = dict(v)
                v['i'] = jnp.where((_p203 & (v['_go']==0)), jnp.int32((v['myis'])+_p206*(jnp.int32(1))), v['i'])
                # WRF:2732 co2nbl(i,k)=co2nbl(i,k)*(one-tlsqu(i,k))+tlsqu(i,k)
                v['co2nbl'] = v['co2nbl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p203 & (v['_go']==0))) & (v['_go']==0)), _r(((v['co2nbl'][(v['i'])-1, (v['k'])-1] * (v['one'] - v['tlsqu'][(v['i'])-1, (v['k'])-1])) + v['tlsqu'][(v['i'])-1, (v['k'])-1])), v['co2nbl'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==290, 0, v['_go'])
                # WRF:2733 continue
                return v
            v = lax.fori_loop(0,_p204,_p205,v)
            v['i'] = jnp.where((_p203 & (v['_go']==0)), jnp.int32((v['myis'])+_p204*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==288, 0, v['_go'])
            # WRF:2734 continue
            return v
        v = lax.fori_loop(0,_p200,_p201,v)
        v['k'] = jnp.where((_p199 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p200*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2756 call fst88(heatra,grnflx,topflx, qh2o,press,p,delp,delp2,temp,t, cldfac,nclds,ktop,kbtm,camt, co21,co2nbl,co2sp1,co2sp2, var1,var2,var3,var4,cntval, toto3,tphio3,totphi,totvo2, emx1,emx2,empl, bo3rnd,ao3rnd, apcm,bpcm,atpcm,btpcm,acomb,bcomb,betacm, ten,hp1,haf,one,four,hm1ez, radcon,quartr,two, hm6666m2,hmp66667,hmp5, hp166666,h41666m2,radcon1, h16e1, h28e1, h25e2, h44194m2,h1p41819, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte                     )
        _p207 = lax.cond(((jnp.bool_(True)) & (v['_go']==0)), lambda _a: _fst88(*_a), lambda _a: _a, (v['heatra'], v['grnflx'], v['topflx'], v['qh2o'], v['press'], v['p'], v['delp'], v['delp2'], v['temp'], v['t'], v['cldfac'], v['nclds'], v['ktop'], v['kbtm'], v['camt'], v['co21'], v['co2nbl'], v['co2sp1'], v['co2sp2'], v['var1'], v['var2'], v['var3'], v['var4'], v['cntval'], v['toto3'], v['tphio3'], v['totphi'], v['totvo2'], v['emx1'], v['emx2'], v['empl'], v['bo3rnd'], v['ao3rnd'], v['apcm'], v['bpcm'], v['atpcm'], v['btpcm'], v['acomb'], v['bcomb'], v['betacm'], v['ten'], v['hp1'], v['haf'], v['one'], v['four'], v['hm1ez'], v['radcon'], v['quartr'], v['two'], v['hm6666m2'], v['hmp66667'], v['hmp5'], v['hp166666'], v['h41666m2'], v['radcon1'], v['h16e1'], v['h28e1'], v['h25e2'], v['h44194m2'], v['h1p41819'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],))
        v['heatra'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[0]), v['heatra'])
        v['grnflx'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[1]), v['grnflx'])
        v['topflx'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[2]), v['topflx'])
        v['qh2o'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[3]), v['qh2o'])
        v['press'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[4]), v['press'])
        v['p'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[5]), v['p'])
        v['delp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[6]), v['delp'])
        v['delp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[7]), v['delp2'])
        v['temp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[8]), v['temp'])
        v['t'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[9]), v['t'])
        v['cldfac'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[10]), v['cldfac'])
        v['nclds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[11]), v['nclds'])
        v['ktop'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[12]), v['ktop'])
        v['kbtm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[13]), v['kbtm'])
        v['camt'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[14]), v['camt'])
        v['co21'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[15]), v['co21'])
        v['co2nbl'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[16]), v['co2nbl'])
        v['co2sp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[17]), v['co2sp1'])
        v['co2sp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[18]), v['co2sp2'])
        v['var1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[19]), v['var1'])
        v['var2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[20]), v['var2'])
        v['var3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[21]), v['var3'])
        v['var4'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[22]), v['var4'])
        v['cntval'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[23]), v['cntval'])
        v['toto3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[24]), v['toto3'])
        v['tphio3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[25]), v['tphio3'])
        v['totphi'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[26]), v['totphi'])
        v['totvo2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[27]), v['totvo2'])
        v['emx1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[28]), v['emx1'])
        v['emx2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[29]), v['emx2'])
        v['empl'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[30]), v['empl'])
        v['bo3rnd'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[31]), v['bo3rnd'])
        v['ao3rnd'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[32]), v['ao3rnd'])
        v['apcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[33]), v['apcm'])
        v['bpcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[34]), v['bpcm'])
        v['atpcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[35]), v['atpcm'])
        v['btpcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[36]), v['btpcm'])
        v['acomb'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[37]), v['acomb'])
        v['bcomb'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[38]), v['bcomb'])
        v['betacm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[39]), v['betacm'])
        v['ten'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[40]), v['ten'])
        v['hp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[41]), v['hp1'])
        v['haf'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[42]), v['haf'])
        v['one'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[43]), v['one'])
        v['four'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[44]), v['four'])
        v['hm1ez'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[45]), v['hm1ez'])
        v['radcon'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[46]), v['radcon'])
        v['quartr'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[47]), v['quartr'])
        v['two'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[48]), v['two'])
        v['hm6666m2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[49]), v['hm6666m2'])
        v['hmp66667'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[50]), v['hmp66667'])
        v['hmp5'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[51]), v['hmp5'])
        v['hp166666'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[52]), v['hp166666'])
        v['h41666m2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[53]), v['h41666m2'])
        v['radcon1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[54]), v['radcon1'])
        v['h16e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[55]), v['h16e1'])
        v['h28e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[56]), v['h28e1'])
        v['h25e2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[57]), v['h25e2'])
        v['h44194m2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[58]), v['h44194m2'])
        v['h1p41819'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p207[59]), v['h1p41819'])
        v['ids'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[60]), v['ids'])
        v['ide'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[61]), v['ide'])
        v['jds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[62]), v['jds'])
        v['jde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[63]), v['jde'])
        v['kds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[64]), v['kds'])
        v['kde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[65]), v['kde'])
        v['ims'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[66]), v['ims'])
        v['ime'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[67]), v['ime'])
        v['jms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[68]), v['jms'])
        v['jme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[69]), v['jme'])
        v['kms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[70]), v['kms'])
        v['kme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[71]), v['kme'])
        v['its'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[72]), v['its'])
        v['ite'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[73]), v['ite'])
        v['jts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[74]), v['jts'])
        v['jte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[75]), v['jte'])
        v['kts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[76]), v['kts'])
        v['kte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p207[77]), v['kte'])
        return (v['heatra'], v['grnflx'], v['topflx'], v['press'], v['temp'], v['rh2o'], v['qo3'], v['cldfac'], v['camt'], v['nclds'], v['ktop'], v['kbtm'], v['bo3rnd'], v['ao3rnd'], v['apcm'], v['bpcm'], v['atpcm'], v['btpcm'], v['acomb'], v['bcomb'], v['betacm'], v['zero'], v['one'], v['h18e3'], v['p0inv'], v['h6p08108'], v['diffctr'], v['ginv'], v['h3m4'], v['betinw'], v['rath2omw'], v['gp0inv'], v['p0'], v['p0xzp8'], v['p0xzp2'], v['h3m3'], v['h1m3'], v['h1m2'], v['h25e2'], v['b0'], v['b2'], v['b1'], v['b3'], v['haf'], v['ten'], v['hp1'], v['four'], v['hm1ez'], v['radcon'], v['quartr'], v['two'], v['hm6666m2'], v['hmp66667'], v['hmp5'], v['hp166666'], v['h41666m2'], v['radcon1'], v['h16e1'], v['h28e1'], v['h44194m2'], v['h1p41819'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)

    def _fst88(heatra, grnflx, topflx, qh2o, press, p, delp, delp2, temp, t, cldfac, nclds, ktop, kbtm, camt, co21, co2nbl, co2sp1, co2sp2, var1, var2, var3, var4, cntval, toto3, tphio3, totphi, totvo2, emx1, emx2, empl, bo3rnd, ao3rnd, apcm, bpcm, atpcm, btpcm, acomb, bcomb, betacm, ten, hp1, haf, one, four, hm1ez, radcon, quartr, two, hm6666m2, hmp66667, hmp5, hp166666, h41666m2, radcon1, h16e1, h28e1, h25e2, h44194m2, h1p41819, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(heatra).dtype == jnp.dtype("float32") else jnp.float64
        v = {'heatra': heatra, 'grnflx': grnflx, 'topflx': topflx, 'qh2o': qh2o, 'press': press, 'p': p, 'delp': delp, 'delp2': delp2, 'temp': temp, 't': t, 'cldfac': cldfac, 'nclds': nclds, 'ktop': ktop, 'kbtm': kbtm, 'camt': camt, 'co21': co21, 'co2nbl': co2nbl, 'co2sp1': co2sp1, 'co2sp2': co2sp2, 'var1': var1, 'var2': var2, 'var3': var3, 'var4': var4, 'cntval': cntval, 'toto3': toto3, 'tphio3': tphio3, 'totphi': totphi, 'totvo2': totvo2, 'emx1': emx1, 'emx2': emx2, 'empl': empl, 'bo3rnd': bo3rnd, 'ao3rnd': ao3rnd, 'apcm': apcm, 'bpcm': bpcm, 'atpcm': atpcm, 'btpcm': btpcm, 'acomb': acomb, 'bcomb': bcomb, 'betacm': betacm, 'ten': ten, 'hp1': hp1, 'haf': haf, 'one': one, 'four': four, 'hm1ez': hm1ez, 'radcon': radcon, 'quartr': quartr, 'two': two, 'hm6666m2': hm6666m2, 'hmp66667': hmp66667, 'hmp5': hmp5, 'hp166666': hp166666, 'h41666m2': h41666m2, 'radcon1': radcon1, 'h16e1': h16e1, 'h28e1': h28e1, 'h25e2': h25e2, 'h44194m2': h44194m2, 'h1p41819': h1p41819, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['alp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['avephi'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['avmo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['avpho3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['avvo2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['c'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['c2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['cnttau'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['co2sp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['cont1d'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['contdg'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['csour'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['css'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['csub'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['csub2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['cts'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['ctso3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['delpr1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['delpr2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['delptc'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['dsorc'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dt'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dtc'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dte2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dtsp'] = jnp.zeros((((1))-((1))+1,2,), dtype=_r)
        v['e1cts1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['e1cts2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['e1ctw1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['e1ctw2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['e1flx'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['emd'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['emisdg'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['emiss'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['emissb'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['emspec'] = jnp.zeros((((1))-((1))+1,2,), dtype=_r)
        v['excts'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['fac1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['fbot'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['flx'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['flx1e1'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['flxnet'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ftop'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['fxo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['fxoe2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['fxosp'] = jnp.zeros((((1))-((1))+1,2,), dtype=_r)
        v['gxcts'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['heatem'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['icnt'] = jnp.int32(0)
        v['ixo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=jnp.int32)
        v['j1'] = jnp.int32(0)
        v['j3'] = jnp.int32(0)
        v['k'] = jnp.int32(0)
        v['kclds'] = jnp.int32(0)
        v['kk'] = jnp.int32(0)
        v['klen'] = jnp.int32(0)
        v['kmax'] = jnp.int32(0)
        v['kmin'] = jnp.int32(0)
        v['kp'] = jnp.int32(0)
        v['l'] = jnp.int32(0)
        v['ll'] = jnp.int32(0)
        v['llm1'] = jnp.int32(0)
        v['llm2'] = jnp.int32(0)
        v['llp1'] = jnp.int32(0)
        v['lm1'] = jnp.int32(0)
        v['lm2'] = jnp.int32(0)
        v['lm3'] = jnp.int32(0)
        v['lp1'] = jnp.int32(0)
        v['lp2'] = jnp.int32(0)
        v['lp3'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['oss'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['over1d'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['pbot'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['ptop'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['rlog'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['sorc'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,(15),), dtype=_r)
        v['ss1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ss2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tc'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['to31d'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['to3dg'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['to3sp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['to3spc'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['totevv'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tpl'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['vsum1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['vtmp3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['z1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        # WRF:2883 l=kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:2884 lp1=l+1
        v['lp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(1))), v['lp1'])
        # WRF:2884 lp2=l+2
        v['lp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(2))), v['lp2'])
        # WRF:2884 lp3=l+3
        v['lp3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(3))), v['lp3'])
        # WRF:2884 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:2885 lm1=l-1
        v['lm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(1))), v['lm1'])
        # WRF:2885 lm2=l-2
        v['lm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(2))), v['lm2'])
        # WRF:2885 lm3=l-3
        v['lm3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(3))), v['lm3'])
        # WRF:2885 ll = 2*l
        v['ll'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((jnp.int32(2) * v['l'])), v['ll'])
        # WRF:2886 llm2 = ll-2
        v['llm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(2))), v['llm2'])
        # WRF:2886 llm1=ll-1
        v['llm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(1))), v['llm1'])
        # WRF:2887 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:2887 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:2890 do k=1,lp1
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p4*(jnp.int32(1))), v['k'])
            # WRF:2891 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:2893 vtmp3(i,k)=aint(temp(i,k)*hp1)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['temp'][(v['i'])-1, (v['k'])-1] * v['hp1']))), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2894 fxo(i,k)=vtmp3(i,k)-9.
                v['fxo'] = v['fxo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] - _r(9.))), v['fxo'][(v['i'])-1, (v['k'])-1]))
                # WRF:2895 dt(i,k)=temp(i,k)-ten*vtmp3(i,k)
                v['dt'] = v['dt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['temp'][(v['i'])-1, (v['k'])-1] - (v['ten'] * v['vtmp3'][(v['i'])-1, (v['k'])-1]))), v['dt'][(v['i'])-1, (v['k'])-1]))
                # WRF:2897 ixo(i,k)=fxo(i,k)
                v['ixo'] = v['ixo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['fxo'][(v['i'])-1, (v['k'])-1]), v['ixo'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==101, 0, v['_go'])
                # WRF:2898 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2899 do k=1,l
        _p9 = ((jnp.bool_(True)) & (v['_go']==0))
        _p10 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p11(_p12,v):
            v = dict(v)
            v['k'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p12*(jnp.int32(1))), v['k'])
            # WRF:2900 do i=myis,myie
            _p13 = (((_p9 & (v['_go']==0))) & (v['_go']==0))
            _p14 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p15(_p16,v):
                v = dict(v)
                v['i'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((v['myis'])+_p16*(jnp.int32(1))), v['i'])
                # WRF:2902 vtmp3(i,k)=aint(t(i,k+1)*hp1)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p13 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['t'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['hp1']))), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2903 fxoe2(i,k)=vtmp3(i,k)-9.
                v['fxoe2'] = v['fxoe2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p13 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] - _r(9.))), v['fxoe2'][(v['i'])-1, (v['k'])-1]))
                # WRF:2904 dte2(i,k)=t(i,k+1)-ten*vtmp3(i,k)
                v['dte2'] = v['dte2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p13 & (v['_go']==0))) & (v['_go']==0)), _r((v['t'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - (v['ten'] * v['vtmp3'][(v['i'])-1, (v['k'])-1]))), v['dte2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==103, 0, v['_go'])
                # WRF:2905 continue
                return v
            v = lax.fori_loop(0,_p14,_p15,v)
            v['i'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((v['myis'])+_p14*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p10,_p11,v)
        v['k'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p10*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2907 do i=myis,myie
        _p17 = ((jnp.bool_(True)) & (v['_go']==0))
        _p18 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p19(_p20,v):
            v = dict(v)
            v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p20*(jnp.int32(1))), v['i'])
            # WRF:2908 fxoe2(i,lp1)=fxo(i,l)
            v['fxoe2'] = v['fxoe2'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['fxo'][(v['i'])-1, (v['l'])-1]), v['fxoe2'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:2909 dte2(i,lp1)=dt(i,l)
            v['dte2'] = v['dte2'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['dt'][(v['i'])-1, (v['l'])-1]), v['dte2'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:2910 fxosp(i,1)=fxoe2(i,lm1)
            v['fxosp'] = v['fxosp'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['fxoe2'][(v['i'])-1, (v['lm1'])-1]), v['fxosp'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2911 fxosp(i,2)=fxo(i,lm1)
            v['fxosp'] = v['fxosp'].at[(v['i'])-1, (jnp.int32(2))-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['fxo'][(v['i'])-1, (v['lm1'])-1]), v['fxosp'][(v['i'])-1, (jnp.int32(2))-1]))
            # WRF:2912 dtsp(i,1)=dte2(i,lm1)
            v['dtsp'] = v['dtsp'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['dte2'][(v['i'])-1, (v['lm1'])-1]), v['dtsp'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:2913 dtsp(i,2)=dt(i,lm1)
            v['dtsp'] = v['dtsp'].at[(v['i'])-1, (jnp.int32(2))-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['dt'][(v['i'])-1, (v['lm1'])-1]), v['dtsp'][(v['i'])-1, (jnp.int32(2))-1]))
            v['_go'] = jnp.where(v['_go']==105, 0, v['_go'])
            # WRF:2914 continue
            return v
        v = lax.fori_loop(0,_p18,_p19,v)
        v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p18*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2917 do i=myis,myie
        _p21 = ((jnp.bool_(True)) & (v['_go']==0))
        _p22 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p23(_p24,v):
            v = dict(v)
            v['i'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((v['myis'])+_p24*(jnp.int32(1))), v['i'])
            # WRF:2918 do k=1,lp1
            _p25 = (((_p21 & (v['_go']==0))) & (v['_go']==0))
            _p26 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p27(_p28,v):
                v = dict(v)
                v['k'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p28*(jnp.int32(1))), v['k'])
                # WRF:2919 vtmp3(i,k)=source(ixo(i,k),1)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p25 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(1))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2920 dsorc(i,k)=dsrce(ixo(i,k),1)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p25 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(1))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4114, 0, v['_go'])
                # WRF:2921 continue
                return v
            v = lax.fori_loop(0,_p26,_p27,v)
            v['k'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p26*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p22,_p23,v)
        v['i'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((v['myis'])+_p22*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2922 do k=1,lp1
        _p29 = ((jnp.bool_(True)) & (v['_go']==0))
        _p30 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p31(_p32,v):
            v = dict(v)
            v['k'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p32*(jnp.int32(1))), v['k'])
            # WRF:2923 do i=myis,myie
            _p33 = (((_p29 & (v['_go']==0))) & (v['_go']==0))
            _p34 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p35(_p36,v):
                v = dict(v)
                v['i'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p36*(jnp.int32(1))), v['i'])
                # WRF:2924 sorc(i,k,1)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(1))-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(1))-1]))
                v['_go'] = jnp.where(v['_go']==4112, 0, v['_go'])
                # WRF:2925 continue
                return v
            v = lax.fori_loop(0,_p34,_p35,v)
            v['i'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p34*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p30,_p31,v)
        v['k'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p30*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2927 do i=myis,myie
        _p37 = ((jnp.bool_(True)) & (v['_go']==0))
        _p38 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p39(_p40,v):
            v = dict(v)
            v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p40*(jnp.int32(1))), v['i'])
            # WRF:2928 do k=1,lp1
            _p41 = (((_p37 & (v['_go']==0))) & (v['_go']==0))
            _p42 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p43(_p44,v):
                v = dict(v)
                v['k'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p44*(jnp.int32(1))), v['k'])
                # WRF:2929 vtmp3(i,k)=source(ixo(i,k),2)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p41 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(2))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2930 dsorc(i,k)=dsrce(ixo(i,k),2)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p41 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(2))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4214, 0, v['_go'])
                # WRF:2931 continue
                return v
            v = lax.fori_loop(0,_p42,_p43,v)
            v['k'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p42*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p38,_p39,v)
        v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p38*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2932 do k=1,lp1
        _p45 = ((jnp.bool_(True)) & (v['_go']==0))
        _p46 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p47(_p48,v):
            v = dict(v)
            v['k'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p48*(jnp.int32(1))), v['k'])
            # WRF:2933 do i=myis,myie
            _p49 = (((_p45 & (v['_go']==0))) & (v['_go']==0))
            _p50 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p51(_p52,v):
                v = dict(v)
                v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p52*(jnp.int32(1))), v['i'])
                # WRF:2934 sorc(i,k,2)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(2))-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(2))-1]))
                v['_go'] = jnp.where(v['_go']==4212, 0, v['_go'])
                # WRF:2935 continue
                return v
            v = lax.fori_loop(0,_p50,_p51,v)
            v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p50*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p46,_p47,v)
        v['k'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p46*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2937 do i=myis,myie
        _p53 = ((jnp.bool_(True)) & (v['_go']==0))
        _p54 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p55(_p56,v):
            v = dict(v)
            v['i'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((v['myis'])+_p56*(jnp.int32(1))), v['i'])
            # WRF:2938 do k=1,lp1
            _p57 = (((_p53 & (v['_go']==0))) & (v['_go']==0))
            _p58 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p59(_p60,v):
                v = dict(v)
                v['k'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p60*(jnp.int32(1))), v['k'])
                # WRF:2939 vtmp3(i,k)=source(ixo(i,k),3)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(3))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2940 dsorc(i,k)=dsrce(ixo(i,k),3)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(3))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4314, 0, v['_go'])
                # WRF:2941 continue
                return v
            v = lax.fori_loop(0,_p58,_p59,v)
            v['k'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p58*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p54,_p55,v)
        v['i'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((v['myis'])+_p54*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2942 do k=1,lp1
        _p61 = ((jnp.bool_(True)) & (v['_go']==0))
        _p62 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p63(_p64,v):
            v = dict(v)
            v['k'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p64*(jnp.int32(1))), v['k'])
            # WRF:2943 do i=myis,myie
            _p65 = (((_p61 & (v['_go']==0))) & (v['_go']==0))
            _p66 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p67(_p68,v):
                v = dict(v)
                v['i'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((v['myis'])+_p68*(jnp.int32(1))), v['i'])
                # WRF:2944 sorc(i,k,3)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(3))-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(3))-1]))
                v['_go'] = jnp.where(v['_go']==4312, 0, v['_go'])
                # WRF:2945 continue
                return v
            v = lax.fori_loop(0,_p66,_p67,v)
            v['i'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((v['myis'])+_p66*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p62,_p63,v)
        v['k'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p62*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2947 do i=myis,myie
        _p69 = ((jnp.bool_(True)) & (v['_go']==0))
        _p70 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p71(_p72,v):
            v = dict(v)
            v['i'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['myis'])+_p72*(jnp.int32(1))), v['i'])
            # WRF:2948 do k=1,lp1
            _p73 = (((_p69 & (v['_go']==0))) & (v['_go']==0))
            _p74 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p75(_p76,v):
                v = dict(v)
                v['k'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p76*(jnp.int32(1))), v['k'])
                # WRF:2949 vtmp3(i,k)=source(ixo(i,k),4)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(4))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2950 dsorc(i,k)=dsrce(ixo(i,k),4)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(4))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4414, 0, v['_go'])
                # WRF:2951 continue
                return v
            v = lax.fori_loop(0,_p74,_p75,v)
            v['k'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p74*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p70,_p71,v)
        v['i'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['myis'])+_p70*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2952 do k=1,lp1
        _p77 = ((jnp.bool_(True)) & (v['_go']==0))
        _p78 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p79(_p80,v):
            v = dict(v)
            v['k'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p80*(jnp.int32(1))), v['k'])
            # WRF:2953 do i=myis,myie
            _p81 = (((_p77 & (v['_go']==0))) & (v['_go']==0))
            _p82 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p83(_p84,v):
                v = dict(v)
                v['i'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((v['myis'])+_p84*(jnp.int32(1))), v['i'])
                # WRF:2954 sorc(i,k,4)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(4))-1].set(jnp.where((((_p81 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(4))-1]))
                v['_go'] = jnp.where(v['_go']==4412, 0, v['_go'])
                # WRF:2955 continue
                return v
            v = lax.fori_loop(0,_p82,_p83,v)
            v['i'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((v['myis'])+_p82*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p78,_p79,v)
        v['k'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p78*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2957 do i=myis,myie
        _p85 = ((jnp.bool_(True)) & (v['_go']==0))
        _p86 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p87(_p88,v):
            v = dict(v)
            v['i'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((v['myis'])+_p88*(jnp.int32(1))), v['i'])
            # WRF:2958 do k=1,lp1
            _p89 = (((_p85 & (v['_go']==0))) & (v['_go']==0))
            _p90 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p91(_p92,v):
                v = dict(v)
                v['k'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p92*(jnp.int32(1))), v['k'])
                # WRF:2959 vtmp3(i,k)=source(ixo(i,k),5)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(5))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2960 dsorc(i,k)=dsrce(ixo(i,k),5)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(5))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4514, 0, v['_go'])
                # WRF:2961 continue
                return v
            v = lax.fori_loop(0,_p90,_p91,v)
            v['k'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p90*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p86,_p87,v)
        v['i'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((v['myis'])+_p86*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2962 do k=1,lp1
        _p93 = ((jnp.bool_(True)) & (v['_go']==0))
        _p94 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p95(_p96,v):
            v = dict(v)
            v['k'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p96*(jnp.int32(1))), v['k'])
            # WRF:2963 do i=myis,myie
            _p97 = (((_p93 & (v['_go']==0))) & (v['_go']==0))
            _p98 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p99(_p100,v):
                v = dict(v)
                v['i'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((v['myis'])+_p100*(jnp.int32(1))), v['i'])
                # WRF:2964 sorc(i,k,5)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(5))-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(5))-1]))
                v['_go'] = jnp.where(v['_go']==4512, 0, v['_go'])
                # WRF:2965 continue
                return v
            v = lax.fori_loop(0,_p98,_p99,v)
            v['i'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((v['myis'])+_p98*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p94,_p95,v)
        v['k'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p94*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2967 do i=myis,myie
        _p101 = ((jnp.bool_(True)) & (v['_go']==0))
        _p102 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p103(_p104,v):
            v = dict(v)
            v['i'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((v['myis'])+_p104*(jnp.int32(1))), v['i'])
            # WRF:2968 do k=1,lp1
            _p105 = (((_p101 & (v['_go']==0))) & (v['_go']==0))
            _p106 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p107(_p108,v):
                v = dict(v)
                v['k'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p108*(jnp.int32(1))), v['k'])
                # WRF:2969 vtmp3(i,k)=source(ixo(i,k),6)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p105 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(6))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2970 dsorc(i,k)=dsrce(ixo(i,k),6)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p105 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(6))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4614, 0, v['_go'])
                # WRF:2971 continue
                return v
            v = lax.fori_loop(0,_p106,_p107,v)
            v['k'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p106*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p102,_p103,v)
        v['i'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((v['myis'])+_p102*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2972 do k=1,lp1
        _p109 = ((jnp.bool_(True)) & (v['_go']==0))
        _p110 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p111(_p112,v):
            v = dict(v)
            v['k'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p112*(jnp.int32(1))), v['k'])
            # WRF:2973 do i=myis,myie
            _p113 = (((_p109 & (v['_go']==0))) & (v['_go']==0))
            _p114 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p115(_p116,v):
                v = dict(v)
                v['i'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((v['myis'])+_p116*(jnp.int32(1))), v['i'])
                # WRF:2974 sorc(i,k,6)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(6))-1].set(jnp.where((((_p113 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(6))-1]))
                v['_go'] = jnp.where(v['_go']==4612, 0, v['_go'])
                # WRF:2975 continue
                return v
            v = lax.fori_loop(0,_p114,_p115,v)
            v['i'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((v['myis'])+_p114*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p110,_p111,v)
        v['k'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p110*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2977 do i=myis,myie
        _p117 = ((jnp.bool_(True)) & (v['_go']==0))
        _p118 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p119(_p120,v):
            v = dict(v)
            v['i'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((v['myis'])+_p120*(jnp.int32(1))), v['i'])
            # WRF:2978 do k=1,lp1
            _p121 = (((_p117 & (v['_go']==0))) & (v['_go']==0))
            _p122 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p123(_p124,v):
                v = dict(v)
                v['k'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p124*(jnp.int32(1))), v['k'])
                # WRF:2979 vtmp3(i,k)=source(ixo(i,k),7)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p121 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(7))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2980 dsorc(i,k)=dsrce(ixo(i,k),7)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p121 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(7))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4714, 0, v['_go'])
                # WRF:2981 continue
                return v
            v = lax.fori_loop(0,_p122,_p123,v)
            v['k'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p122*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p118,_p119,v)
        v['i'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((v['myis'])+_p118*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2982 do k=1,lp1
        _p125 = ((jnp.bool_(True)) & (v['_go']==0))
        _p126 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p127(_p128,v):
            v = dict(v)
            v['k'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p128*(jnp.int32(1))), v['k'])
            # WRF:2983 do i=myis,myie
            _p129 = (((_p125 & (v['_go']==0))) & (v['_go']==0))
            _p130 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p131(_p132,v):
                v = dict(v)
                v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p132*(jnp.int32(1))), v['i'])
                # WRF:2984 sorc(i,k,7)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(7))-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(7))-1]))
                v['_go'] = jnp.where(v['_go']==4712, 0, v['_go'])
                # WRF:2985 continue
                return v
            v = lax.fori_loop(0,_p130,_p131,v)
            v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p130*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p126,_p127,v)
        v['k'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p126*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2987 do i=myis,myie
        _p133 = ((jnp.bool_(True)) & (v['_go']==0))
        _p134 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p135(_p136,v):
            v = dict(v)
            v['i'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((v['myis'])+_p136*(jnp.int32(1))), v['i'])
            # WRF:2988 do k=1,lp1
            _p137 = (((_p133 & (v['_go']==0))) & (v['_go']==0))
            _p138 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p139(_p140,v):
                v = dict(v)
                v['k'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p140*(jnp.int32(1))), v['k'])
                # WRF:2989 vtmp3(i,k)=source(ixo(i,k),8)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(8))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:2990 dsorc(i,k)=dsrce(ixo(i,k),8)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(8))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4814, 0, v['_go'])
                # WRF:2991 continue
                return v
            v = lax.fori_loop(0,_p138,_p139,v)
            v['k'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p138*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p134,_p135,v)
        v['i'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((v['myis'])+_p134*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2992 do k=1,lp1
        _p141 = ((jnp.bool_(True)) & (v['_go']==0))
        _p142 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p143(_p144,v):
            v = dict(v)
            v['k'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p144*(jnp.int32(1))), v['k'])
            # WRF:2993 do i=myis,myie
            _p145 = (((_p141 & (v['_go']==0))) & (v['_go']==0))
            _p146 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p147(_p148,v):
                v = dict(v)
                v['i'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32((v['myis'])+_p148*(jnp.int32(1))), v['i'])
                # WRF:2994 sorc(i,k,8)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(8))-1].set(jnp.where((((_p145 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(8))-1]))
                v['_go'] = jnp.where(v['_go']==4812, 0, v['_go'])
                # WRF:2995 continue
                return v
            v = lax.fori_loop(0,_p146,_p147,v)
            v['i'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32((v['myis'])+_p146*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p142,_p143,v)
        v['k'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p142*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:2997 do i=myis,myie
        _p149 = ((jnp.bool_(True)) & (v['_go']==0))
        _p150 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p151(_p152,v):
            v = dict(v)
            v['i'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((v['myis'])+_p152*(jnp.int32(1))), v['i'])
            # WRF:2998 do k=1,lp1
            _p153 = (((_p149 & (v['_go']==0))) & (v['_go']==0))
            _p154 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p155(_p156,v):
                v = dict(v)
                v['k'] = jnp.where((_p153 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p156*(jnp.int32(1))), v['k'])
                # WRF:2999 vtmp3(i,k)=source(ixo(i,k),9)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p153 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(9))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3000 dsorc(i,k)=dsrce(ixo(i,k),9)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p153 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(9))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==4914, 0, v['_go'])
                # WRF:3001 continue
                return v
            v = lax.fori_loop(0,_p154,_p155,v)
            v['k'] = jnp.where((_p153 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p154*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p150,_p151,v)
        v['i'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((v['myis'])+_p150*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3002 do k=1,lp1
        _p157 = ((jnp.bool_(True)) & (v['_go']==0))
        _p158 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p159(_p160,v):
            v = dict(v)
            v['k'] = jnp.where((_p157 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p160*(jnp.int32(1))), v['k'])
            # WRF:3003 do i=myis,myie
            _p161 = (((_p157 & (v['_go']==0))) & (v['_go']==0))
            _p162 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p163(_p164,v):
                v = dict(v)
                v['i'] = jnp.where((_p161 & (v['_go']==0)), jnp.int32((v['myis'])+_p164*(jnp.int32(1))), v['i'])
                # WRF:3004 sorc(i,k,9)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(9))-1].set(jnp.where((((_p161 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(9))-1]))
                v['_go'] = jnp.where(v['_go']==4912, 0, v['_go'])
                # WRF:3005 continue
                return v
            v = lax.fori_loop(0,_p162,_p163,v)
            v['i'] = jnp.where((_p161 & (v['_go']==0)), jnp.int32((v['myis'])+_p162*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p158,_p159,v)
        v['k'] = jnp.where((_p157 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p158*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3007 do i=myis,myie
        _p165 = ((jnp.bool_(True)) & (v['_go']==0))
        _p166 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p167(_p168,v):
            v = dict(v)
            v['i'] = jnp.where((_p165 & (v['_go']==0)), jnp.int32((v['myis'])+_p168*(jnp.int32(1))), v['i'])
            # WRF:3008 do k=1,lp1
            _p169 = (((_p165 & (v['_go']==0))) & (v['_go']==0))
            _p170 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p171(_p172,v):
                v = dict(v)
                v['k'] = jnp.where((_p169 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p172*(jnp.int32(1))), v['k'])
                # WRF:3009 vtmp3(i,k)=source(ixo(i,k),10)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p169 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(10))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3010 dsorc(i,k)=dsrce(ixo(i,k),10)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p169 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(10))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==5014, 0, v['_go'])
                # WRF:3011 continue
                return v
            v = lax.fori_loop(0,_p170,_p171,v)
            v['k'] = jnp.where((_p169 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p170*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p166,_p167,v)
        v['i'] = jnp.where((_p165 & (v['_go']==0)), jnp.int32((v['myis'])+_p166*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3012 do k=1,lp1
        _p173 = ((jnp.bool_(True)) & (v['_go']==0))
        _p174 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p175(_p176,v):
            v = dict(v)
            v['k'] = jnp.where((_p173 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p176*(jnp.int32(1))), v['k'])
            # WRF:3013 do i=myis,myie
            _p177 = (((_p173 & (v['_go']==0))) & (v['_go']==0))
            _p178 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p179(_p180,v):
                v = dict(v)
                v['i'] = jnp.where((_p177 & (v['_go']==0)), jnp.int32((v['myis'])+_p180*(jnp.int32(1))), v['i'])
                # WRF:3014 sorc(i,k,10)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(10))-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(10))-1]))
                v['_go'] = jnp.where(v['_go']==5012, 0, v['_go'])
                # WRF:3015 continue
                return v
            v = lax.fori_loop(0,_p178,_p179,v)
            v['i'] = jnp.where((_p177 & (v['_go']==0)), jnp.int32((v['myis'])+_p178*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p174,_p175,v)
        v['k'] = jnp.where((_p173 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p174*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3017 do i=myis,myie
        _p181 = ((jnp.bool_(True)) & (v['_go']==0))
        _p182 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p183(_p184,v):
            v = dict(v)
            v['i'] = jnp.where((_p181 & (v['_go']==0)), jnp.int32((v['myis'])+_p184*(jnp.int32(1))), v['i'])
            # WRF:3018 do k=1,lp1
            _p185 = (((_p181 & (v['_go']==0))) & (v['_go']==0))
            _p186 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p187(_p188,v):
                v = dict(v)
                v['k'] = jnp.where((_p185 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p188*(jnp.int32(1))), v['k'])
                # WRF:3019 vtmp3(i,k)=source(ixo(i,k),11)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p185 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(11))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3020 dsorc(i,k)=dsrce(ixo(i,k),11)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p185 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(11))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==5114, 0, v['_go'])
                # WRF:3021 continue
                return v
            v = lax.fori_loop(0,_p186,_p187,v)
            v['k'] = jnp.where((_p185 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p186*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p182,_p183,v)
        v['i'] = jnp.where((_p181 & (v['_go']==0)), jnp.int32((v['myis'])+_p182*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3022 do k=1,lp1
        _p189 = ((jnp.bool_(True)) & (v['_go']==0))
        _p190 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p191(_p192,v):
            v = dict(v)
            v['k'] = jnp.where((_p189 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p192*(jnp.int32(1))), v['k'])
            # WRF:3023 do i=myis,myie
            _p193 = (((_p189 & (v['_go']==0))) & (v['_go']==0))
            _p194 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p195(_p196,v):
                v = dict(v)
                v['i'] = jnp.where((_p193 & (v['_go']==0)), jnp.int32((v['myis'])+_p196*(jnp.int32(1))), v['i'])
                # WRF:3024 sorc(i,k,11)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(11))-1].set(jnp.where((((_p193 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(11))-1]))
                v['_go'] = jnp.where(v['_go']==5112, 0, v['_go'])
                # WRF:3025 continue
                return v
            v = lax.fori_loop(0,_p194,_p195,v)
            v['i'] = jnp.where((_p193 & (v['_go']==0)), jnp.int32((v['myis'])+_p194*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p190,_p191,v)
        v['k'] = jnp.where((_p189 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p190*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3027 do i=myis,myie
        _p197 = ((jnp.bool_(True)) & (v['_go']==0))
        _p198 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p199(_p200,v):
            v = dict(v)
            v['i'] = jnp.where((_p197 & (v['_go']==0)), jnp.int32((v['myis'])+_p200*(jnp.int32(1))), v['i'])
            # WRF:3028 do k=1,lp1
            _p201 = (((_p197 & (v['_go']==0))) & (v['_go']==0))
            _p202 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p203(_p204,v):
                v = dict(v)
                v['k'] = jnp.where((_p201 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p204*(jnp.int32(1))), v['k'])
                # WRF:3029 vtmp3(i,k)=source(ixo(i,k),12)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p201 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(12))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3030 dsorc(i,k)=dsrce(ixo(i,k),12)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p201 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(12))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==5214, 0, v['_go'])
                # WRF:3031 continue
                return v
            v = lax.fori_loop(0,_p202,_p203,v)
            v['k'] = jnp.where((_p201 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p202*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p198,_p199,v)
        v['i'] = jnp.where((_p197 & (v['_go']==0)), jnp.int32((v['myis'])+_p198*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3032 do k=1,lp1
        _p205 = ((jnp.bool_(True)) & (v['_go']==0))
        _p206 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p207(_p208,v):
            v = dict(v)
            v['k'] = jnp.where((_p205 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p208*(jnp.int32(1))), v['k'])
            # WRF:3033 do i=myis,myie
            _p209 = (((_p205 & (v['_go']==0))) & (v['_go']==0))
            _p210 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p211(_p212,v):
                v = dict(v)
                v['i'] = jnp.where((_p209 & (v['_go']==0)), jnp.int32((v['myis'])+_p212*(jnp.int32(1))), v['i'])
                # WRF:3034 sorc(i,k,12)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(12))-1].set(jnp.where((((_p209 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(12))-1]))
                v['_go'] = jnp.where(v['_go']==5212, 0, v['_go'])
                # WRF:3035 continue
                return v
            v = lax.fori_loop(0,_p210,_p211,v)
            v['i'] = jnp.where((_p209 & (v['_go']==0)), jnp.int32((v['myis'])+_p210*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p206,_p207,v)
        v['k'] = jnp.where((_p205 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p206*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3037 do i=myis,myie
        _p213 = ((jnp.bool_(True)) & (v['_go']==0))
        _p214 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p215(_p216,v):
            v = dict(v)
            v['i'] = jnp.where((_p213 & (v['_go']==0)), jnp.int32((v['myis'])+_p216*(jnp.int32(1))), v['i'])
            # WRF:3038 do k=1,lp1
            _p217 = (((_p213 & (v['_go']==0))) & (v['_go']==0))
            _p218 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p219(_p220,v):
                v = dict(v)
                v['k'] = jnp.where((_p217 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p220*(jnp.int32(1))), v['k'])
                # WRF:3039 vtmp3(i,k)=source(ixo(i,k),13)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(13))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3040 dsorc(i,k)=dsrce(ixo(i,k),13)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(13))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==5314, 0, v['_go'])
                # WRF:3041 continue
                return v
            v = lax.fori_loop(0,_p218,_p219,v)
            v['k'] = jnp.where((_p217 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p218*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p214,_p215,v)
        v['i'] = jnp.where((_p213 & (v['_go']==0)), jnp.int32((v['myis'])+_p214*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3042 do k=1,lp1
        _p221 = ((jnp.bool_(True)) & (v['_go']==0))
        _p222 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p223(_p224,v):
            v = dict(v)
            v['k'] = jnp.where((_p221 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p224*(jnp.int32(1))), v['k'])
            # WRF:3043 do i=myis,myie
            _p225 = (((_p221 & (v['_go']==0))) & (v['_go']==0))
            _p226 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p227(_p228,v):
                v = dict(v)
                v['i'] = jnp.where((_p225 & (v['_go']==0)), jnp.int32((v['myis'])+_p228*(jnp.int32(1))), v['i'])
                # WRF:3044 sorc(i,k,13)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(13))-1].set(jnp.where((((_p225 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(13))-1]))
                v['_go'] = jnp.where(v['_go']==5312, 0, v['_go'])
                # WRF:3045 continue
                return v
            v = lax.fori_loop(0,_p226,_p227,v)
            v['i'] = jnp.where((_p225 & (v['_go']==0)), jnp.int32((v['myis'])+_p226*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p222,_p223,v)
        v['k'] = jnp.where((_p221 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p222*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3047 do i=myis,myie
        _p229 = ((jnp.bool_(True)) & (v['_go']==0))
        _p230 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p231(_p232,v):
            v = dict(v)
            v['i'] = jnp.where((_p229 & (v['_go']==0)), jnp.int32((v['myis'])+_p232*(jnp.int32(1))), v['i'])
            # WRF:3048 do k=1,lp1
            _p233 = (((_p229 & (v['_go']==0))) & (v['_go']==0))
            _p234 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p235(_p236,v):
                v = dict(v)
                v['k'] = jnp.where((_p233 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p236*(jnp.int32(1))), v['k'])
                # WRF:3049 vtmp3(i,k)=source(ixo(i,k),14)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p233 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['source'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(14))-1]), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3050 dsorc(i,k)=dsrce(ixo(i,k),14)
                v['dsorc'] = v['dsorc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p233 & (v['_go']==0))) & (v['_go']==0)), _r(_r(_TABLES['dsrce'])[(v['ixo'][(v['i'])-1, (v['k'])-1])-1, (jnp.int32(14))-1]), v['dsorc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==5414, 0, v['_go'])
                # WRF:3051 continue
                return v
            v = lax.fori_loop(0,_p234,_p235,v)
            v['k'] = jnp.where((_p233 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p234*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p230,_p231,v)
        v['i'] = jnp.where((_p229 & (v['_go']==0)), jnp.int32((v['myis'])+_p230*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3052 do k=1,lp1
        _p237 = ((jnp.bool_(True)) & (v['_go']==0))
        _p238 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p239(_p240,v):
            v = dict(v)
            v['k'] = jnp.where((_p237 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p240*(jnp.int32(1))), v['k'])
            # WRF:3053 do i=myis,myie
            _p241 = (((_p237 & (v['_go']==0))) & (v['_go']==0))
            _p242 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p243(_p244,v):
                v = dict(v)
                v['i'] = jnp.where((_p241 & (v['_go']==0)), jnp.int32((v['myis'])+_p244*(jnp.int32(1))), v['i'])
                # WRF:3054 sorc(i,k,14)=vtmp3(i,k)+dt(i,k)*dsorc(i,k)
                v['sorc'] = v['sorc'].at[(v['i'])-1, (v['k'])-1, (jnp.int32(14))-1].set(jnp.where((((_p241 & (v['_go']==0))) & (v['_go']==0)), _r((v['vtmp3'][(v['i'])-1, (v['k'])-1] + (v['dt'][(v['i'])-1, (v['k'])-1] * v['dsorc'][(v['i'])-1, (v['k'])-1]))), v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(14))-1]))
                v['_go'] = jnp.where(v['_go']==5412, 0, v['_go'])
                # WRF:3055 continue
                return v
            v = lax.fori_loop(0,_p242,_p243,v)
            v['i'] = jnp.where((_p241 & (v['_go']==0)), jnp.int32((v['myis'])+_p242*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p238,_p239,v)
        v['k'] = jnp.where((_p237 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p238*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3065 do k=1,lp1
        _p245 = ((jnp.bool_(True)) & (v['_go']==0))
        _p246 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p247(_p248,v):
            v = dict(v)
            v['k'] = jnp.where((_p245 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p248*(jnp.int32(1))), v['k'])
            # WRF:3066 do i=myis,myie
            _p249 = (((_p245 & (v['_go']==0))) & (v['_go']==0))
            _p250 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p251(_p252,v):
                v = dict(v)
                v['i'] = jnp.where((_p249 & (v['_go']==0)), jnp.int32((v['myis'])+_p252*(jnp.int32(1))), v['i'])
                # WRF:3067 ss1(i,k)=sorc(i,k,11)+sorc(i,k,12)+sorc(i,k,14)
                v['ss1'] = v['ss1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p249 & (v['_go']==0))) & (v['_go']==0)), _r(((v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(11))-1] + v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(12))-1]) + v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(14))-1])), v['ss1'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==131, 0, v['_go'])
                # WRF:3068 continue
                return v
            v = lax.fori_loop(0,_p250,_p251,v)
            v['i'] = jnp.where((_p249 & (v['_go']==0)), jnp.int32((v['myis'])+_p250*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p246,_p247,v)
        v['k'] = jnp.where((_p245 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p246*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3069 do k=1,lp1
        _p253 = ((jnp.bool_(True)) & (v['_go']==0))
        _p254 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p255(_p256,v):
            v = dict(v)
            v['k'] = jnp.where((_p253 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p256*(jnp.int32(1))), v['k'])
            # WRF:3070 do i=myis,myie
            _p257 = (((_p253 & (v['_go']==0))) & (v['_go']==0))
            _p258 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p259(_p260,v):
                v = dict(v)
                v['i'] = jnp.where((_p257 & (v['_go']==0)), jnp.int32((v['myis'])+_p260*(jnp.int32(1))), v['i'])
                # WRF:3071 csour(i,k)=sorc(i,k,9)+sorc(i,k,10)
                v['csour'] = v['csour'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(9))-1] + v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(10))-1])), v['csour'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==143, 0, v['_go'])
                # WRF:3072 continue
                return v
            v = lax.fori_loop(0,_p258,_p259,v)
            v['i'] = jnp.where((_p257 & (v['_go']==0)), jnp.int32((v['myis'])+_p258*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p254,_p255,v)
        v['k'] = jnp.where((_p253 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p254*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3078 do k=1,lp1
        _p261 = ((jnp.bool_(True)) & (v['_go']==0))
        _p262 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p263(_p264,v):
            v = dict(v)
            v['k'] = jnp.where((_p261 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p264*(jnp.int32(1))), v['k'])
            # WRF:3079 do i=myis,myie
            _p265 = (((_p261 & (v['_go']==0))) & (v['_go']==0))
            _p266 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p267(_p268,v):
                v = dict(v)
                v['i'] = jnp.where((_p265 & (v['_go']==0)), jnp.int32((v['myis'])+_p268*(jnp.int32(1))), v['i'])
                # WRF:3080 tc(i,k)=temp(i,k)*temp(i,k)*temp(i,k)*temp(i,k)
                v['tc'] = v['tc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p265 & (v['_go']==0))) & (v['_go']==0)), _r((((v['temp'][(v['i'])-1, (v['k'])-1] * v['temp'][(v['i'])-1, (v['k'])-1]) * v['temp'][(v['i'])-1, (v['k'])-1]) * v['temp'][(v['i'])-1, (v['k'])-1])), v['tc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==901, 0, v['_go'])
                # WRF:3081 continue
                return v
            v = lax.fori_loop(0,_p266,_p267,v)
            v['i'] = jnp.where((_p265 & (v['_go']==0)), jnp.int32((v['myis'])+_p266*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p262,_p263,v)
        v['k'] = jnp.where((_p261 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p262*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3082 do k=1,l
        _p269 = ((jnp.bool_(True)) & (v['_go']==0))
        _p270 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p271(_p272,v):
            v = dict(v)
            v['k'] = jnp.where((_p269 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p272*(jnp.int32(1))), v['k'])
            # WRF:3083 do i=myis,myie
            _p273 = (((_p269 & (v['_go']==0))) & (v['_go']==0))
            _p274 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p275(_p276,v):
                v = dict(v)
                v['i'] = jnp.where((_p273 & (v['_go']==0)), jnp.int32((v['myis'])+_p276*(jnp.int32(1))), v['i'])
                # WRF:3084 oss(i,k+1)=sorc(i,k+1,13)-sorc(i,k,13)
                v['oss'] = v['oss'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p273 & (v['_go']==0))) & (v['_go']==0)), _r((v['sorc'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(13))-1] - v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(13))-1])), v['oss'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3085 css(i,k+1)=csour(i,k+1)-csour(i,k)
                v['css'] = v['css'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p273 & (v['_go']==0))) & (v['_go']==0)), _r((v['csour'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['csour'][(v['i'])-1, (v['k'])-1])), v['css'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3086 dtc(i,k+1)=tc(i,k+1)-tc(i,k)
                v['dtc'] = v['dtc'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p273 & (v['_go']==0))) & (v['_go']==0)), _r((v['tc'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['tc'][(v['i'])-1, (v['k'])-1])), v['dtc'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3087 ss2(i,k+1)=ss1(i,k+1)-ss1(i,k)
                v['ss2'] = v['ss2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p273 & (v['_go']==0))) & (v['_go']==0)), _r((v['ss1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ss1'][(v['i'])-1, (v['k'])-1])), v['ss2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==903, 0, v['_go'])
                # WRF:3088 continue
                return v
            v = lax.fori_loop(0,_p274,_p275,v)
            v['i'] = jnp.where((_p273 & (v['_go']==0)), jnp.int32((v['myis'])+_p274*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p270,_p271,v)
        v['k'] = jnp.where((_p269 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p270*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3128 do k=1,l
        _p277 = ((jnp.bool_(True)) & (v['_go']==0))
        _p278 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p279(_p280,v):
            v = dict(v)
            v['k'] = jnp.where((_p277 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p280*(jnp.int32(1))), v['k'])
            # WRF:3129 do i=myis,myie
            _p281 = (((_p277 & (v['_go']==0))) & (v['_go']==0))
            _p282 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p283(_p284,v):
                v = dict(v)
                v['i'] = jnp.where((_p281 & (v['_go']==0)), jnp.int32((v['myis'])+_p284*(jnp.int32(1))), v['i'])
                # WRF:3130 avephi(i,k)=totphi(i,k+1)
                v['avephi'] = v['avephi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p281 & (v['_go']==0))) & (v['_go']==0)), _r(v['totphi'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]), v['avephi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==3021, 0, v['_go'])
                # WRF:3131 continue
                return v
            v = lax.fori_loop(0,_p282,_p283,v)
            v['i'] = jnp.where((_p281 & (v['_go']==0)), jnp.int32((v['myis'])+_p282*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p278,_p279,v)
        v['k'] = jnp.where((_p277 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p278*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3137 do i=myis,myie
        _p285 = ((jnp.bool_(True)) & (v['_go']==0))
        _p286 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p287(_p288,v):
            v = dict(v)
            v['i'] = jnp.where((_p285 & (v['_go']==0)), jnp.int32((v['myis'])+_p288*(jnp.int32(1))), v['i'])
            # WRF:3138 avephi(i,lp1)=avephi(i,lm1)+emx1(i)
            v['avephi'] = v['avephi'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p285 & (v['_go']==0))) & (v['_go']==0)), _r((v['avephi'][(v['i'])-1, (v['lm1'])-1] + v['emx1'][(v['i'])-1])), v['avephi'][(v['i'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==803, 0, v['_go'])
            # WRF:3139 continue
            return v
        v = lax.fori_loop(0,_p286,_p287,v)
        v['i'] = jnp.where((_p285 & (v['_go']==0)), jnp.int32((v['myis'])+_p286*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3141 call e1e290(e1cts1,e1cts2,e1flx,e1ctw1,e1ctw2,emiss, fxo,dt,fxoe2,dte2,avephi,temp,t, h16e1,ten,hp1,h28e1,haf, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte                )
        _p289 = lax.cond(((jnp.bool_(True)) & (v['_go']==0)), lambda _a: _e1e290(*_a), lambda _a: _a, (v['e1cts1'], v['e1cts2'], v['e1flx'], v['e1ctw1'], v['e1ctw2'], v['emiss'], v['fxo'], v['dt'], v['fxoe2'], v['dte2'], v['avephi'], v['temp'], v['t'], v['h16e1'], v['ten'], v['hp1'], v['h28e1'], v['haf'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],))
        v['e1cts1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[0]), v['e1cts1'])
        v['e1cts2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[1]), v['e1cts2'])
        v['e1flx'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[2]), v['e1flx'])
        v['e1ctw1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[3]), v['e1ctw1'])
        v['e1ctw2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[4]), v['e1ctw2'])
        v['emiss'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[5]), v['emiss'])
        v['fxo'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[6]), v['fxo'])
        v['dt'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[7]), v['dt'])
        v['fxoe2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[8]), v['fxoe2'])
        v['dte2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[9]), v['dte2'])
        v['avephi'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[10]), v['avephi'])
        v['temp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[11]), v['temp'])
        v['t'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[12]), v['t'])
        v['h16e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[13]), v['h16e1'])
        v['ten'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[14]), v['ten'])
        v['hp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[15]), v['hp1'])
        v['h28e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[16]), v['h28e1'])
        v['haf'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p289[17]), v['haf'])
        v['ids'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[18]), v['ids'])
        v['ide'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[19]), v['ide'])
        v['jds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[20]), v['jds'])
        v['jde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[21]), v['jde'])
        v['kds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[22]), v['kds'])
        v['kde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[23]), v['kde'])
        v['ims'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[24]), v['ims'])
        v['ime'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[25]), v['ime'])
        v['jms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[26]), v['jms'])
        v['jme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[27]), v['jme'])
        v['kms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[28]), v['kms'])
        v['kme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[29]), v['kme'])
        v['its'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[30]), v['its'])
        v['ite'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[31]), v['ite'])
        v['jts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[32]), v['jts'])
        v['jte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[33]), v['jte'])
        v['kts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[34]), v['kts'])
        v['kte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p289[35]), v['kte'])
        # WRF:3149 do k=1,l
        _p290 = ((jnp.bool_(True)) & (v['_go']==0))
        _p291 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p292(_p293,v):
            v = dict(v)
            v['k'] = jnp.where((_p290 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p293*(jnp.int32(1))), v['k'])
            # WRF:3150 do i=myis,myie
            _p294 = (((_p290 & (v['_go']==0))) & (v['_go']==0))
            _p295 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p296(_p297,v):
                v = dict(v)
                v['i'] = jnp.where((_p294 & (v['_go']==0)), jnp.int32((v['myis'])+_p297*(jnp.int32(1))), v['i'])
                # WRF:3151 fac1(i,k)=bo3rnd(2)*tphio3(i,k+1)/toto3(i,k+1)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p294 & (v['_go']==0))) & (v['_go']==0)), _r(((v['bo3rnd'][(jnp.int32(2))-1] * v['tphio3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) / v['toto3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:3152 to3spc(i,k)=haf*(fac1(i,k)* (sqrt(one+(four*ao3rnd(2)*toto3(i,k+1))/fac1(i,k))-one))
                v['to3spc'] = v['to3spc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p294 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['fac1'][(v['i'])-1, (v['k'])-1] * (jnp.sqrt((v['one'] + (((v['four'] * v['ao3rnd'][(jnp.int32(2))-1]) * v['toto3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) / v['fac1'][(v['i'])-1, (v['k'])-1]))) - v['one'])))), v['to3spc'][(v['i'])-1, (v['k'])-1]))
                # WRF:3156 to3sp(i,k)=exp(hm1ez*(to3spc(i,k)+sko3r*totvo2(i,k+1)))
                v['to3sp'] = v['to3sp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p294 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (v['to3spc'][(v['i'])-1, (v['k'])-1] + (_r(_TABLES['sko3r']) * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))))), v['to3sp'][(v['i'])-1, (v['k'])-1]))
                # WRF:3157 over1d(i,k)=exp(hm1ez*(sqrt(ab15wd*totphi(i,k+1))+ skc1r*totvo2(i,k+1)))
                v['over1d'] = v['over1d'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p294 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (jnp.sqrt((_r(_TABLES['ab15wd']) * v['totphi'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])) + (_r(_TABLES['skc1r']) * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))))), v['over1d'][(v['i'])-1, (v['k'])-1]))
                # WRF:3162 cnttau(i,k)=exp(hm1ez*totvo2(i,k+1))
                v['cnttau'] = v['cnttau'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p294 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))), v['cnttau'][(v['i'])-1, (v['k'])-1]))
                # WRF:3163 totevv(i,k)=1./cnttau(i,k)
                v['totevv'] = v['totevv'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p294 & (v['_go']==0))) & (v['_go']==0)), _r((_r(1.) / v['cnttau'][(v['i'])-1, (v['k'])-1])), v['totevv'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==302, 0, v['_go'])
                # WRF:3164 continue
                return v
            v = lax.fori_loop(0,_p295,_p296,v)
            v['i'] = jnp.where((_p294 & (v['_go']==0)), jnp.int32((v['myis'])+_p295*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p291,_p292,v)
        v['k'] = jnp.where((_p290 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p291*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3165 do k=1,l
        _p298 = ((jnp.bool_(True)) & (v['_go']==0))
        _p299 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p300(_p301,v):
            v = dict(v)
            v['k'] = jnp.where((_p298 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p301*(jnp.int32(1))), v['k'])
            # WRF:3166 do i=myis,myie
            _p302 = (((_p298 & (v['_go']==0))) & (v['_go']==0))
            _p303 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p304(_p305,v):
                v = dict(v)
                v['i'] = jnp.where((_p302 & (v['_go']==0)), jnp.int32((v['myis'])+_p305*(jnp.int32(1))), v['i'])
                # WRF:3167 co2sp(i,k+1)=over1d(i,k)*co21(i,1,k+1)
                v['co2sp'] = v['co2sp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p302 & (v['_go']==0))) & (v['_go']==0)), _r((v['over1d'][(v['i'])-1, (v['k'])-1] * v['co21'][(v['i'])-1, (jnp.int32(1))-1, ((v['k'] + jnp.int32(1)))-1])), v['co2sp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==3022, 0, v['_go'])
                # WRF:3168 continue
                return v
            v = lax.fori_loop(0,_p303,_p304,v)
            v['i'] = jnp.where((_p302 & (v['_go']==0)), jnp.int32((v['myis'])+_p303*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p299,_p300,v)
        v['k'] = jnp.where((_p298 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p299*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3169 do k=1,l
        _p306 = ((jnp.bool_(True)) & (v['_go']==0))
        _p307 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p308(_p309,v):
            v = dict(v)
            v['k'] = jnp.where((_p306 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p309*(jnp.int32(1))), v['k'])
            # WRF:3170 do i=myis,myie
            _p310 = (((_p306 & (v['_go']==0))) & (v['_go']==0))
            _p311 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p312(_p313,v):
                v = dict(v)
                v['i'] = jnp.where((_p310 & (v['_go']==0)), jnp.int32((v['myis'])+_p313*(jnp.int32(1))), v['i'])
                # WRF:3171 co21(i,k+1,1)=co21(i,k+1,1)*over1d(i,k)
                v['co21'] = v['co21'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1].set(jnp.where((((_p310 & (v['_go']==0))) & (v['_go']==0)), _r((v['co21'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1] * v['over1d'][(v['i'])-1, (v['k'])-1])), v['co21'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1]))
                v['_go'] = jnp.where(v['_go']==3023, 0, v['_go'])
                # WRF:3172 continue
                return v
            v = lax.fori_loop(0,_p311,_p312,v)
            v['i'] = jnp.where((_p310 & (v['_go']==0)), jnp.int32((v['myis'])+_p311*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p307,_p308,v)
        v['k'] = jnp.where((_p306 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p307*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3174 do i=myis,myie
        _p314 = ((jnp.bool_(True)) & (v['_go']==0))
        _p315 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p316(_p317,v):
            v = dict(v)
            v['i'] = jnp.where((_p314 & (v['_go']==0)), jnp.int32((v['myis'])+_p317*(jnp.int32(1))), v['i'])
            # WRF:3175 rlog(i,1)=over1d(i,1)*co2nbl(i,1)
            v['rlog'] = v['rlog'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p314 & (v['_go']==0))) & (v['_go']==0)), _r((v['over1d'][(v['i'])-1, (jnp.int32(1))-1] * v['co2nbl'][(v['i'])-1, (jnp.int32(1))-1])), v['rlog'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1808, 0, v['_go'])
            # WRF:3176 continue
            return v
        v = lax.fori_loop(0,_p315,_p316,v)
        v['i'] = jnp.where((_p314 & (v['_go']==0)), jnp.int32((v['myis'])+_p315*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3180 do k=2,lp1
        _p318 = ((jnp.bool_(True)) & (v['_go']==0))
        _p319 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p320(_p321,v):
            v = dict(v)
            v['k'] = jnp.where((_p318 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p321*(jnp.int32(1))), v['k'])
            # WRF:3181 do i=myis,myie
            _p322 = (((_p318 & (v['_go']==0))) & (v['_go']==0))
            _p323 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p324(_p325,v):
                v = dict(v)
                v['i'] = jnp.where((_p322 & (v['_go']==0)), jnp.int32((v['myis'])+_p325*(jnp.int32(1))), v['i'])
                # WRF:3182 flx(i,k)= (tc(i,1)*e1flx(i,k) +ss1(i,1)*cnttau(i,k-1) +sorc(i,1,13)*to3sp(i,k-1) +csour(i,1)*co2sp(i,k)) *cldfac(i,1,k)
                v['flx'] = v['flx'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p322 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['tc'][(v['i'])-1, (jnp.int32(1))-1] * v['e1flx'][(v['i'])-1, (v['k'])-1]) + (v['ss1'][(v['i'])-1, (jnp.int32(1))-1] * v['cnttau'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])) + (v['sorc'][(v['i'])-1, (jnp.int32(1))-1, (jnp.int32(13))-1] * v['to3sp'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])) + (v['csour'][(v['i'])-1, (jnp.int32(1))-1] * v['co2sp'][(v['i'])-1, (v['k'])-1])) * v['cldfac'][(v['i'])-1, (jnp.int32(1))-1, (v['k'])-1])), v['flx'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==305, 0, v['_go'])
                # WRF:3187 continue
                return v
            v = lax.fori_loop(0,_p323,_p324,v)
            v['i'] = jnp.where((_p322 & (v['_go']==0)), jnp.int32((v['myis'])+_p323*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p319,_p320,v)
        v['k'] = jnp.where((_p318 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p319*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3188 do i=myis,myie
        _p326 = ((jnp.bool_(True)) & (v['_go']==0))
        _p327 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p328(_p329,v):
            v = dict(v)
            v['i'] = jnp.where((_p326 & (v['_go']==0)), jnp.int32((v['myis'])+_p329*(jnp.int32(1))), v['i'])
            # WRF:3189 flx(i,1)= tc(i,1)*e1flx(i,1)+ss1(i,1)+sorc(i,1,13) +csour(i,1)
            v['flx'] = v['flx'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p326 & (v['_go']==0))) & (v['_go']==0)), _r(((((v['tc'][(v['i'])-1, (jnp.int32(1))-1] * v['e1flx'][(v['i'])-1, (jnp.int32(1))-1]) + v['ss1'][(v['i'])-1, (jnp.int32(1))-1]) + v['sorc'][(v['i'])-1, (jnp.int32(1))-1, (jnp.int32(13))-1]) + v['csour'][(v['i'])-1, (jnp.int32(1))-1])), v['flx'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==307, 0, v['_go'])
            # WRF:3191 continue
            return v
        v = lax.fori_loop(0,_p327,_p328,v)
        v['i'] = jnp.where((_p326 & (v['_go']==0)), jnp.int32((v['myis'])+_p327*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3193 do kp=2,lp1
        _p330 = ((jnp.bool_(True)) & (v['_go']==0))
        _p331 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p332(_p333,v):
            v = dict(v)
            v['kp'] = jnp.where((_p330 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p333*(jnp.int32(1))), v['kp'])
            # WRF:3194 do i=myis,myie
            _p334 = (((_p330 & (v['_go']==0))) & (v['_go']==0))
            _p335 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p336(_p337,v):
                v = dict(v)
                v['i'] = jnp.where((_p334 & (v['_go']==0)), jnp.int32((v['myis'])+_p337*(jnp.int32(1))), v['i'])
                # WRF:3195 flx(i,1)=flx(i,1)+(oss(i,kp)*to3sp(i,kp-1) +ss2(i,kp)*cnttau(i,kp-1) +css(i,kp)*co21(i,kp,1) +dtc(i,kp)*emiss(i,kp-1))*cldfac(i,kp,1)
                v['flx'] = v['flx'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p334 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx'][(v['i'])-1, (jnp.int32(1))-1] + (((((v['oss'][(v['i'])-1, (v['kp'])-1] * v['to3sp'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1]) + (v['ss2'][(v['i'])-1, (v['kp'])-1] * v['cnttau'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1])) + (v['css'][(v['i'])-1, (v['kp'])-1] * v['co21'][(v['i'])-1, (v['kp'])-1, (jnp.int32(1))-1])) + (v['dtc'][(v['i'])-1, (v['kp'])-1] * v['emiss'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1])) * v['cldfac'][(v['i'])-1, (v['kp'])-1, (jnp.int32(1))-1]))), v['flx'][(v['i'])-1, (jnp.int32(1))-1]))
                v['_go'] = jnp.where(v['_go']==303, 0, v['_go'])
                # WRF:3199 continue
                return v
            v = lax.fori_loop(0,_p335,_p336,v)
            v['i'] = jnp.where((_p334 & (v['_go']==0)), jnp.int32((v['myis'])+_p335*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p331,_p332,v)
        v['kp'] = jnp.where((_p330 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p331*(jnp.int32(1))), v['kp'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3203 call spa88(excts,ctso3,gxcts,sorc,csour, cldfac,temp,press,var1,var2, p,delp,delp2,totvo2,to3sp,to3spc, co2sp1,co2sp2,co2sp, apcm,bpcm,atpcm,btpcm,acomb,bcomb,betacm, h25e2,one,h44194m2,h1p41819,haf,hm1ez,two, radcon, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte                     )
        _p338 = lax.cond(((jnp.bool_(True)) & (v['_go']==0)), lambda _a: _spa88(*_a), lambda _a: _a, (v['excts'], v['ctso3'], v['gxcts'], v['sorc'], v['csour'], v['cldfac'], v['temp'], v['press'], v['var1'], v['var2'], v['p'], v['delp'], v['delp2'], v['totvo2'], v['to3sp'], v['to3spc'], v['co2sp1'], v['co2sp2'], v['co2sp'], v['apcm'], v['bpcm'], v['atpcm'], v['btpcm'], v['acomb'], v['bcomb'], v['betacm'], v['h25e2'], v['one'], v['h44194m2'], v['h1p41819'], v['haf'], v['hm1ez'], v['two'], v['radcon'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],))
        v['excts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[0]), v['excts'])
        v['ctso3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[1]), v['ctso3'])
        v['gxcts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[2]), v['gxcts'])
        v['sorc'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[3]), v['sorc'])
        v['csour'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[4]), v['csour'])
        v['cldfac'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[5]), v['cldfac'])
        v['temp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[6]), v['temp'])
        v['press'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[7]), v['press'])
        v['var1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[8]), v['var1'])
        v['var2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[9]), v['var2'])
        v['p'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[10]), v['p'])
        v['delp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[11]), v['delp'])
        v['delp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[12]), v['delp2'])
        v['totvo2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[13]), v['totvo2'])
        v['to3sp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[14]), v['to3sp'])
        v['to3spc'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[15]), v['to3spc'])
        v['co2sp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[16]), v['co2sp1'])
        v['co2sp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[17]), v['co2sp2'])
        v['co2sp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[18]), v['co2sp'])
        v['apcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[19]), v['apcm'])
        v['bpcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[20]), v['bpcm'])
        v['atpcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[21]), v['atpcm'])
        v['btpcm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[22]), v['btpcm'])
        v['acomb'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[23]), v['acomb'])
        v['bcomb'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[24]), v['bcomb'])
        v['betacm'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[25]), v['betacm'])
        v['h25e2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[26]), v['h25e2'])
        v['one'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[27]), v['one'])
        v['h44194m2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[28]), v['h44194m2'])
        v['h1p41819'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[29]), v['h1p41819'])
        v['haf'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[30]), v['haf'])
        v['hm1ez'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[31]), v['hm1ez'])
        v['two'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[32]), v['two'])
        v['radcon'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p338[33]), v['radcon'])
        v['ids'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[34]), v['ids'])
        v['ide'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[35]), v['ide'])
        v['jds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[36]), v['jds'])
        v['jde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[37]), v['jde'])
        v['kds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[38]), v['kds'])
        v['kde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[39]), v['kde'])
        v['ims'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[40]), v['ims'])
        v['ime'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[41]), v['ime'])
        v['jms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[42]), v['jms'])
        v['jme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[43]), v['jme'])
        v['kms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[44]), v['kms'])
        v['kme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[45]), v['kme'])
        v['its'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[46]), v['its'])
        v['ite'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[47]), v['ite'])
        v['jts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[48]), v['jts'])
        v['jte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[49]), v['jte'])
        v['kts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[50]), v['kts'])
        v['kte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p338[51]), v['kte'])
        # WRF:3221 do i=myis,myie
        _p339 = ((jnp.bool_(True)) & (v['_go']==0))
        _p340 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p341(_p342,v):
            v = dict(v)
            v['i'] = jnp.where((_p339 & (v['_go']==0)), jnp.int32((v['myis'])+_p342*(jnp.int32(1))), v['i'])
            # WRF:3222 vtmp3(i,1)=1.
            v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p339 & (v['_go']==0))) & (v['_go']==0)), _r(_r(1.)), v['vtmp3'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==998, 0, v['_go'])
            # WRF:3223 continue
            return v
        v = lax.fori_loop(0,_p340,_p341,v)
        v['i'] = jnp.where((_p339 & (v['_go']==0)), jnp.int32((v['myis'])+_p340*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3224 do k=1,l
        _p343 = ((jnp.bool_(True)) & (v['_go']==0))
        _p344 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p345(_p346,v):
            v = dict(v)
            v['k'] = jnp.where((_p343 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p346*(jnp.int32(1))), v['k'])
            # WRF:3225 do i=myis,myie
            _p347 = (((_p343 & (v['_go']==0))) & (v['_go']==0))
            _p348 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p349(_p350,v):
                v = dict(v)
                v['i'] = jnp.where((_p347 & (v['_go']==0)), jnp.int32((v['myis'])+_p350*(jnp.int32(1))), v['i'])
                # WRF:3226 vtmp3(i,k+1)=cnttau(i,k)*cldfac(i,k+1,1)
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p347 & (v['_go']==0))) & (v['_go']==0)), _r((v['cnttau'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['vtmp3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==999, 0, v['_go'])
                # WRF:3227 continue
                return v
            v = lax.fori_loop(0,_p348,_p349,v)
            v['i'] = jnp.where((_p347 & (v['_go']==0)), jnp.int32((v['myis'])+_p348*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p344,_p345,v)
        v['k'] = jnp.where((_p343 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p344*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3228 do k=1,l
        _p351 = ((jnp.bool_(True)) & (v['_go']==0))
        _p352 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p353(_p354,v):
            v = dict(v)
            v['k'] = jnp.where((_p351 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p354*(jnp.int32(1))), v['k'])
            # WRF:3229 do i=myis,myie
            _p355 = (((_p351 & (v['_go']==0))) & (v['_go']==0))
            _p356 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p357(_p358,v):
                v = dict(v)
                v['i'] = jnp.where((_p355 & (v['_go']==0)), jnp.int32((v['myis'])+_p358*(jnp.int32(1))), v['i'])
                # WRF:3230 cts(i,k)=radcon*delp(i,k)*(tc(i,k)* (e1ctw2(i,k)*cldfac(i,k+1,1)-e1ctw1(i,k)*cldfac(i,k,1)) + ss1(i,k)*(vtmp3(i,k+1)-vtmp3(i,k)))
                v['cts'] = v['cts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p355 & (v['_go']==0))) & (v['_go']==0)), _r(((v['radcon'] * v['delp'][(v['i'])-1, (v['k'])-1]) * ((v['tc'][(v['i'])-1, (v['k'])-1] * ((v['e1ctw2'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1]) - (v['e1ctw1'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, (v['k'])-1, (jnp.int32(1))-1]))) + (v['ss1'][(v['i'])-1, (v['k'])-1] * (v['vtmp3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['vtmp3'][(v['i'])-1, (v['k'])-1]))))), v['cts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1001, 0, v['_go'])
                # WRF:3233 continue
                return v
            v = lax.fori_loop(0,_p356,_p357,v)
            v['i'] = jnp.where((_p355 & (v['_go']==0)), jnp.int32((v['myis'])+_p356*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p352,_p353,v)
        v['k'] = jnp.where((_p351 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p352*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3235 do k=1,l
        _p359 = ((jnp.bool_(True)) & (v['_go']==0))
        _p360 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p361(_p362,v):
            v = dict(v)
            v['k'] = jnp.where((_p359 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p362*(jnp.int32(1))), v['k'])
            # WRF:3236 do i=myis,myie
            _p363 = (((_p359 & (v['_go']==0))) & (v['_go']==0))
            _p364 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p365(_p366,v):
                v = dict(v)
                v['i'] = jnp.where((_p363 & (v['_go']==0)), jnp.int32((v['myis'])+_p366*(jnp.int32(1))), v['i'])
                # WRF:3237 vtmp3(i,k)=tc(i,k)*(cldfac(i,k,1)*(e1cts1(i,k)-e1ctw1(i,k)) - cldfac(i,k+1,1)*(e1cts2(i,k)-e1ctw2(i,k)))
                v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p363 & (v['_go']==0))) & (v['_go']==0)), _r((v['tc'][(v['i'])-1, (v['k'])-1] * ((v['cldfac'][(v['i'])-1, (v['k'])-1, (jnp.int32(1))-1] * (v['e1cts1'][(v['i'])-1, (v['k'])-1] - v['e1ctw1'][(v['i'])-1, (v['k'])-1])) - (v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1] * (v['e1cts2'][(v['i'])-1, (v['k'])-1] - v['e1ctw2'][(v['i'])-1, (v['k'])-1]))))), v['vtmp3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1011, 0, v['_go'])
                # WRF:3239 continue
                return v
            v = lax.fori_loop(0,_p364,_p365,v)
            v['i'] = jnp.where((_p363 & (v['_go']==0)), jnp.int32((v['myis'])+_p364*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p360,_p361,v)
        v['k'] = jnp.where((_p359 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p360*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3240 do i=myis,myie
        _p367 = ((jnp.bool_(True)) & (v['_go']==0))
        _p368 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p369(_p370,v):
            v = dict(v)
            v['i'] = jnp.where((_p367 & (v['_go']==0)), jnp.int32((v['myis'])+_p370*(jnp.int32(1))), v['i'])
            # WRF:3241 flx1e1(i)=tc(i,lp1)*cldfac(i,lp1,1)* (e1cts1(i,lp1)-e1ctw1(i,lp1))
            v['flx1e1'] = v['flx1e1'].at[(v['i'])-1].set(jnp.where((((_p367 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tc'][(v['i'])-1, (v['lp1'])-1] * v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1]) * (v['e1cts1'][(v['i'])-1, (v['lp1'])-1] - v['e1ctw1'][(v['i'])-1, (v['lp1'])-1]))), v['flx1e1'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1012, 0, v['_go'])
            # WRF:3243 continue
            return v
        v = lax.fori_loop(0,_p368,_p369,v)
        v['i'] = jnp.where((_p367 & (v['_go']==0)), jnp.int32((v['myis'])+_p368*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3244 do k=1,l
        _p371 = ((jnp.bool_(True)) & (v['_go']==0))
        _p372 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p373(_p374,v):
            v = dict(v)
            v['k'] = jnp.where((_p371 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p374*(jnp.int32(1))), v['k'])
            # WRF:3245 do i=myis,myie
            _p375 = (((_p371 & (v['_go']==0))) & (v['_go']==0))
            _p376 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p377(_p378,v):
                v = dict(v)
                v['i'] = jnp.where((_p375 & (v['_go']==0)), jnp.int32((v['myis'])+_p378*(jnp.int32(1))), v['i'])
                # WRF:3246 flx1e1(i)=flx1e1(i)+vtmp3(i,k)
                v['flx1e1'] = v['flx1e1'].at[(v['i'])-1].set(jnp.where((((_p375 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx1e1'][(v['i'])-1] + v['vtmp3'][(v['i'])-1, (v['k'])-1])), v['flx1e1'][(v['i'])-1]))
                v['_go'] = jnp.where(v['_go']==1013, 0, v['_go'])
                # WRF:3247 continue
                return v
            v = lax.fori_loop(0,_p376,_p377,v)
            v['i'] = jnp.where((_p375 & (v['_go']==0)), jnp.int32((v['myis'])+_p376*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1014, 0, v['_go'])
            # WRF:3248 continue
            return v
        v = lax.fori_loop(0,_p372,_p373,v)
        v['k'] = jnp.where((_p371 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p372*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3254 do k=2,lm1
        _p379 = ((jnp.bool_(True)) & (v['_go']==0))
        _p380 = jnp.maximum(jnp.int32(0), ((v['lm1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p381(_p382,v):
            v = dict(v)
            v['k'] = jnp.where((_p379 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p382*(jnp.int32(1))), v['k'])
            # WRF:3255 klen=k
            v['klen'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['k']), v['klen'])
            # WRF:3257 do kk=1,lp1-k
            _p383 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p384 = jnp.maximum(jnp.int32(0), (((v['lp1'] - v['k']))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p385(_p386,v):
                v = dict(v)
                v['kk'] = jnp.where((_p383 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p386*(jnp.int32(1))), v['kk'])
                # WRF:3258 do i=myis,myie
                _p387 = (((_p383 & (v['_go']==0))) & (v['_go']==0))
                _p388 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p389(_p390,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p387 & (v['_go']==0)), jnp.int32((v['myis'])+_p390*(jnp.int32(1))), v['i'])
                    # WRF:3259 avephi(i,kk+k-1)=totphi(i,kk+k)-totphi(i,k)
                    v['avephi'] = v['avephi'].at[(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p387 & (v['_go']==0))) & (v['_go']==0)), _r((v['totphi'][(v['i'])-1, ((v['kk'] + v['k']))-1] - v['totphi'][(v['i'])-1, (v['k'])-1])), v['avephi'][(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1]))
                    v['_go'] = jnp.where(v['_go']==3218, 0, v['_go'])
                    # WRF:3260 continue
                    return v
                v = lax.fori_loop(0,_p388,_p389,v)
                v['i'] = jnp.where((_p387 & (v['_go']==0)), jnp.int32((v['myis'])+_p388*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p384,_p385,v)
            v['kk'] = jnp.where((_p383 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p384*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3261 do i=myis,myie
            _p391 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p392 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p393(_p394,v):
                v = dict(v)
                v['i'] = jnp.where((_p391 & (v['_go']==0)), jnp.int32((v['myis'])+_p394*(jnp.int32(1))), v['i'])
                # WRF:3262 avephi(i,lp1)=avephi(i,lm1)+emx1(i)
                v['avephi'] = v['avephi'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p391 & (v['_go']==0))) & (v['_go']==0)), _r((v['avephi'][(v['i'])-1, (v['lm1'])-1] + v['emx1'][(v['i'])-1])), v['avephi'][(v['i'])-1, (v['lp1'])-1]))
                v['_go'] = jnp.where(v['_go']==1803, 0, v['_go'])
                # WRF:3263 continue
                return v
            v = lax.fori_loop(0,_p392,_p393,v)
            v['i'] = jnp.where((_p391 & (v['_go']==0)), jnp.int32((v['myis'])+_p392*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3270 call e290(emissb,emiss,avephi,klen,fxoe2,dte2, h16e1,hp1,h28e1,haf,ten, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte      )
            _p395 = lax.cond((((_p379 & (v['_go']==0))) & (v['_go']==0)), lambda _a: _e290(*_a), lambda _a: _a, (v['emissb'], v['emiss'], v['avephi'], v['klen'], v['fxoe2'], v['dte2'], v['h16e1'], v['hp1'], v['h28e1'], v['haf'], v['ten'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],))
            v['emissb'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[0]), v['emissb'])
            v['emiss'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[1]), v['emiss'])
            v['avephi'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[2]), v['avephi'])
            v['klen'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[3]), v['klen'])
            v['fxoe2'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[4]), v['fxoe2'])
            v['dte2'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[5]), v['dte2'])
            v['h16e1'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[6]), v['h16e1'])
            v['hp1'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[7]), v['hp1'])
            v['h28e1'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[8]), v['h28e1'])
            v['haf'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[9]), v['haf'])
            v['ten'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), _r(_p395[10]), v['ten'])
            v['ids'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[11]), v['ids'])
            v['ide'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[12]), v['ide'])
            v['jds'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[13]), v['jds'])
            v['jde'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[14]), v['jde'])
            v['kds'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[15]), v['kds'])
            v['kde'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[16]), v['kde'])
            v['ims'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[17]), v['ims'])
            v['ime'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[18]), v['ime'])
            v['jms'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[19]), v['jms'])
            v['jme'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[20]), v['jme'])
            v['kms'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[21]), v['kms'])
            v['kme'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[22]), v['kme'])
            v['its'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[23]), v['its'])
            v['ite'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[24]), v['ite'])
            v['jts'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[25]), v['jts'])
            v['jte'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[26]), v['jte'])
            v['kts'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[27]), v['kts'])
            v['kte'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_p395[28]), v['kte'])
            # WRF:3277 do kk=1,lp1-k
            _p396 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p397 = jnp.maximum(jnp.int32(0), (((v['lp1'] - v['k']))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p398(_p399,v):
                v = dict(v)
                v['kk'] = jnp.where((_p396 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p399*(jnp.int32(1))), v['kk'])
                # WRF:3278 do i=myis,myie
                _p400 = (((_p396 & (v['_go']==0))) & (v['_go']==0))
                _p401 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p402(_p403,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p400 & (v['_go']==0)), jnp.int32((v['myis'])+_p403*(jnp.int32(1))), v['i'])
                    # WRF:3279 avmo3(i,kk+k-1)=toto3(i,kk+k)-toto3(i,k)
                    v['avmo3'] = v['avmo3'].at[(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p400 & (v['_go']==0))) & (v['_go']==0)), _r((v['toto3'][(v['i'])-1, ((v['kk'] + v['k']))-1] - v['toto3'][(v['i'])-1, (v['k'])-1])), v['avmo3'][(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1]))
                    # WRF:3280 avpho3(i,kk+k-1)=tphio3(i,kk+k)-tphio3(i,k)
                    v['avpho3'] = v['avpho3'].at[(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p400 & (v['_go']==0))) & (v['_go']==0)), _r((v['tphio3'][(v['i'])-1, ((v['kk'] + v['k']))-1] - v['tphio3'][(v['i'])-1, (v['k'])-1])), v['avpho3'][(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1]))
                    # WRF:3281 avvo2(i,kk+k-1)=totvo2(i,kk+k)-totvo2(i,k)
                    v['avvo2'] = v['avvo2'].at[(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p400 & (v['_go']==0))) & (v['_go']==0)), _r((v['totvo2'][(v['i'])-1, ((v['kk'] + v['k']))-1] - v['totvo2'][(v['i'])-1, (v['k'])-1])), v['avvo2'][(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1]))
                    # WRF:3282 cont1d(i,kk+k-1)=cnttau(i,kk+k-1)*totevv(i,k-1)
                    v['cont1d'] = v['cont1d'].at[(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p400 & (v['_go']==0))) & (v['_go']==0)), _r((v['cnttau'][(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1] * v['totevv'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])), v['cont1d'][(v['i'])-1, (((v['kk'] + v['k']) - jnp.int32(1)))-1]))
                    v['_go'] = jnp.where(v['_go']==322, 0, v['_go'])
                    # WRF:3283 continue
                    return v
                v = lax.fori_loop(0,_p401,_p402,v)
                v['i'] = jnp.where((_p400 & (v['_go']==0)), jnp.int32((v['myis'])+_p401*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p397,_p398,v)
            v['kk'] = jnp.where((_p396 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p397*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3285 do kk=1,lp1-k
            _p404 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p405 = jnp.maximum(jnp.int32(0), (((v['lp1'] - v['k']))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p406(_p407,v):
                v = dict(v)
                v['kk'] = jnp.where((_p404 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p407*(jnp.int32(1))), v['kk'])
                # WRF:3286 do i=myis,myie
                _p408 = (((_p404 & (v['_go']==0))) & (v['_go']==0))
                _p409 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p410(_p411,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p408 & (v['_go']==0)), jnp.int32((v['myis'])+_p411*(jnp.int32(1))), v['i'])
                    # WRF:3287 fac1(i,k+kk-1)=bo3rnd(2)*avpho3(i,k+kk-1)/avmo3(i,k+kk-1)
                    v['fac1'] = v['fac1'].at[(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1].set(jnp.where((((_p408 & (v['_go']==0))) & (v['_go']==0)), _r(((v['bo3rnd'][(jnp.int32(2))-1] * v['avpho3'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]) / v['avmo3'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1])), v['fac1'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))
                    # WRF:3288 vtmp3(i,k+kk-1)=haf*(fac1(i,k+kk-1)* (sqrt(one+(four*ao3rnd(2)*avmo3(i,k+kk-1))/ fac1(i,k+kk-1))-one))
                    v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1].set(jnp.where((((_p408 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['fac1'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1] * (jnp.sqrt((v['one'] + (((v['four'] * v['ao3rnd'][(jnp.int32(2))-1]) * v['avmo3'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]) / v['fac1'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))) - v['one'])))), v['vtmp3'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))
                    # WRF:3291 to31d(i,k+kk-1)=exp(hm1ez*(vtmp3(i,k+kk-1) +sko3r*avvo2(i,k+kk-1)))
                    v['to31d'] = v['to31d'].at[(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1].set(jnp.where((((_p408 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (v['vtmp3'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1] + (_r(_TABLES['sko3r']) * v['avvo2'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))))), v['to31d'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))
                    # WRF:3293 over1d(i,k+kk-1)=exp(hm1ez*(sqrt(ab15wd*avephi(i,k+kk-1))+ skc1r*avvo2(i,k+kk-1)))
                    v['over1d'] = v['over1d'].at[(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1].set(jnp.where((((_p408 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (jnp.sqrt((_r(_TABLES['ab15wd']) * v['avephi'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1])) + (_r(_TABLES['skc1r']) * v['avvo2'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))))), v['over1d'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1]))
                    # WRF:3295 co21(i,k+kk,k)=over1d(i,k+kk-1)*co21(i,k+kk,k)
                    v['co21'] = v['co21'].at[(v['i'])-1, ((v['k'] + v['kk']))-1, (v['k'])-1].set(jnp.where((((_p408 & (v['_go']==0))) & (v['_go']==0)), _r((v['over1d'][(v['i'])-1, (((v['k'] + v['kk']) - jnp.int32(1)))-1] * v['co21'][(v['i'])-1, ((v['k'] + v['kk']))-1, (v['k'])-1])), v['co21'][(v['i'])-1, ((v['k'] + v['kk']))-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==3221, 0, v['_go'])
                    # WRF:3296 continue
                    return v
                v = lax.fori_loop(0,_p409,_p410,v)
                v['i'] = jnp.where((_p408 & (v['_go']==0)), jnp.int32((v['myis'])+_p409*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p405,_p406,v)
            v['kk'] = jnp.where((_p404 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p405*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3297 do kp=k+1,lp1
            _p412 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p413 = jnp.maximum(jnp.int32(0), ((v['lp1'])-((v['k'] + jnp.int32(1))))//(jnp.int32(1))+jnp.int32(1))
            def _p414(_p415,v):
                v = dict(v)
                v['kp'] = jnp.where((_p412 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p415*(jnp.int32(1))), v['kp'])
                # WRF:3298 do i=myis,myie
                _p416 = (((_p412 & (v['_go']==0))) & (v['_go']==0))
                _p417 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p418(_p419,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p416 & (v['_go']==0)), jnp.int32((v['myis'])+_p419*(jnp.int32(1))), v['i'])
                    # WRF:3299 co21(i,k,kp)=over1d(i,kp-1)*co21(i,k,kp)
                    v['co21'] = v['co21'].at[(v['i'])-1, (v['k'])-1, (v['kp'])-1].set(jnp.where((((_p416 & (v['_go']==0))) & (v['_go']==0)), _r((v['over1d'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1] * v['co21'][(v['i'])-1, (v['k'])-1, (v['kp'])-1])), v['co21'][(v['i'])-1, (v['k'])-1, (v['kp'])-1]))
                    v['_go'] = jnp.where(v['_go']==3223, 0, v['_go'])
                    # WRF:3300 continue
                    return v
                v = lax.fori_loop(0,_p417,_p418,v)
                v['i'] = jnp.where((_p416 & (v['_go']==0)), jnp.int32((v['myis'])+_p417*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p413,_p414,v)
            v['kp'] = jnp.where((_p412 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p413*(jnp.int32(1))), v['kp'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3302 do i=myis,myie
            _p420 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p421 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p422(_p423,v):
                v = dict(v)
                v['i'] = jnp.where((_p420 & (v['_go']==0)), jnp.int32((v['myis'])+_p423*(jnp.int32(1))), v['i'])
                # WRF:3303 rlog(i,k)=over1d(i,k)*co2nbl(i,k)
                v['rlog'] = v['rlog'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p420 & (v['_go']==0))) & (v['_go']==0)), _r((v['over1d'][(v['i'])-1, (v['k'])-1] * v['co2nbl'][(v['i'])-1, (v['k'])-1])), v['rlog'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1804, 0, v['_go'])
                # WRF:3304 continue
                return v
            v = lax.fori_loop(0,_p421,_p422,v)
            v['i'] = jnp.where((_p420 & (v['_go']==0)), jnp.int32((v['myis'])+_p421*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3306 do kp=k+1,lp1
            _p424 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p425 = jnp.maximum(jnp.int32(0), ((v['lp1'])-((v['k'] + jnp.int32(1))))//(jnp.int32(1))+jnp.int32(1))
            def _p426(_p427,v):
                v = dict(v)
                v['kp'] = jnp.where((_p424 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p427*(jnp.int32(1))), v['kp'])
                # WRF:3307 do i=myis,myie
                _p428 = (((_p424 & (v['_go']==0))) & (v['_go']==0))
                _p429 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p430(_p431,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p428 & (v['_go']==0)), jnp.int32((v['myis'])+_p431*(jnp.int32(1))), v['i'])
                    # WRF:3308 flx(i,k)=flx(i,k)+(oss(i,kp)*to31d(i,kp-1) +ss2(i,kp)*cont1d(i,kp-1) +css(i,kp)*co21(i,kp,k) +dtc(i,kp)*emiss(i,kp-1))*cldfac(i,kp,k)
                    v['flx'] = v['flx'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p428 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx'][(v['i'])-1, (v['k'])-1] + (((((v['oss'][(v['i'])-1, (v['kp'])-1] * v['to31d'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1]) + (v['ss2'][(v['i'])-1, (v['kp'])-1] * v['cont1d'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1])) + (v['css'][(v['i'])-1, (v['kp'])-1] * v['co21'][(v['i'])-1, (v['kp'])-1, (v['k'])-1])) + (v['dtc'][(v['i'])-1, (v['kp'])-1] * v['emiss'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1])) * v['cldfac'][(v['i'])-1, (v['kp'])-1, (v['k'])-1]))), v['flx'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==3423, 0, v['_go'])
                    # WRF:3312 continue
                    return v
                v = lax.fori_loop(0,_p429,_p430,v)
                v['i'] = jnp.where((_p428 & (v['_go']==0)), jnp.int32((v['myis'])+_p429*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p425,_p426,v)
            v['kp'] = jnp.where((_p424 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p425*(jnp.int32(1))), v['kp'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3313 do kp=k+1,lp1
            _p432 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p433 = jnp.maximum(jnp.int32(0), ((v['lp1'])-((v['k'] + jnp.int32(1))))//(jnp.int32(1))+jnp.int32(1))
            def _p434(_p435,v):
                v = dict(v)
                v['kp'] = jnp.where((_p432 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p435*(jnp.int32(1))), v['kp'])
                # WRF:3314 do i=myis,myie
                _p436 = (((_p432 & (v['_go']==0))) & (v['_go']==0))
                _p437 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p438(_p439,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p436 & (v['_go']==0)), jnp.int32((v['myis'])+_p439*(jnp.int32(1))), v['i'])
                    # WRF:3315 flx(i,kp)=flx(i,kp)+(oss(i,k)*to31d(i,kp-1) +ss2(i,k)*cont1d(i,kp-1) +css(i,k)*co21(i,k,kp) +dtc(i,k)*emissb(i,kp-1))*cldfac(i,k,kp)
                    v['flx'] = v['flx'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p436 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx'][(v['i'])-1, (v['kp'])-1] + (((((v['oss'][(v['i'])-1, (v['k'])-1] * v['to31d'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1]) + (v['ss2'][(v['i'])-1, (v['k'])-1] * v['cont1d'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1])) + (v['css'][(v['i'])-1, (v['k'])-1] * v['co21'][(v['i'])-1, (v['k'])-1, (v['kp'])-1])) + (v['dtc'][(v['i'])-1, (v['k'])-1] * v['emissb'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1])) * v['cldfac'][(v['i'])-1, (v['k'])-1, (v['kp'])-1]))), v['flx'][(v['i'])-1, (v['kp'])-1]))
                    v['_go'] = jnp.where(v['_go']==3425, 0, v['_go'])
                    # WRF:3319 continue
                    return v
                v = lax.fori_loop(0,_p437,_p438,v)
                v['i'] = jnp.where((_p436 & (v['_go']==0)), jnp.int32((v['myis'])+_p437*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p433,_p434,v)
            v['kp'] = jnp.where((_p432 & (v['_go']==0)), jnp.int32(((v['k'] + jnp.int32(1)))+_p433*(jnp.int32(1))), v['kp'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==321, 0, v['_go'])
            # WRF:3320 continue
            return v
        v = lax.fori_loop(0,_p380,_p381,v)
        v['k'] = jnp.where((_p379 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p380*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3322 do i=myis,myie
        _p440 = ((jnp.bool_(True)) & (v['_go']==0))
        _p441 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p442(_p443,v):
            v = dict(v)
            v['i'] = jnp.where((_p440 & (v['_go']==0)), jnp.int32((v['myis'])+_p443*(jnp.int32(1))), v['i'])
            # WRF:3323 tpl(i,1)=temp(i,l)
            v['tpl'] = v['tpl'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p440 & (v['_go']==0))) & (v['_go']==0)), _r(v['temp'][(v['i'])-1, (v['l'])-1]), v['tpl'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3324 tpl(i,lp1)=haf*(t(i,lp1)+temp(i,l))
            v['tpl'] = v['tpl'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p440 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['t'][(v['i'])-1, (v['lp1'])-1] + v['temp'][(v['i'])-1, (v['l'])-1]))), v['tpl'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:3325 tpl(i,llp1)=haf*(t(i,l)+temp(i,l))
            v['tpl'] = v['tpl'].at[(v['i'])-1, (v['llp1'])-1].set(jnp.where((((_p440 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['t'][(v['i'])-1, (v['l'])-1] + v['temp'][(v['i'])-1, (v['l'])-1]))), v['tpl'][(v['i'])-1, (v['llp1'])-1]))
            v['_go'] = jnp.where(v['_go']==821, 0, v['_go'])
            # WRF:3326 continue
            return v
        v = lax.fori_loop(0,_p441,_p442,v)
        v['i'] = jnp.where((_p440 & (v['_go']==0)), jnp.int32((v['myis'])+_p441*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3327 do k=2,l
        _p444 = ((jnp.bool_(True)) & (v['_go']==0))
        _p445 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p446(_p447,v):
            v = dict(v)
            v['k'] = jnp.where((_p444 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p447*(jnp.int32(1))), v['k'])
            # WRF:3328 do i=myis,myie
            _p448 = (((_p444 & (v['_go']==0))) & (v['_go']==0))
            _p449 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p450(_p451,v):
                v = dict(v)
                v['i'] = jnp.where((_p448 & (v['_go']==0)), jnp.int32((v['myis'])+_p451*(jnp.int32(1))), v['i'])
                # WRF:3329 tpl(i,k)=t(i,k)
                v['tpl'] = v['tpl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p448 & (v['_go']==0))) & (v['_go']==0)), _r(v['t'][(v['i'])-1, (v['k'])-1]), v['tpl'][(v['i'])-1, (v['k'])-1]))
                # WRF:3330 tpl(i,k+l)=t(i,k)
                v['tpl'] = v['tpl'].at[(v['i'])-1, ((v['k'] + v['l']))-1].set(jnp.where((((_p448 & (v['_go']==0))) & (v['_go']==0)), _r(v['t'][(v['i'])-1, (v['k'])-1]), v['tpl'][(v['i'])-1, ((v['k'] + v['l']))-1]))
                v['_go'] = jnp.where(v['_go']==823, 0, v['_go'])
                # WRF:3331 continue
                return v
            v = lax.fori_loop(0,_p449,_p450,v)
            v['i'] = jnp.where((_p448 & (v['_go']==0)), jnp.int32((v['myis'])+_p449*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p445,_p446,v)
        v['k'] = jnp.where((_p444 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p445*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3335 do i=myis,myie
        _p452 = ((jnp.bool_(True)) & (v['_go']==0))
        _p453 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p454(_p455,v):
            v = dict(v)
            v['i'] = jnp.where((_p452 & (v['_go']==0)), jnp.int32((v['myis'])+_p455*(jnp.int32(1))), v['i'])
            # WRF:3336 avephi(i,1)=var2(i,l)
            v['avephi'] = v['avephi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p452 & (v['_go']==0))) & (v['_go']==0)), _r(v['var2'][(v['i'])-1, (v['l'])-1]), v['avephi'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3337 avephi(i,2)=var2(i,l)+empl(i,l)
            v['avephi'] = v['avephi'].at[(v['i'])-1, (jnp.int32(2))-1].set(jnp.where((((_p452 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['l'])-1] + v['empl'][(v['i'])-1, (v['l'])-1])), v['avephi'][(v['i'])-1, (jnp.int32(2))-1]))
            v['_go'] = jnp.where(v['_go']==833, 0, v['_go'])
            # WRF:3338 continue
            return v
        v = lax.fori_loop(0,_p453,_p454,v)
        v['i'] = jnp.where((_p452 & (v['_go']==0)), jnp.int32((v['myis'])+_p453*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3339 call e2spec(emiss,avephi,fxosp,dtsp, h16e1,ten,h28e1,hp1, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte                     )
        _p456 = lax.cond(((jnp.bool_(True)) & (v['_go']==0)), lambda _a: _e2spec(*_a), lambda _a: _a, (v['emiss'], v['avephi'], v['fxosp'], v['dtsp'], v['h16e1'], v['ten'], v['h28e1'], v['hp1'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],))
        v['emiss'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[0]), v['emiss'])
        v['avephi'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[1]), v['avephi'])
        v['fxosp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[2]), v['fxosp'])
        v['dtsp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[3]), v['dtsp'])
        v['h16e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[4]), v['h16e1'])
        v['ten'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[5]), v['ten'])
        v['h28e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[6]), v['h28e1'])
        v['hp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p456[7]), v['hp1'])
        v['ids'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[8]), v['ids'])
        v['ide'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[9]), v['ide'])
        v['jds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[10]), v['jds'])
        v['jde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[11]), v['jde'])
        v['kds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[12]), v['kds'])
        v['kde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[13]), v['kde'])
        v['ims'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[14]), v['ims'])
        v['ime'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[15]), v['ime'])
        v['jms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[16]), v['jms'])
        v['jme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[17]), v['jme'])
        v['kms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[18]), v['kms'])
        v['kme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[19]), v['kme'])
        v['its'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[20]), v['its'])
        v['ite'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[21]), v['ite'])
        v['jts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[22]), v['jts'])
        v['jte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[23]), v['jte'])
        v['kts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[24]), v['kts'])
        v['kte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p456[25]), v['kte'])
        # WRF:3349 call e3v88(emd,tpl,empl, ten,hp1,h28e1,h16e1, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte                     )
        _p457 = lax.cond(((jnp.bool_(True)) & (v['_go']==0)), lambda _a: _e3v88(*_a), lambda _a: _a, (v['emd'], v['tpl'], v['empl'], v['ten'], v['hp1'], v['h28e1'], v['h16e1'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],))
        v['emd'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[0]), v['emd'])
        v['tpl'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[1]), v['tpl'])
        v['empl'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[2]), v['empl'])
        v['ten'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[3]), v['ten'])
        v['hp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[4]), v['hp1'])
        v['h28e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[5]), v['h28e1'])
        v['h16e1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(_p457[6]), v['h16e1'])
        v['ids'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[7]), v['ids'])
        v['ide'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[8]), v['ide'])
        v['jds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[9]), v['jds'])
        v['jde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[10]), v['jde'])
        v['kds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[11]), v['kds'])
        v['kde'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[12]), v['kde'])
        v['ims'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[13]), v['ims'])
        v['ime'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[14]), v['ime'])
        v['jms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[15]), v['jms'])
        v['jme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[16]), v['jme'])
        v['kms'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[17]), v['kms'])
        v['kme'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[18]), v['kme'])
        v['its'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[19]), v['its'])
        v['ite'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[20]), v['ite'])
        v['jts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[21]), v['jts'])
        v['jte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[22]), v['jte'])
        v['kts'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[23]), v['kts'])
        v['kte'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(_p457[24]), v['kte'])
        # WRF:3357 do k=2,l
        _p458 = ((jnp.bool_(True)) & (v['_go']==0))
        _p459 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p460(_p461,v):
            v = dict(v)
            v['k'] = jnp.where((_p458 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p461*(jnp.int32(1))), v['k'])
            # WRF:3358 do i=myis,myie
            _p462 = (((_p458 & (v['_go']==0))) & (v['_go']==0))
            _p463 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p464(_p465,v):
                v = dict(v)
                v['i'] = jnp.where((_p462 & (v['_go']==0)), jnp.int32((v['myis'])+_p465*(jnp.int32(1))), v['i'])
                # WRF:3359 emisdg(i,k)=emd(i,k+l)+emd(i,k)
                v['emisdg'] = v['emisdg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p462 & (v['_go']==0))) & (v['_go']==0)), _r((v['emd'][(v['i'])-1, ((v['k'] + v['l']))-1] + v['emd'][(v['i'])-1, (v['k'])-1])), v['emisdg'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==851, 0, v['_go'])
                # WRF:3360 continue
                return v
            v = lax.fori_loop(0,_p463,_p464,v)
            v['i'] = jnp.where((_p462 & (v['_go']==0)), jnp.int32((v['myis'])+_p463*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p459,_p460,v)
        v['k'] = jnp.where((_p458 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p459*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3364 do i=myis,myie
        _p466 = ((jnp.bool_(True)) & (v['_go']==0))
        _p467 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p468(_p469,v):
            v = dict(v)
            v['i'] = jnp.where((_p466 & (v['_go']==0)), jnp.int32((v['myis'])+_p469*(jnp.int32(1))), v['i'])
            # WRF:3365 emspec(i,1)=(emd(i,1)*empl(i,1)-emd(i,lp1)*empl(i,lp1))/ emx1(i) + quartr*(emiss(i,1)+emiss(i,2))
            v['emspec'] = v['emspec'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p466 & (v['_go']==0))) & (v['_go']==0)), _r(((((v['emd'][(v['i'])-1, (jnp.int32(1))-1] * v['empl'][(v['i'])-1, (jnp.int32(1))-1]) - (v['emd'][(v['i'])-1, (v['lp1'])-1] * v['empl'][(v['i'])-1, (v['lp1'])-1])) / v['emx1'][(v['i'])-1]) + (v['quartr'] * (v['emiss'][(v['i'])-1, (jnp.int32(1))-1] + v['emiss'][(v['i'])-1, (jnp.int32(2))-1])))), v['emspec'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3367 emisdg(i,lp1)=two*emd(i,lp1)
            v['emisdg'] = v['emisdg'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p466 & (v['_go']==0))) & (v['_go']==0)), _r((v['two'] * v['emd'][(v['i'])-1, (v['lp1'])-1])), v['emisdg'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:3368 emspec(i,2)=two*(emd(i,1)*empl(i,1)-emd(i,llp1)*empl(i,llp1))/ emx2(i)
            v['emspec'] = v['emspec'].at[(v['i'])-1, (jnp.int32(2))-1].set(jnp.where((((_p466 & (v['_go']==0))) & (v['_go']==0)), _r(((v['two'] * ((v['emd'][(v['i'])-1, (jnp.int32(1))-1] * v['empl'][(v['i'])-1, (jnp.int32(1))-1]) - (v['emd'][(v['i'])-1, (v['llp1'])-1] * v['empl'][(v['i'])-1, (v['llp1'])-1]))) / v['emx2'][(v['i'])-1])), v['emspec'][(v['i'])-1, (jnp.int32(2))-1]))
            v['_go'] = jnp.where(v['_go']==861, 0, v['_go'])
            # WRF:3370 continue
            return v
        v = lax.fori_loop(0,_p467,_p468,v)
        v['i'] = jnp.where((_p466 & (v['_go']==0)), jnp.int32((v['myis'])+_p467*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3371 do i=myis,myie
        _p470 = ((jnp.bool_(True)) & (v['_go']==0))
        _p471 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p472(_p473,v):
            v = dict(v)
            v['i'] = jnp.where((_p470 & (v['_go']==0)), jnp.int32((v['myis'])+_p473*(jnp.int32(1))), v['i'])
            # WRF:3372 fac1(i,l)=bo3rnd(2)*var4(i,l)/var3(i,l)
            v['fac1'] = v['fac1'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p470 & (v['_go']==0))) & (v['_go']==0)), _r(((v['bo3rnd'][(jnp.int32(2))-1] * v['var4'][(v['i'])-1, (v['l'])-1]) / v['var3'][(v['i'])-1, (v['l'])-1])), v['fac1'][(v['i'])-1, (v['l'])-1]))
            # WRF:3373 vtmp3(i,l)=haf*(fac1(i,l)* (sqrt(one+(four*ao3rnd(2)*var3(i,l))/fac1(i,l))-one))
            v['vtmp3'] = v['vtmp3'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p470 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['fac1'][(v['i'])-1, (v['l'])-1] * (jnp.sqrt((v['one'] + (((v['four'] * v['ao3rnd'][(jnp.int32(2))-1]) * v['var3'][(v['i'])-1, (v['l'])-1]) / v['fac1'][(v['i'])-1, (v['l'])-1]))) - v['one'])))), v['vtmp3'][(v['i'])-1, (v['l'])-1]))
            # WRF:3375 to31d(i,l)=exp(hm1ez*(vtmp3(i,l)+sko3r*cntval(i,l)))
            v['to31d'] = v['to31d'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p470 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (v['vtmp3'][(v['i'])-1, (v['l'])-1] + (_r(_TABLES['sko3r']) * v['cntval'][(v['i'])-1, (v['l'])-1]))))), v['to31d'][(v['i'])-1, (v['l'])-1]))
            # WRF:3376 over1d(i,l)=exp(hm1ez*(sqrt(ab15wd*var2(i,l))+ skc1r*cntval(i,l)))
            v['over1d'] = v['over1d'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p470 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (jnp.sqrt((_r(_TABLES['ab15wd']) * v['var2'][(v['i'])-1, (v['l'])-1])) + (_r(_TABLES['skc1r']) * v['cntval'][(v['i'])-1, (v['l'])-1]))))), v['over1d'][(v['i'])-1, (v['l'])-1]))
            # WRF:3378 cont1d(i,l)=cnttau(i,l)*totevv(i,lm1)
            v['cont1d'] = v['cont1d'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p470 & (v['_go']==0))) & (v['_go']==0)), _r((v['cnttau'][(v['i'])-1, (v['l'])-1] * v['totevv'][(v['i'])-1, (v['lm1'])-1])), v['cont1d'][(v['i'])-1, (v['l'])-1]))
            # WRF:3379 rlog(i,l)=over1d(i,l)*co2nbl(i,l)
            v['rlog'] = v['rlog'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p470 & (v['_go']==0))) & (v['_go']==0)), _r((v['over1d'][(v['i'])-1, (v['l'])-1] * v['co2nbl'][(v['i'])-1, (v['l'])-1])), v['rlog'][(v['i'])-1, (v['l'])-1]))
            v['_go'] = jnp.where(v['_go']==331, 0, v['_go'])
            # WRF:3380 continue
            return v
        v = lax.fori_loop(0,_p471,_p472,v)
        v['i'] = jnp.where((_p470 & (v['_go']==0)), jnp.int32((v['myis'])+_p471*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3381 do k=1,l
        _p474 = ((jnp.bool_(True)) & (v['_go']==0))
        _p475 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p476(_p477,v):
            v = dict(v)
            v['k'] = jnp.where((_p474 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p477*(jnp.int32(1))), v['k'])
            # WRF:3382 do i=myis,myie
            _p478 = (((_p474 & (v['_go']==0))) & (v['_go']==0))
            _p479 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p480(_p481,v):
                v = dict(v)
                v['i'] = jnp.where((_p478 & (v['_go']==0)), jnp.int32((v['myis'])+_p481*(jnp.int32(1))), v['i'])
                # WRF:3383 rlog(i,k)=log(rlog(i,k))
                v['rlog'] = v['rlog'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p478 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.log(v['rlog'][(v['i'])-1, (v['k'])-1])), v['rlog'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==618, 0, v['_go'])
                # WRF:3384 continue
                return v
            v = lax.fori_loop(0,_p479,_p480,v)
            v['i'] = jnp.where((_p478 & (v['_go']==0)), jnp.int32((v['myis'])+_p479*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p475,_p476,v)
        v['k'] = jnp.where((_p474 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p475*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3385 do k=1,lm1
        _p482 = ((jnp.bool_(True)) & (v['_go']==0))
        _p483 = jnp.maximum(jnp.int32(0), ((v['lm1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p484(_p485,v):
            v = dict(v)
            v['k'] = jnp.where((_p482 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p485*(jnp.int32(1))), v['k'])
            # WRF:3386 do i=myis,myie
            _p486 = (((_p482 & (v['_go']==0))) & (v['_go']==0))
            _p487 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p488(_p489,v):
                v = dict(v)
                v['i'] = jnp.where((_p486 & (v['_go']==0)), jnp.int32((v['myis'])+_p489*(jnp.int32(1))), v['i'])
                # WRF:3387 delpr1(i,k+1)=delp(i,k+1)*(press(i,k+1)-p(i,k+1))
                v['delpr1'] = v['delpr1'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p486 & (v['_go']==0))) & (v['_go']==0)), _r((v['delp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['press'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))), v['delpr1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3388 alp(i,lp1+k-1)=-sqrt(delpr1(i,k+1))*rlog(i,k+1)
                v['alp'] = v['alp'].at[(v['i'])-1, (((v['lp1'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p486 & (v['_go']==0))) & (v['_go']==0)), _r(((-jnp.sqrt(v['delpr1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])) * v['rlog'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['alp'][(v['i'])-1, (((v['lp1'] + v['k']) - jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==601, 0, v['_go'])
                # WRF:3389 continue
                return v
            v = lax.fori_loop(0,_p487,_p488,v)
            v['i'] = jnp.where((_p486 & (v['_go']==0)), jnp.int32((v['myis'])+_p487*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p483,_p484,v)
        v['k'] = jnp.where((_p482 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p483*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3390 do k=1,l
        _p490 = ((jnp.bool_(True)) & (v['_go']==0))
        _p491 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p492(_p493,v):
            v = dict(v)
            v['k'] = jnp.where((_p490 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p493*(jnp.int32(1))), v['k'])
            # WRF:3391 do i=myis,myie
            _p494 = (((_p490 & (v['_go']==0))) & (v['_go']==0))
            _p495 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p496(_p497,v):
                v = dict(v)
                v['i'] = jnp.where((_p494 & (v['_go']==0)), jnp.int32((v['myis'])+_p497*(jnp.int32(1))), v['i'])
                # WRF:3392 delpr2(i,k+1)=delp(i,k)*(p(i,k+1)-press(i,k))
                v['delpr2'] = v['delpr2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p494 & (v['_go']==0))) & (v['_go']==0)), _r((v['delp'][(v['i'])-1, (v['k'])-1] * (v['p'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['press'][(v['i'])-1, (v['k'])-1]))), v['delpr2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3393 alp(i,k)=-sqrt(delpr2(i,k+1))*rlog(i,k)
                v['alp'] = v['alp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p494 & (v['_go']==0))) & (v['_go']==0)), _r(((-jnp.sqrt(v['delpr2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])) * v['rlog'][(v['i'])-1, (v['k'])-1])), v['alp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==603, 0, v['_go'])
                # WRF:3394 continue
                return v
            v = lax.fori_loop(0,_p495,_p496,v)
            v['i'] = jnp.where((_p494 & (v['_go']==0)), jnp.int32((v['myis'])+_p495*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p491,_p492,v)
        v['k'] = jnp.where((_p490 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p491*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3395 do i=myis,myie
        _p498 = ((jnp.bool_(True)) & (v['_go']==0))
        _p499 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p500(_p501,v):
            v = dict(v)
            v['i'] = jnp.where((_p498 & (v['_go']==0)), jnp.int32((v['myis'])+_p501*(jnp.int32(1))), v['i'])
            # WRF:3396 alp(i,ll)=-rlog(i,l)
            v['alp'] = v['alp'].at[(v['i'])-1, (v['ll'])-1].set(jnp.where((((_p498 & (v['_go']==0))) & (v['_go']==0)), _r((-v['rlog'][(v['i'])-1, (v['l'])-1])), v['alp'][(v['i'])-1, (v['ll'])-1]))
            # WRF:3397 alp(i,llp1)=-rlog(i,l)*sqrt(delp(i,l)*(p(i,lp1)-press(i,lm1)))
            v['alp'] = v['alp'].at[(v['i'])-1, (v['llp1'])-1].set(jnp.where((((_p498 & (v['_go']==0))) & (v['_go']==0)), _r(((-v['rlog'][(v['i'])-1, (v['l'])-1]) * jnp.sqrt((v['delp'][(v['i'])-1, (v['l'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['lm1'])-1]))))), v['alp'][(v['i'])-1, (v['llp1'])-1]))
            v['_go'] = jnp.where(v['_go']==625, 0, v['_go'])
            # WRF:3398 continue
            return v
        v = lax.fori_loop(0,_p499,_p500,v)
        v['i'] = jnp.where((_p498 & (v['_go']==0)), jnp.int32((v['myis'])+_p499*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3405 do k=1,llp1
        _p502 = ((jnp.bool_(True)) & (v['_go']==0))
        _p503 = jnp.maximum(jnp.int32(0), ((v['llp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p504(_p505,v):
            v = dict(v)
            v['k'] = jnp.where((_p502 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p505*(jnp.int32(1))), v['k'])
            # WRF:3406 do i=myis,myie
            _p506 = (((_p502 & (v['_go']==0))) & (v['_go']==0))
            _p507 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p508(_p509,v):
                v = dict(v)
                v['i'] = jnp.where((_p506 & (v['_go']==0)), jnp.int32((v['myis'])+_p509*(jnp.int32(1))), v['i'])
                # WRF:3407 c(i,k)=alp(i,k)*(hmp66667+alp(i,k)*(quartr+alp(i,k)*hm6666m2))
                v['c'] = v['c'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p506 & (v['_go']==0))) & (v['_go']==0)), _r((v['alp'][(v['i'])-1, (v['k'])-1] * (v['hmp66667'] + (v['alp'][(v['i'])-1, (v['k'])-1] * (v['quartr'] + (v['alp'][(v['i'])-1, (v['k'])-1] * v['hm6666m2'])))))), v['c'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==631, 0, v['_go'])
                # WRF:3408 continue
                return v
            v = lax.fori_loop(0,_p507,_p508,v)
            v['i'] = jnp.where((_p506 & (v['_go']==0)), jnp.int32((v['myis'])+_p507*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p503,_p504,v)
        v['k'] = jnp.where((_p502 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p503*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3409 do i=myis,myie
        _p510 = ((jnp.bool_(True)) & (v['_go']==0))
        _p511 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p512(_p513,v):
            v = dict(v)
            v['i'] = jnp.where((_p510 & (v['_go']==0)), jnp.int32((v['myis'])+_p513*(jnp.int32(1))), v['i'])
            # WRF:3410 co21(i,lp1,lp1)=one+c(i,l)
            v['co21'] = v['co21'].at[(v['i'])-1, (v['lp1'])-1, (v['lp1'])-1].set(jnp.where((((_p510 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] + v['c'][(v['i'])-1, (v['l'])-1])), v['co21'][(v['i'])-1, (v['lp1'])-1, (v['lp1'])-1]))
            # WRF:3411 co21(i,lp1,l)=one+(delp2(i,l)*c(i,ll)-(press(i,l)-p(i,l))* c(i,llm1))/(p(i,lp1)-press(i,l))
            v['co21'] = v['co21'].at[(v['i'])-1, (v['lp1'])-1, (v['l'])-1].set(jnp.where((((_p510 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] + (((v['delp2'][(v['i'])-1, (v['l'])-1] * v['c'][(v['i'])-1, (v['ll'])-1]) - ((v['press'][(v['i'])-1, (v['l'])-1] - v['p'][(v['i'])-1, (v['l'])-1]) * v['c'][(v['i'])-1, (v['llm1'])-1])) / (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])))), v['co21'][(v['i'])-1, (v['lp1'])-1, (v['l'])-1]))
            # WRF:3413 co21(i,l,lp1)=one+((p(i,lp1)-press(i,lm1))*c(i,llp1)- (p(i,lp1)-press(i,l))*c(i,l))/(press(i,l)-press(i,lm1))
            v['co21'] = v['co21'].at[(v['i'])-1, (v['l'])-1, (v['lp1'])-1].set(jnp.where((((_p510 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] + ((((v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['lm1'])-1]) * v['c'][(v['i'])-1, (v['llp1'])-1]) - ((v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1]) * v['c'][(v['i'])-1, (v['l'])-1])) / (v['press'][(v['i'])-1, (v['l'])-1] - v['press'][(v['i'])-1, (v['lm1'])-1])))), v['co21'][(v['i'])-1, (v['l'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==641, 0, v['_go'])
            # WRF:3415 continue
            return v
        v = lax.fori_loop(0,_p511,_p512,v)
        v['i'] = jnp.where((_p510 & (v['_go']==0)), jnp.int32((v['myis'])+_p511*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3416 do k=2,l
        _p514 = ((jnp.bool_(True)) & (v['_go']==0))
        _p515 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p516(_p517,v):
            v = dict(v)
            v['k'] = jnp.where((_p514 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p517*(jnp.int32(1))), v['k'])
            # WRF:3417 do i=myis,myie
            _p518 = (((_p514 & (v['_go']==0))) & (v['_go']==0))
            _p519 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p520(_p521,v):
                v = dict(v)
                v['i'] = jnp.where((_p518 & (v['_go']==0)), jnp.int32((v['myis'])+_p521*(jnp.int32(1))), v['i'])
                # WRF:3418 co21(i,k,k)=one+haf*(c(i,lm1+k)+c(i,k-1))
                v['co21'] = v['co21'].at[(v['i'])-1, (v['k'])-1, (v['k'])-1].set(jnp.where((((_p518 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] + (v['haf'] * (v['c'][(v['i'])-1, ((v['lm1'] + v['k']))-1] + v['c'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])))), v['co21'][(v['i'])-1, (v['k'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==643, 0, v['_go'])
                # WRF:3419 continue
                return v
            v = lax.fori_loop(0,_p519,_p520,v)
            v['i'] = jnp.where((_p518 & (v['_go']==0)), jnp.int32((v['myis'])+_p519*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p515,_p516,v)
        v['k'] = jnp.where((_p514 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p515*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3424 do k=1,lm1
        _p522 = ((jnp.bool_(True)) & (v['_go']==0))
        _p523 = jnp.maximum(jnp.int32(0), ((v['lm1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p524(_p525,v):
            v = dict(v)
            v['k'] = jnp.where((_p522 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p525*(jnp.int32(1))), v['k'])
            # WRF:3425 do i=myis,myie
            _p526 = (((_p522 & (v['_go']==0))) & (v['_go']==0))
            _p527 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p528(_p529,v):
                v = dict(v)
                v['i'] = jnp.where((_p526 & (v['_go']==0)), jnp.int32((v['myis'])+_p529*(jnp.int32(1))), v['i'])
                # WRF:3426 csub(i,k+1)=cntval(i,k+1)*delpr1(i,k+1)
                v['csub'] = v['csub'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p526 & (v['_go']==0))) & (v['_go']==0)), _r((v['cntval'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['delpr1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['csub'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3427 csub(i,lp1+k-1)=cntval(i,k)*delpr2(i,k+1)
                v['csub'] = v['csub'].at[(v['i'])-1, (((v['lp1'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p526 & (v['_go']==0))) & (v['_go']==0)), _r((v['cntval'][(v['i'])-1, (v['k'])-1] * v['delpr2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['csub'][(v['i'])-1, (((v['lp1'] + v['k']) - jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==651, 0, v['_go'])
                # WRF:3428 continue
                return v
            v = lax.fori_loop(0,_p527,_p528,v)
            v['i'] = jnp.where((_p526 & (v['_go']==0)), jnp.int32((v['myis'])+_p527*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p523,_p524,v)
        v['k'] = jnp.where((_p522 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p523*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3430 do k=1,llm2
        _p530 = ((jnp.bool_(True)) & (v['_go']==0))
        _p531 = jnp.maximum(jnp.int32(0), ((v['llm2'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p532(_p533,v):
            v = dict(v)
            v['k'] = jnp.where((_p530 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p533*(jnp.int32(1))), v['k'])
            # WRF:3431 do i=myis,myie
            _p534 = (((_p530 & (v['_go']==0))) & (v['_go']==0))
            _p535 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p536(_p537,v):
                v = dict(v)
                v['i'] = jnp.where((_p534 & (v['_go']==0)), jnp.int32((v['myis'])+_p537*(jnp.int32(1))), v['i'])
                # WRF:3432 csub2(i,k+1)=sko3r*csub(i,k+1)
                v['csub2'] = v['csub2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p534 & (v['_go']==0))) & (v['_go']==0)), _r((_r(_TABLES['sko3r']) * v['csub'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['csub2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3433 c(i,k+1)=csub(i,k+1)*(hmp5+csub(i,k+1)* (hp166666-csub(i,k+1)*h41666m2))
                v['c'] = v['c'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p534 & (v['_go']==0))) & (v['_go']==0)), _r((v['csub'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['hmp5'] + (v['csub'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['hp166666'] - (v['csub'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['h41666m2'])))))), v['c'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3435 c2(i,k+1)=csub2(i,k+1)*(hmp5+csub2(i,k+1)* (hp166666-csub2(i,k+1)*h41666m2))
                v['c2'] = v['c2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p534 & (v['_go']==0))) & (v['_go']==0)), _r((v['csub2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['hmp5'] + (v['csub2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['hp166666'] - (v['csub2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['h41666m2'])))))), v['c2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==655, 0, v['_go'])
                # WRF:3437 continue
                return v
            v = lax.fori_loop(0,_p535,_p536,v)
            v['i'] = jnp.where((_p534 & (v['_go']==0)), jnp.int32((v['myis'])+_p535*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p531,_p532,v)
        v['k'] = jnp.where((_p530 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p531*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3438 do i=myis,myie
        _p538 = ((jnp.bool_(True)) & (v['_go']==0))
        _p539 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p540(_p541,v):
            v = dict(v)
            v['i'] = jnp.where((_p538 & (v['_go']==0)), jnp.int32((v['myis'])+_p541*(jnp.int32(1))), v['i'])
            # WRF:3439 contdg(i,lp1)=1.+c(i,llm1)
            v['contdg'] = v['contdg'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p538 & (v['_go']==0))) & (v['_go']==0)), _r((_r(1.) + v['c'][(v['i'])-1, (v['llm1'])-1])), v['contdg'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:3440 to3dg(i,lp1)=1.+c2(i,llm1)
            v['to3dg'] = v['to3dg'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p538 & (v['_go']==0))) & (v['_go']==0)), _r((_r(1.) + v['c2'][(v['i'])-1, (v['llm1'])-1])), v['to3dg'][(v['i'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==661, 0, v['_go'])
            # WRF:3441 continue
            return v
        v = lax.fori_loop(0,_p539,_p540,v)
        v['i'] = jnp.where((_p538 & (v['_go']==0)), jnp.int32((v['myis'])+_p539*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3442 do k=2,l
        _p542 = ((jnp.bool_(True)) & (v['_go']==0))
        _p543 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p544(_p545,v):
            v = dict(v)
            v['k'] = jnp.where((_p542 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p545*(jnp.int32(1))), v['k'])
            # WRF:3443 do i=myis,myie
            _p546 = (((_p542 & (v['_go']==0))) & (v['_go']==0))
            _p547 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p548(_p549,v):
                v = dict(v)
                v['i'] = jnp.where((_p546 & (v['_go']==0)), jnp.int32((v['myis'])+_p549*(jnp.int32(1))), v['i'])
                # WRF:3444 contdg(i,k)=one+haf*(c(i,k)+c(i,lm1+k))
                v['contdg'] = v['contdg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p546 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] + (v['haf'] * (v['c'][(v['i'])-1, (v['k'])-1] + v['c'][(v['i'])-1, ((v['lm1'] + v['k']))-1])))), v['contdg'][(v['i'])-1, (v['k'])-1]))
                # WRF:3445 to3dg(i,k)=one+haf*(c2(i,k)+c2(i,lm1+k))
                v['to3dg'] = v['to3dg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p546 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] + (v['haf'] * (v['c2'][(v['i'])-1, (v['k'])-1] + v['c2'][(v['i'])-1, ((v['lm1'] + v['k']))-1])))), v['to3dg'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==663, 0, v['_go'])
                # WRF:3446 continue
                return v
            v = lax.fori_loop(0,_p547,_p548,v)
            v['i'] = jnp.where((_p546 & (v['_go']==0)), jnp.int32((v['myis'])+_p547*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p543,_p544,v)
        v['k'] = jnp.where((_p542 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p543*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3450 do k=2,lp1
        _p550 = ((jnp.bool_(True)) & (v['_go']==0))
        _p551 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p552(_p553,v):
            v = dict(v)
            v['k'] = jnp.where((_p550 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p553*(jnp.int32(1))), v['k'])
            # WRF:3451 do i=myis,myie
            _p554 = (((_p550 & (v['_go']==0))) & (v['_go']==0))
            _p555 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p556(_p557,v):
                v = dict(v)
                v['i'] = jnp.where((_p554 & (v['_go']==0)), jnp.int32((v['myis'])+_p557*(jnp.int32(1))), v['i'])
                # WRF:3452 flx(i,k)=flx(i,k)+(dtc(i,k)*emisdg(i,k) +ss2(i,k)*contdg(i,k) +oss(i,k)*to3dg(i,k) +css(i,k)*co21(i,k,k))*cldfac(i,k,k)
                v['flx'] = v['flx'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p554 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx'][(v['i'])-1, (v['k'])-1] + (((((v['dtc'][(v['i'])-1, (v['k'])-1] * v['emisdg'][(v['i'])-1, (v['k'])-1]) + (v['ss2'][(v['i'])-1, (v['k'])-1] * v['contdg'][(v['i'])-1, (v['k'])-1])) + (v['oss'][(v['i'])-1, (v['k'])-1] * v['to3dg'][(v['i'])-1, (v['k'])-1])) + (v['css'][(v['i'])-1, (v['k'])-1] * v['co21'][(v['i'])-1, (v['k'])-1, (v['k'])-1])) * v['cldfac'][(v['i'])-1, (v['k'])-1, (v['k'])-1]))), v['flx'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==871, 0, v['_go'])
                # WRF:3456 continue
                return v
            v = lax.fori_loop(0,_p555,_p556,v)
            v['i'] = jnp.where((_p554 & (v['_go']==0)), jnp.int32((v['myis'])+_p555*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p551,_p552,v)
        v['k'] = jnp.where((_p550 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p551*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3458 do i=myis,myie
        _p558 = ((jnp.bool_(True)) & (v['_go']==0))
        _p559 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p560(_p561,v):
            v = dict(v)
            v['i'] = jnp.where((_p558 & (v['_go']==0)), jnp.int32((v['myis'])+_p561*(jnp.int32(1))), v['i'])
            # WRF:3459 flx(i,l)=flx(i,l)+(css(i,lp1)*co21(i,lp1,l) +dtc(i,lp1)*emspec(i,2) +oss(i,lp1)*to31d(i,l) +ss2(i,lp1)*cont1d(i,l))*cldfac(i,lp1,l)
            v['flx'] = v['flx'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p558 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx'][(v['i'])-1, (v['l'])-1] + (((((v['css'][(v['i'])-1, (v['lp1'])-1] * v['co21'][(v['i'])-1, (v['lp1'])-1, (v['l'])-1]) + (v['dtc'][(v['i'])-1, (v['lp1'])-1] * v['emspec'][(v['i'])-1, (jnp.int32(2))-1])) + (v['oss'][(v['i'])-1, (v['lp1'])-1] * v['to31d'][(v['i'])-1, (v['l'])-1])) + (v['ss2'][(v['i'])-1, (v['lp1'])-1] * v['cont1d'][(v['i'])-1, (v['l'])-1])) * v['cldfac'][(v['i'])-1, (v['lp1'])-1, (v['l'])-1]))), v['flx'][(v['i'])-1, (v['l'])-1]))
            # WRF:3463 flx(i,lp1)=flx(i,lp1)+(css(i,l)*co21(i,l,lp1) +oss(i,l)*to31d(i,l) +ss2(i,l)*cont1d(i,l) +dtc(i,l)*emspec(i,1))*cldfac(i,l,lp1)
            v['flx'] = v['flx'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p558 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx'][(v['i'])-1, (v['lp1'])-1] + (((((v['css'][(v['i'])-1, (v['l'])-1] * v['co21'][(v['i'])-1, (v['l'])-1, (v['lp1'])-1]) + (v['oss'][(v['i'])-1, (v['l'])-1] * v['to31d'][(v['i'])-1, (v['l'])-1])) + (v['ss2'][(v['i'])-1, (v['l'])-1] * v['cont1d'][(v['i'])-1, (v['l'])-1])) + (v['dtc'][(v['i'])-1, (v['l'])-1] * v['emspec'][(v['i'])-1, (jnp.int32(1))-1])) * v['cldfac'][(v['i'])-1, (v['l'])-1, (v['lp1'])-1]))), v['flx'][(v['i'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==873, 0, v['_go'])
            # WRF:3467 continue
            return v
        v = lax.fori_loop(0,_p559,_p560,v)
        v['i'] = jnp.where((_p558 & (v['_go']==0)), jnp.int32((v['myis'])+_p559*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3473 do k=1,l
        _p562 = ((jnp.bool_(True)) & (v['_go']==0))
        _p563 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p564(_p565,v):
            v = dict(v)
            v['k'] = jnp.where((_p562 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p565*(jnp.int32(1))), v['k'])
            # WRF:3474 do i=myis,myie
            _p566 = (((_p562 & (v['_go']==0))) & (v['_go']==0))
            _p567 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p568(_p569,v):
                v = dict(v)
                v['i'] = jnp.where((_p566 & (v['_go']==0)), jnp.int32((v['myis'])+_p569*(jnp.int32(1))), v['i'])
                # WRF:3475 heatem(i,k)=radcon*(flx(i,k+1)-flx(i,k))*delp(i,k)
                v['heatem'] = v['heatem'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p566 & (v['_go']==0))) & (v['_go']==0)), _r(((v['radcon'] * (v['flx'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['flx'][(v['i'])-1, (v['k'])-1])) * v['delp'][(v['i'])-1, (v['k'])-1])), v['heatem'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1101, 0, v['_go'])
                # WRF:3476 continue
                return v
            v = lax.fori_loop(0,_p567,_p568,v)
            v['i'] = jnp.where((_p566 & (v['_go']==0)), jnp.int32((v['myis'])+_p567*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p563,_p564,v)
        v['k'] = jnp.where((_p562 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p563*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3478 do k=1,l
        _p570 = ((jnp.bool_(True)) & (v['_go']==0))
        _p571 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p572(_p573,v):
            v = dict(v)
            v['k'] = jnp.where((_p570 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p573*(jnp.int32(1))), v['k'])
            # WRF:3479 do i=myis,myie
            _p574 = (((_p570 & (v['_go']==0))) & (v['_go']==0))
            _p575 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p576(_p577,v):
                v = dict(v)
                v['i'] = jnp.where((_p574 & (v['_go']==0)), jnp.int32((v['myis'])+_p577*(jnp.int32(1))), v['i'])
                # WRF:3480 heatra(i,k)=heatem(i,k)-cts(i,k)-ctso3(i,k)+excts(i,k)
                v['heatra'] = v['heatra'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p574 & (v['_go']==0))) & (v['_go']==0)), _r((((v['heatem'][(v['i'])-1, (v['k'])-1] - v['cts'][(v['i'])-1, (v['k'])-1]) - v['ctso3'][(v['i'])-1, (v['k'])-1]) + v['excts'][(v['i'])-1, (v['k'])-1])), v['heatra'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1103, 0, v['_go'])
                # WRF:3481 continue
                return v
            v = lax.fori_loop(0,_p575,_p576,v)
            v['i'] = jnp.where((_p574 & (v['_go']==0)), jnp.int32((v['myis'])+_p575*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p571,_p572,v)
        v['k'] = jnp.where((_p570 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p571*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3484 do k=1,l
        _p578 = ((jnp.bool_(True)) & (v['_go']==0))
        _p579 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p580(_p581,v):
            v = dict(v)
            v['k'] = jnp.where((_p578 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p581*(jnp.int32(1))), v['k'])
            # WRF:3485 do i=myis,myie
            _p582 = (((_p578 & (v['_go']==0))) & (v['_go']==0))
            _p583 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p584(_p585,v):
                v = dict(v)
                v['i'] = jnp.where((_p582 & (v['_go']==0)), jnp.int32((v['myis'])+_p585*(jnp.int32(1))), v['i'])
                # WRF:3486 vsum1(i,k)=heatra(i,k)*delp2(i,k)*radcon1
                v['vsum1'] = v['vsum1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p582 & (v['_go']==0))) & (v['_go']==0)), _r(((v['heatra'][(v['i'])-1, (v['k'])-1] * v['delp2'][(v['i'])-1, (v['k'])-1]) * v['radcon1'])), v['vsum1'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1111, 0, v['_go'])
                # WRF:3487 continue
                return v
            v = lax.fori_loop(0,_p583,_p584,v)
            v['i'] = jnp.where((_p582 & (v['_go']==0)), jnp.int32((v['myis'])+_p583*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p579,_p580,v)
        v['k'] = jnp.where((_p578 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p579*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3488 do i=myis,myie
        _p586 = ((jnp.bool_(True)) & (v['_go']==0))
        _p587 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p588(_p589,v):
            v = dict(v)
            v['i'] = jnp.where((_p586 & (v['_go']==0)), jnp.int32((v['myis'])+_p589*(jnp.int32(1))), v['i'])
            # WRF:3489 topflx(i)=flx1e1(i)+gxcts(i)
            v['topflx'] = v['topflx'].at[(v['i'])-1].set(jnp.where((((_p586 & (v['_go']==0))) & (v['_go']==0)), _r((v['flx1e1'][(v['i'])-1] + v['gxcts'][(v['i'])-1])), v['topflx'][(v['i'])-1]))
            # WRF:3490 flxnet(i,1)=topflx(i)
            v['flxnet'] = v['flxnet'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p586 & (v['_go']==0))) & (v['_go']==0)), _r(v['topflx'][(v['i'])-1]), v['flxnet'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1115, 0, v['_go'])
            # WRF:3491 continue
            return v
        v = lax.fori_loop(0,_p587,_p588,v)
        v['i'] = jnp.where((_p586 & (v['_go']==0)), jnp.int32((v['myis'])+_p587*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3494 do k=2,lp1
        _p590 = ((jnp.bool_(True)) & (v['_go']==0))
        _p591 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p592(_p593,v):
            v = dict(v)
            v['k'] = jnp.where((_p590 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p593*(jnp.int32(1))), v['k'])
            # WRF:3495 do i=myis,myie
            _p594 = (((_p590 & (v['_go']==0))) & (v['_go']==0))
            _p595 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p596(_p597,v):
                v = dict(v)
                v['i'] = jnp.where((_p594 & (v['_go']==0)), jnp.int32((v['myis'])+_p597*(jnp.int32(1))), v['i'])
                # WRF:3496 flxnet(i,k)=flxnet(i,k-1)+vsum1(i,k-1)
                v['flxnet'] = v['flxnet'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p594 & (v['_go']==0))) & (v['_go']==0)), _r((v['flxnet'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['vsum1'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1])), v['flxnet'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1123, 0, v['_go'])
                # WRF:3497 continue
                return v
            v = lax.fori_loop(0,_p595,_p596,v)
            v['i'] = jnp.where((_p594 & (v['_go']==0)), jnp.int32((v['myis'])+_p595*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p591,_p592,v)
        v['k'] = jnp.where((_p590 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p591*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3498 do i=myis,myie
        _p598 = ((jnp.bool_(True)) & (v['_go']==0))
        _p599 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p600(_p601,v):
            v = dict(v)
            v['i'] = jnp.where((_p598 & (v['_go']==0)), jnp.int32((v['myis'])+_p601*(jnp.int32(1))), v['i'])
            # WRF:3499 grnflx(i)=flxnet(i,lp1)
            v['grnflx'] = v['grnflx'].at[(v['i'])-1].set(jnp.where((((_p598 & (v['_go']==0))) & (v['_go']==0)), _r(v['flxnet'][(v['i'])-1, (v['lp1'])-1]), v['grnflx'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1125, 0, v['_go'])
            # WRF:3500 continue
            return v
        v = lax.fori_loop(0,_p599,_p600,v)
        v['i'] = jnp.where((_p598 & (v['_go']==0)), jnp.int32((v['myis'])+_p599*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3507 icnt=0
        v['icnt'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(jnp.int32(0)), v['icnt'])
        # WRF:3508 do i=myis,myie
        _p602 = ((jnp.bool_(True)) & (v['_go']==0))
        _p603 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p604(_p605,v):
            v = dict(v)
            v['i'] = jnp.where((_p602 & (v['_go']==0)), jnp.int32((v['myis'])+_p605*(jnp.int32(1))), v['i'])
            # WRF:3509 icnt=icnt+nclds(i)
            v['icnt'] = jnp.where((((_p602 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['icnt'] + v['nclds'][(v['i'])-1])), v['icnt'])
            v['_go'] = jnp.where(v['_go']==1301, 0, v['_go'])
            # WRF:3510 continue
            return v
        v = lax.fori_loop(0,_p603,_p604,v)
        v['i'] = jnp.where((_p602 & (v['_go']==0)), jnp.int32((v['myis'])+_p603*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3511 if (icnt.eq.0) go to 6999
        _p606 = ((jnp.bool_(True)) & (v['_go']==0)) & ((v['icnt'] == jnp.int32(0)))
        # WRF:3511 go to 6999
        v['_go'] = jnp.where(((_p606) & (v['_go']==0)), 6999, v['_go'])
        # WRF:3513 kclds=nclds(myis)
        v['kclds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['nclds'][(v['myis'])-1]), v['kclds'])
        # WRF:3514 do i=myis,myie
        _p607 = ((jnp.bool_(True)) & (v['_go']==0))
        _p608 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p609(_p610,v):
            v = dict(v)
            v['i'] = jnp.where((_p607 & (v['_go']==0)), jnp.int32((v['myis'])+_p610*(jnp.int32(1))), v['i'])
            # WRF:3515 kclds=max(nclds(i),kclds)
            v['kclds'] = jnp.where((((_p607 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_max(v['nclds'][(v['i'])-1], v['kclds'])), v['kclds'])
            v['_go'] = jnp.where(v['_go']==2106, 0, v['_go'])
            # WRF:3516 continue
            return v
        v = lax.fori_loop(0,_p608,_p609,v)
        v['i'] = jnp.where((_p607 & (v['_go']==0)), jnp.int32((v['myis'])+_p608*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3522 do kk=1,kclds
        _p611 = ((jnp.bool_(True)) & (v['_go']==0))
        _p612 = jnp.maximum(jnp.int32(0), ((v['kclds'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p613(_p614,v):
            v = dict(v)
            v['kk'] = jnp.where((_p611 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p614*(jnp.int32(1))), v['kk'])
            # WRF:3523 kmin=lp1
            v['kmin'] = jnp.where((((_p611 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['lp1']), v['kmin'])
            # WRF:3524 kmax=0
            v['kmax'] = jnp.where((((_p611 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(jnp.int32(0)), v['kmax'])
            # WRF:3525 do i=myis,myie
            _p615 = (((_p611 & (v['_go']==0))) & (v['_go']==0))
            _p616 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p617(_p618,v):
                v = dict(v)
                v['i'] = jnp.where((_p615 & (v['_go']==0)), jnp.int32((v['myis'])+_p618*(jnp.int32(1))), v['i'])
                # WRF:3526 j1=ktop(i,kk+1)
                v['j1'] = jnp.where((((_p615 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['ktop'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]), v['j1'])
                # WRF:3528 j3=kbtm(i,kk+1)
                v['j3'] = jnp.where((((_p615 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtm'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]), v['j3'])
                # WRF:3529 __if__ j3.gt.j1
                _p619 = (((_p615 & (v['_go']==0))) & (v['_go']==0)) & ((v['j3'] > v['j1']))
                _p620 = (((_p615 & (v['_go']==0))) & (v['_go']==0)) & ~((v['j3'] > v['j1']))
                # WRF:3530 ptop(i)=p(i,j1)
                v['ptop'] = v['ptop'].at[(v['i'])-1].set(jnp.where(((_p619) & (v['_go']==0)), _r(v['p'][(v['i'])-1, (v['j1'])-1]), v['ptop'][(v['i'])-1]))
                # WRF:3531 pbot(i)=p(i,j3+1)
                v['pbot'] = v['pbot'].at[(v['i'])-1].set(jnp.where(((_p619) & (v['_go']==0)), _r(v['p'][(v['i'])-1, ((v['j3'] + jnp.int32(1)))-1]), v['pbot'][(v['i'])-1]))
                # WRF:3532 ftop(i)=flxnet(i,j1)
                v['ftop'] = v['ftop'].at[(v['i'])-1].set(jnp.where(((_p619) & (v['_go']==0)), _r(v['flxnet'][(v['i'])-1, (v['j1'])-1]), v['ftop'][(v['i'])-1]))
                # WRF:3533 fbot(i)=flxnet(i,j3+1)
                v['fbot'] = v['fbot'].at[(v['i'])-1].set(jnp.where(((_p619) & (v['_go']==0)), _r(v['flxnet'][(v['i'])-1, ((v['j3'] + jnp.int32(1)))-1]), v['fbot'][(v['i'])-1]))
                # WRF:3535 delptc(i)=(ftop(i)-fbot(i))/(ptop(i)-pbot(i))
                v['delptc'] = v['delptc'].at[(v['i'])-1].set(jnp.where(((_p619) & (v['_go']==0)), _r(((v['ftop'][(v['i'])-1] - v['fbot'][(v['i'])-1]) / (v['ptop'][(v['i'])-1] - v['pbot'][(v['i'])-1]))), v['delptc'][(v['i'])-1]))
                # WRF:3536 kmin=min(kmin,j1)
                v['kmin'] = jnp.where(((_p619) & (v['_go']==0)), jnp.int32(_min(v['kmin'], v['j1'])), v['kmin'])
                # WRF:3537 kmax=max(kmax,j3)
                v['kmax'] = jnp.where(((_p619) & (v['_go']==0)), jnp.int32(_max(v['kmax'], v['j3'])), v['kmax'])
                v['_go'] = jnp.where(v['_go']==1362, 0, v['_go'])
                # WRF:3539 continue
                return v
            v = lax.fori_loop(0,_p616,_p617,v)
            v['i'] = jnp.where((_p615 & (v['_go']==0)), jnp.int32((v['myis'])+_p616*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:3540 kmin=kmin+1
            v['kmin'] = jnp.where((((_p611 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['kmin'] + jnp.int32(1))), v['kmin'])
            # WRF:3543 do k=kmin,kmax
            _p621 = (((_p611 & (v['_go']==0))) & (v['_go']==0))
            _p622 = jnp.maximum(jnp.int32(0), ((v['kmax'])-(v['kmin']))//(jnp.int32(1))+jnp.int32(1))
            def _p623(_p624,v):
                v = dict(v)
                v['k'] = jnp.where((_p621 & (v['_go']==0)), jnp.int32((v['kmin'])+_p624*(jnp.int32(1))), v['k'])
                # WRF:3544 do i=myis,myie
                _p625 = (((_p621 & (v['_go']==0))) & (v['_go']==0))
                _p626 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p627(_p628,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p625 & (v['_go']==0)), jnp.int32((v['myis'])+_p628*(jnp.int32(1))), v['i'])
                    # WRF:3546 __if__ ktop(i,kk+1).lt.k .and. k.le.kbtm(i,kk+1)
                    _p629 = (((_p625 & (v['_go']==0))) & (v['_go']==0)) & (((v['ktop'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] < v['k']) & (v['k'] <= v['kbtm'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])))
                    _p630 = (((_p625 & (v['_go']==0))) & (v['_go']==0)) & ~(((v['ktop'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] < v['k']) & (v['k'] <= v['kbtm'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])))
                    # WRF:3547 z1(i,k)=(p(i,k)-ptop(i))*delptc(i)+ftop(i)
                    v['z1'] = v['z1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where(((_p629) & (v['_go']==0)), _r((((v['p'][(v['i'])-1, (v['k'])-1] - v['ptop'][(v['i'])-1]) * v['delptc'][(v['i'])-1]) + v['ftop'][(v['i'])-1])), v['z1'][(v['i'])-1, (v['k'])-1]))
                    # WRF:3550 flxnet(i,k)=z1(i,k)
                    v['flxnet'] = v['flxnet'].at[(v['i'])-1, (v['k'])-1].set(jnp.where(((_p629) & (v['_go']==0)), _r(v['z1'][(v['i'])-1, (v['k'])-1]), v['flxnet'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==1363, 0, v['_go'])
                    # WRF:3552 continue
                    return v
                v = lax.fori_loop(0,_p626,_p627,v)
                v['i'] = jnp.where((_p625 & (v['_go']==0)), jnp.int32((v['myis'])+_p626*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==1365, 0, v['_go'])
                # WRF:3553 continue
                return v
            v = lax.fori_loop(0,_p622,_p623,v)
            v['k'] = jnp.where((_p621 & (v['_go']==0)), jnp.int32((v['kmin'])+_p622*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1361, 0, v['_go'])
            # WRF:3554 continue
            return v
        v = lax.fori_loop(0,_p612,_p613,v)
        v['kk'] = jnp.where((_p611 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p612*(jnp.int32(1))), v['kk'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        v['_go'] = jnp.where(v['_go']==6001, 0, v['_go'])
        # WRF:3573 continue
        v['_go'] = jnp.where(v['_go']==6999, 0, v['_go'])
        # WRF:3574 continue
        # WRF:3577 do k=1,l
        _p631 = ((jnp.bool_(True)) & (v['_go']==0))
        _p632 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p633(_p634,v):
            v = dict(v)
            v['k'] = jnp.where((_p631 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p634*(jnp.int32(1))), v['k'])
            # WRF:3578 do i=myis,myie
            _p635 = (((_p631 & (v['_go']==0))) & (v['_go']==0))
            _p636 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p637(_p638,v):
                v = dict(v)
                v['i'] = jnp.where((_p635 & (v['_go']==0)), jnp.int32((v['myis'])+_p638*(jnp.int32(1))), v['i'])
                # WRF:3579 heatra(i,k)=radcon*(flxnet(i,k+1)-flxnet(i,k))*delp(i,k)
                v['heatra'] = v['heatra'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p635 & (v['_go']==0))) & (v['_go']==0)), _r(((v['radcon'] * (v['flxnet'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['flxnet'][(v['i'])-1, (v['k'])-1])) * v['delp'][(v['i'])-1, (v['k'])-1])), v['heatra'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==6101, 0, v['_go'])
                # WRF:3580 continue
                return v
            v = lax.fori_loop(0,_p636,_p637,v)
            v['i'] = jnp.where((_p635 & (v['_go']==0)), jnp.int32((v['myis'])+_p636*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p632,_p633,v)
        v['k'] = jnp.where((_p631 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p632*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['heatra'], v['grnflx'], v['topflx'], v['qh2o'], v['press'], v['p'], v['delp'], v['delp2'], v['temp'], v['t'], v['cldfac'], v['nclds'], v['ktop'], v['kbtm'], v['camt'], v['co21'], v['co2nbl'], v['co2sp1'], v['co2sp2'], v['var1'], v['var2'], v['var3'], v['var4'], v['cntval'], v['toto3'], v['tphio3'], v['totphi'], v['totvo2'], v['emx1'], v['emx2'], v['empl'], v['bo3rnd'], v['ao3rnd'], v['apcm'], v['bpcm'], v['atpcm'], v['btpcm'], v['acomb'], v['bcomb'], v['betacm'], v['ten'], v['hp1'], v['haf'], v['one'], v['four'], v['hm1ez'], v['radcon'], v['quartr'], v['two'], v['hm6666m2'], v['hmp66667'], v['hmp5'], v['hp166666'], v['h41666m2'], v['radcon1'], v['h16e1'], v['h28e1'], v['h25e2'], v['h44194m2'], v['h1p41819'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)

    def _e1e290(g1, g2, g3, g4, g5, emiss, fxoe1, dte1, fxoe2, dte2, avephi, temp, t, h16e1, ten, hp1, h28e1, haf, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(g1).dtype == jnp.dtype("float32") else jnp.float64
        v = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4, 'g5': g5, 'emiss': emiss, 'fxoe1': fxoe1, 'dte1': dte1, 'fxoe2': fxoe2, 'dte2': dte2, 'avephi': avephi, 'temp': temp, 't': t, 'h16e1': h16e1, 'ten': ten, 'hp1': hp1, 'h28e1': h28e1, 'haf': haf, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['du'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['fyo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['it1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*3+2)-((1))+1,), dtype=jnp.int32)
        v['ival'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=jnp.int32)
        v['k'] = jnp.int32(0)
        v['kp'] = jnp.int32(0)
        v['l'] = jnp.int32(0)
        v['ll'] = jnp.int32(0)
        v['llm1'] = jnp.int32(0)
        v['llm2'] = jnp.int32(0)
        v['llp1'] = jnp.int32(0)
        v['lm1'] = jnp.int32(0)
        v['lm2'] = jnp.int32(0)
        v['lm3'] = jnp.int32(0)
        v['lp1'] = jnp.int32(0)
        v['lp2'] = jnp.int32(0)
        v['lp3'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['tmp3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ww1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ww2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        # WRF:3621 l=kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:3622 lp1=l+1
        v['lp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(1))), v['lp1'])
        # WRF:3622 lp2=l+2
        v['lp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(2))), v['lp2'])
        # WRF:3622 lp3=l+3
        v['lp3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(3))), v['lp3'])
        # WRF:3622 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:3623 lm1=l-1
        v['lm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(1))), v['lm1'])
        # WRF:3623 lm2=l-2
        v['lm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(2))), v['lm2'])
        # WRF:3623 lm3=l-3
        v['lm3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(3))), v['lm3'])
        # WRF:3623 ll = 2*l
        v['ll'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((jnp.int32(2) * v['l'])), v['ll'])
        # WRF:3624 llm2 = ll-2
        v['llm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(2))), v['llm2'])
        # WRF:3624 llm1=ll-1
        v['llm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(1))), v['llm1'])
        # WRF:3625 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:3625 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:3637 do k=1,lp1
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p4*(jnp.int32(1))), v['k'])
            # WRF:3638 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:3639 tmp3(i,k)=log10(avephi(i,k))+h16e1
                v['tmp3'] = v['tmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.log10(v['avephi'][(v['i'])-1, (v['k'])-1]) + v['h16e1'])), v['tmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:3640 fyo(i,k)=aint(tmp3(i,k)*ten)
                v['fyo'] = v['fyo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['tmp3'][(v['i'])-1, (v['k'])-1] * v['ten']))), v['fyo'][(v['i'])-1, (v['k'])-1]))
                # WRF:3641 du(i,k)=tmp3(i,k)-hp1*fyo(i,k)
                v['du'] = v['du'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['tmp3'][(v['i'])-1, (v['k'])-1] - (v['hp1'] * v['fyo'][(v['i'])-1, (v['k'])-1]))), v['du'][(v['i'])-1, (v['k'])-1]))
                # WRF:3642 fyo(i,k)=h28e1*fyo(i,k)
                v['fyo'] = v['fyo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['h28e1'] * v['fyo'][(v['i'])-1, (v['k'])-1])), v['fyo'][(v['i'])-1, (v['k'])-1]))
                # WRF:3643 ival(i,k)=fyo(i,k)+fxoe2(i,k)
                v['ival'] = v['ival'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fyo'][(v['i'])-1, (v['k'])-1] + v['fxoe2'][(v['i'])-1, (v['k'])-1])), v['ival'][(v['i'])-1, (v['k'])-1]))
                # WRF:3644 emiss(i,k)=t1(ival(i,k))+du(i,k)*t2(ival(i,k)) +dte2(i,k)*t4(ival(i,k))
                v['emiss'] = v['emiss'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(((_r(_TABLES['t1'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1] + (v['du'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t2'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1])) + (v['dte2'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t4'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1]))), v['emiss'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1322, 0, v['_go'])
                # WRF:3646 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3650 do i=myis,myie
        _p9 = ((jnp.bool_(True)) & (v['_go']==0))
        _p10 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p11(_p12,v):
            v = dict(v)
            v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p12*(jnp.int32(1))), v['i'])
            # WRF:3651 emiss(i,l)=haf*(emiss(i,l)+emiss(i,lp1))
            v['emiss'] = v['emiss'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['emiss'][(v['i'])-1, (v['l'])-1] + v['emiss'][(v['i'])-1, (v['lp1'])-1]))), v['emiss'][(v['i'])-1, (v['l'])-1]))
            v['_go'] = jnp.where(v['_go']==1344, 0, v['_go'])
            # WRF:3652 continue
            return v
        v = lax.fori_loop(0,_p10,_p11,v)
        v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p10*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3670 do i=myis,myie
        _p13 = ((jnp.bool_(True)) & (v['_go']==0))
        _p14 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p15(_p16,v):
            v = dict(v)
            v['i'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((v['myis'])+_p16*(jnp.int32(1))), v['i'])
            # WRF:3671 it1(i,1)=fxoe1(i,1)
            v['it1'] = v['it1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p13 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['fxoe1'][(v['i'])-1, (jnp.int32(1))-1]), v['it1'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3672 ww1(i,1)=ten-dte1(i,1)
            v['ww1'] = v['ww1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p13 & (v['_go']==0))) & (v['_go']==0)), _r((v['ten'] - v['dte1'][(v['i'])-1, (jnp.int32(1))-1])), v['ww1'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3673 ww2(i,1)=hp1
            v['ww2'] = v['ww2'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p13 & (v['_go']==0))) & (v['_go']==0)), _r(v['hp1']), v['ww2'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==208, 0, v['_go'])
            # WRF:3674 continue
            return v
        v = lax.fori_loop(0,_p14,_p15,v)
        v['i'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((v['myis'])+_p14*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3675 do k=1,l
        _p17 = ((jnp.bool_(True)) & (v['_go']==0))
        _p18 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p19(_p20,v):
            v = dict(v)
            v['k'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p20*(jnp.int32(1))), v['k'])
            # WRF:3676 do i=myis,myie
            _p21 = (((_p17 & (v['_go']==0))) & (v['_go']==0))
            _p22 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p23(_p24,v):
                v = dict(v)
                v['i'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((v['myis'])+_p24*(jnp.int32(1))), v['i'])
                # WRF:3677 it1(i,k+1)=fyo(i,k)+fxoe1(i,k+1)
                v['it1'] = v['it1'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p21 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fyo'][(v['i'])-1, (v['k'])-1] + v['fxoe1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3678 it1(i,lp2+k-1)=fyo(i,k)+fxoe1(i,k)
                v['it1'] = v['it1'].at[(v['i'])-1, (((v['lp2'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p21 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fyo'][(v['i'])-1, (v['k'])-1] + v['fxoe1'][(v['i'])-1, (v['k'])-1])), v['it1'][(v['i'])-1, (((v['lp2'] + v['k']) - jnp.int32(1)))-1]))
                # WRF:3679 ww1(i,k+1)=ten-dte1(i,k+1)
                v['ww1'] = v['ww1'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p21 & (v['_go']==0))) & (v['_go']==0)), _r((v['ten'] - v['dte1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['ww1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3680 ww2(i,k+1)=hp1-du(i,k)
                v['ww2'] = v['ww2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p21 & (v['_go']==0))) & (v['_go']==0)), _r((v['hp1'] - v['du'][(v['i'])-1, (v['k'])-1])), v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==209, 0, v['_go'])
                # WRF:3681 continue
                return v
            v = lax.fori_loop(0,_p22,_p23,v)
            v['i'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((v['myis'])+_p22*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p18,_p19,v)
        v['k'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p18*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3682 do kp=1,l
        _p25 = ((jnp.bool_(True)) & (v['_go']==0))
        _p26 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p27(_p28,v):
            v = dict(v)
            v['kp'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p28*(jnp.int32(1))), v['kp'])
            # WRF:3683 do i=myis,myie
            _p29 = (((_p25 & (v['_go']==0))) & (v['_go']==0))
            _p30 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p31(_p32,v):
                v = dict(v)
                v['i'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((v['myis'])+_p32*(jnp.int32(1))), v['i'])
                # WRF:3684 it1(i,kp+llp1)=fyo(i,kp)+fxoe1(i,1)
                v['it1'] = v['it1'].at[(v['i'])-1, ((v['kp'] + v['llp1']))-1].set(jnp.where((((_p29 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fyo'][(v['i'])-1, (v['kp'])-1] + v['fxoe1'][(v['i'])-1, (jnp.int32(1))-1])), v['it1'][(v['i'])-1, ((v['kp'] + v['llp1']))-1]))
                v['_go'] = jnp.where(v['_go']==211, 0, v['_go'])
                # WRF:3685 continue
                return v
            v = lax.fori_loop(0,_p30,_p31,v)
            v['i'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((v['myis'])+_p30*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p26,_p27,v)
        v['kp'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p26*(jnp.int32(1))), v['kp'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3689 do i=myis,myie
        _p33 = ((jnp.bool_(True)) & (v['_go']==0))
        _p34 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p35(_p36,v):
            v = dict(v)
            v['i'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p36*(jnp.int32(1))), v['i'])
            # WRF:3690 g1(i,1)=ww1(i,1)*ww2(i,1)*em1v(it1(i,1))+ ww2(i,1)*dte1(i,1)*em1v(it1(i,1)+1)
            v['g1'] = v['g1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r((((v['ww1'][(v['i'])-1, (jnp.int32(1))-1] * v['ww2'][(v['i'])-1, (jnp.int32(1))-1]) * _r(_TABLES['em1v'])[(v['it1'][(v['i'])-1, (jnp.int32(1))-1])-1]) + ((v['ww2'][(v['i'])-1, (jnp.int32(1))-1] * v['dte1'][(v['i'])-1, (jnp.int32(1))-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, (jnp.int32(1))-1] + jnp.int32(1)))-1]))), v['g1'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3692 g3(i,1)=g1(i,1)
            v['g3'] = v['g3'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r(v['g1'][(v['i'])-1, (jnp.int32(1))-1]), v['g3'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==230, 0, v['_go'])
            # WRF:3693 continue
            return v
        v = lax.fori_loop(0,_p34,_p35,v)
        v['i'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p34*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3694 do k=1,l
        _p37 = ((jnp.bool_(True)) & (v['_go']==0))
        _p38 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p39(_p40,v):
            v = dict(v)
            v['k'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p40*(jnp.int32(1))), v['k'])
            # WRF:3695 do i=myis,myie
            _p41 = (((_p37 & (v['_go']==0))) & (v['_go']==0))
            _p42 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p43(_p44,v):
                v = dict(v)
                v['i'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((v['myis'])+_p44*(jnp.int32(1))), v['i'])
                # WRF:3696 g1(i,k+1)=ww1(i,k+1)*ww2(i,k+1)*em1v(it1(i,k+1))+ ww2(i,k+1)*dte1(i,k+1)*em1v(it1(i,k+1)+1)+ ww1(i,k+1)*du(i,k)*em1v(it1(i,k+1)+28)+ dte1(i,k+1)*du(i,k)*em1v(it1(i,k+1)+29)
                v['g1'] = v['g1'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p41 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['ww1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['em1v'])[(v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])-1]) + ((v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['dte1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + jnp.int32(1)))-1])) + ((v['ww1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + jnp.int32(28)))-1])) + ((v['dte1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + jnp.int32(29)))-1]))), v['g1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3700 g2(i,k)=ww1(i,k)*ww2(i,k+1)*em1v(it1(i,k+lp2-1))+ ww2(i,k+1)*dte1(i,k)*em1v(it1(i,k+lp2-1)+1)+ ww1(i,k)*du(i,k)*em1v(it1(i,k+lp2-1)+28)+ dte1(i,k)*du(i,k)*em1v(it1(i,k+lp2-1)+29)
                v['g2'] = v['g2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p41 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['ww1'][(v['i'])-1, (v['k'])-1] * v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['em1v'])[(v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1])-1]) + ((v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['dte1'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1] + jnp.int32(1)))-1])) + ((v['ww1'][(v['i'])-1, (v['k'])-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1] + jnp.int32(28)))-1])) + ((v['dte1'][(v['i'])-1, (v['k'])-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1] + jnp.int32(29)))-1]))), v['g2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==240, 0, v['_go'])
                # WRF:3704 continue
                return v
            v = lax.fori_loop(0,_p42,_p43,v)
            v['i'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((v['myis'])+_p42*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p38,_p39,v)
        v['k'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p38*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3705 do kp=2,lp1
        _p45 = ((jnp.bool_(True)) & (v['_go']==0))
        _p46 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p47(_p48,v):
            v = dict(v)
            v['kp'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p48*(jnp.int32(1))), v['kp'])
            # WRF:3706 do i=myis,myie
            _p49 = (((_p45 & (v['_go']==0))) & (v['_go']==0))
            _p50 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p51(_p52,v):
                v = dict(v)
                v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p52*(jnp.int32(1))), v['i'])
                # WRF:3707 g3(i,kp)=ww1(i,1)*ww2(i,kp)*em1v(it1(i,ll+kp))+ ww2(i,kp)*dte1(i,1)*em1v(it1(i,ll+kp)+1)+ ww1(i,1)*du(i,kp-1)*em1v(it1(i,ll+kp)+28)+ dte1(i,1)*du(i,kp-1)*em1v(it1(i,ll+kp)+29)
                v['g3'] = v['g3'].at[(v['i'])-1, (v['kp'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['ww1'][(v['i'])-1, (jnp.int32(1))-1] * v['ww2'][(v['i'])-1, (v['kp'])-1]) * _r(_TABLES['em1v'])[(v['it1'][(v['i'])-1, ((v['ll'] + v['kp']))-1])-1]) + ((v['ww2'][(v['i'])-1, (v['kp'])-1] * v['dte1'][(v['i'])-1, (jnp.int32(1))-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, ((v['ll'] + v['kp']))-1] + jnp.int32(1)))-1])) + ((v['ww1'][(v['i'])-1, (jnp.int32(1))-1] * v['du'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, ((v['ll'] + v['kp']))-1] + jnp.int32(28)))-1])) + ((v['dte1'][(v['i'])-1, (jnp.int32(1))-1] * v['du'][(v['i'])-1, ((v['kp'] - jnp.int32(1)))-1]) * _r(_TABLES['em1v'])[((v['it1'][(v['i'])-1, ((v['ll'] + v['kp']))-1] + jnp.int32(29)))-1]))), v['g3'][(v['i'])-1, (v['kp'])-1]))
                v['_go'] = jnp.where(v['_go']==241, 0, v['_go'])
                # WRF:3711 continue
                return v
            v = lax.fori_loop(0,_p50,_p51,v)
            v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p50*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p46,_p47,v)
        v['kp'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p46*(jnp.int32(1))), v['kp'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3713 do i=myis,myie
        _p53 = ((jnp.bool_(True)) & (v['_go']==0))
        _p54 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p55(_p56,v):
            v = dict(v)
            v['i'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((v['myis'])+_p56*(jnp.int32(1))), v['i'])
            # WRF:3714 g4(i,1)=ww1(i,1)*ww2(i,1)*em1vw(it1(i,1))+ ww2(i,1)*dte1(i,1)*em1vw(it1(i,1)+1)
            v['g4'] = v['g4'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p53 & (v['_go']==0))) & (v['_go']==0)), _r((((v['ww1'][(v['i'])-1, (jnp.int32(1))-1] * v['ww2'][(v['i'])-1, (jnp.int32(1))-1]) * _r(_TABLES['em1vw'])[(v['it1'][(v['i'])-1, (jnp.int32(1))-1])-1]) + ((v['ww2'][(v['i'])-1, (jnp.int32(1))-1] * v['dte1'][(v['i'])-1, (jnp.int32(1))-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, (jnp.int32(1))-1] + jnp.int32(1)))-1]))), v['g4'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==244, 0, v['_go'])
            # WRF:3716 continue
            return v
        v = lax.fori_loop(0,_p54,_p55,v)
        v['i'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((v['myis'])+_p54*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3717 do k=1,l
        _p57 = ((jnp.bool_(True)) & (v['_go']==0))
        _p58 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p59(_p60,v):
            v = dict(v)
            v['k'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p60*(jnp.int32(1))), v['k'])
            # WRF:3718 do i=myis,myie
            _p61 = (((_p57 & (v['_go']==0))) & (v['_go']==0))
            _p62 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p63(_p64,v):
                v = dict(v)
                v['i'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((v['myis'])+_p64*(jnp.int32(1))), v['i'])
                # WRF:3719 g4(i,k+1)=ww1(i,k+1)*ww2(i,k+1)*em1vw(it1(i,k+1))+ ww2(i,k+1)*dte1(i,k+1)*em1vw(it1(i,k+1)+1)+ ww1(i,k+1)*du(i,k)*em1vw(it1(i,k+1)+28)+ dte1(i,k+1)*du(i,k)*em1vw(it1(i,k+1)+29)
                v['g4'] = v['g4'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['ww1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['em1vw'])[(v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])-1]) + ((v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['dte1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + jnp.int32(1)))-1])) + ((v['ww1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + jnp.int32(28)))-1])) + ((v['dte1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + jnp.int32(29)))-1]))), v['g4'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:3723 g5(i,k)=ww1(i,k)*ww2(i,k+1)*em1vw(it1(i,k+lp2-1))+ ww2(i,k+1)*dte1(i,k)*em1vw(it1(i,k+lp2-1)+1)+ ww1(i,k)*du(i,k)*em1vw(it1(i,k+lp2-1)+28)+ dte1(i,k)*du(i,k)*em1vw(it1(i,k+lp2-1)+29)
                v['g5'] = v['g5'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['ww1'][(v['i'])-1, (v['k'])-1] * v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['em1vw'])[(v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1])-1]) + ((v['ww2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['dte1'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1] + jnp.int32(1)))-1])) + ((v['ww1'][(v['i'])-1, (v['k'])-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1] + jnp.int32(28)))-1])) + ((v['dte1'][(v['i'])-1, (v['k'])-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em1vw'])[((v['it1'][(v['i'])-1, (((v['k'] + v['lp2']) - jnp.int32(1)))-1] + jnp.int32(29)))-1]))), v['g5'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==242, 0, v['_go'])
                # WRF:3727 continue
                return v
            v = lax.fori_loop(0,_p62,_p63,v)
            v['i'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((v['myis'])+_p62*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p58,_p59,v)
        v['k'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p58*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['g1'], v['g2'], v['g3'], v['g4'], v['g5'], v['emiss'], v['fxoe1'], v['dte1'], v['fxoe2'], v['dte2'], v['avephi'], v['temp'], v['t'], v['h16e1'], v['ten'], v['hp1'], v['h28e1'], v['haf'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)

    def _spa88(excts, ctso3, gxcts, sorc, csour, cldfac, temp, press, var1, var2, p, delp, delp2, totvo2, to3sp, to3spc, co2sp1, co2sp2, co2sp, apcm, bpcm, atpcm, btpcm, acomb, bcomb, betacm, h25e2, one, h44194m2, h1p41819, haf, hm1ez, two, radcon, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(excts).dtype == jnp.dtype("float32") else jnp.float64
        v = {'excts': excts, 'ctso3': ctso3, 'gxcts': gxcts, 'sorc': sorc, 'csour': csour, 'cldfac': cldfac, 'temp': temp, 'press': press, 'var1': var1, 'var2': var2, 'p': p, 'delp': delp, 'delp2': delp2, 'totvo2': totvo2, 'to3sp': to3sp, 'to3spc': to3spc, 'co2sp1': co2sp1, 'co2sp2': co2sp2, 'co2sp': co2sp, 'apcm': apcm, 'bpcm': bpcm, 'atpcm': atpcm, 'btpcm': btpcm, 'acomb': acomb, 'bcomb': bcomb, 'betacm': betacm, 'h25e2': h25e2, 'one': one, 'h44194m2': h44194m2, 'h1p41819': h1p41819, 'haf': haf, 'hm1ez': hm1ez, 'two': two, 'radcon': radcon, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['ag'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['agg'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['ctmp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ctmp2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ctmp3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['f'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['fac1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['fac2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['ff'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['k'] = jnp.int32(0)
        v['l'] = jnp.int32(0)
        v['ll'] = jnp.int32(0)
        v['llm1'] = jnp.int32(0)
        v['llm2'] = jnp.int32(0)
        v['llp1'] = jnp.int32(0)
        v['lm1'] = jnp.int32(0)
        v['lm2'] = jnp.int32(0)
        v['lm3'] = jnp.int32(0)
        v['lp1'] = jnp.int32(0)
        v['lp2'] = jnp.int32(0)
        v['lp3'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['phitmp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['psitmp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['topm'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['topphi'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['tt'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['x'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        v['y'] = jnp.zeros((((1))-((1))+1,((_NLAYERS))-((1))+1,), dtype=_r)
        # WRF:3779 l=kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:3780 lp1=l+1
        v['lp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(1))), v['lp1'])
        # WRF:3780 lp2=l+2
        v['lp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(2))), v['lp2'])
        # WRF:3780 lp3=l+3
        v['lp3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(3))), v['lp3'])
        # WRF:3780 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:3781 lm1=l-1
        v['lm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(1))), v['lm1'])
        # WRF:3781 lm2=l-2
        v['lm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(2))), v['lm2'])
        # WRF:3781 lm3=l-3
        v['lm3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(3))), v['lm3'])
        # WRF:3781 ll = 2*l
        v['ll'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((jnp.int32(2) * v['l'])), v['ll'])
        # WRF:3782 llm2 = ll-2
        v['llm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(2))), v['llm2'])
        # WRF:3782 llm1=ll-1
        v['llm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(1))), v['llm1'])
        # WRF:3783 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:3783 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:3787 do k=1,l
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p4*(jnp.int32(1))), v['k'])
            # WRF:3788 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:3789 x(i,k)=temp(i,k)-h25e2
                v['x'] = v['x'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['temp'][(v['i'])-1, (v['k'])-1] - v['h25e2'])), v['x'][(v['i'])-1, (v['k'])-1]))
                # WRF:3790 y(i,k)=x(i,k)*x(i,k)
                v['y'] = v['y'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['x'][(v['i'])-1, (v['k'])-1] * v['x'][(v['i'])-1, (v['k'])-1])), v['y'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==101, 0, v['_go'])
                # WRF:3791 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3794 do i=myis,myie
        _p9 = ((jnp.bool_(True)) & (v['_go']==0))
        _p10 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p11(_p12,v):
            v = dict(v)
            v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p12*(jnp.int32(1))), v['i'])
            # WRF:3795 ctmp(i,1)=one
            v['ctmp'] = v['ctmp'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(v['one']), v['ctmp'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3796 ctmp2(i,1)=1.
            v['ctmp2'] = v['ctmp2'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(_r(1.)), v['ctmp2'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3797 ctmp3(i,1)=1.
            v['ctmp3'] = v['ctmp3'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r(_r(1.)), v['ctmp3'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==345, 0, v['_go'])
            # WRF:3798 continue
            return v
        v = lax.fori_loop(0,_p10,_p11,v)
        v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p10*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3806 do k=1,l
        _p13 = ((jnp.bool_(True)) & (v['_go']==0))
        _p14 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p15(_p16,v):
            v = dict(v)
            v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p16*(jnp.int32(1))), v['k'])
            # WRF:3807 do i=myis,myie
            _p17 = (((_p13 & (v['_go']==0))) & (v['_go']==0))
            _p18 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p19(_p20,v):
                v = dict(v)
                v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p20*(jnp.int32(1))), v['i'])
                # WRF:3808 f(i,k)=h44194m2*(apcm(1)*x(i,k)+bpcm(1)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(1))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(1))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:3809 ff(i,k)=h44194m2*(atpcm(1)*x(i,k)+btpcm(1)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(1))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(1))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:3810 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:3811 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:3812 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:3813 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==301, 0, v['_go'])
                # WRF:3814 continue
                return v
            v = lax.fori_loop(0,_p18,_p19,v)
            v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p18*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p14,_p15,v)
        v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p14*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3817 do i=myis,myie
        _p21 = ((jnp.bool_(True)) & (v['_go']==0))
        _p22 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p23(_p24,v):
            v = dict(v)
            v['i'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((v['myis'])+_p24*(jnp.int32(1))), v['i'])
            # WRF:3818 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p21 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3819 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p21 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==315, 0, v['_go'])
            # WRF:3820 continue
            return v
        v = lax.fori_loop(0,_p22,_p23,v)
        v['i'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((v['myis'])+_p22*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3821 do k=2,l
        _p25 = ((jnp.bool_(True)) & (v['_go']==0))
        _p26 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p27(_p28,v):
            v = dict(v)
            v['k'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p28*(jnp.int32(1))), v['k'])
            # WRF:3822 do i=myis,myie
            _p29 = (((_p25 & (v['_go']==0))) & (v['_go']==0))
            _p30 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p31(_p32,v):
                v = dict(v)
                v['i'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((v['myis'])+_p32*(jnp.int32(1))), v['i'])
                # WRF:3823 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p29 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:3824 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p29 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==317, 0, v['_go'])
                # WRF:3825 continue
                return v
            v = lax.fori_loop(0,_p30,_p31,v)
            v['i'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((v['myis'])+_p30*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==319, 0, v['_go'])
            # WRF:3826 continue
            return v
        v = lax.fori_loop(0,_p26,_p27,v)
        v['k'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p26*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3828 do k=1,l
        _p33 = ((jnp.bool_(True)) & (v['_go']==0))
        _p34 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p35(_p36,v):
            v = dict(v)
            v['k'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p36*(jnp.int32(1))), v['k'])
            # WRF:3829 do i=myis,myie
            _p37 = (((_p33 & (v['_go']==0))) & (v['_go']==0))
            _p38 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p39(_p40,v):
                v = dict(v)
                v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p40*(jnp.int32(1))), v['i'])
                # WRF:3830 fac1(i,k)=acomb(1)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(1))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:3831 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(1)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(1))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:3832 tt(i,k)=exp(hm1ez*fac1(i,k)/sqrt(1.+fac2(i,k)))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp(((v['hm1ez'] * v['fac1'][(v['i'])-1, (v['k'])-1]) / jnp.sqrt((_r(1.) + v['fac2'][(v['i'])-1, (v['k'])-1]))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:3833 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==321, 0, v['_go'])
                # WRF:3834 continue
                return v
            v = lax.fori_loop(0,_p38,_p39,v)
            v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p38*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p34,_p35,v)
        v['k'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p34*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3836 do k=1,l
        _p41 = ((jnp.bool_(True)) & (v['_go']==0))
        _p42 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p43(_p44,v):
            v = dict(v)
            v['k'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p44*(jnp.int32(1))), v['k'])
            # WRF:3837 do i=myis,myie
            _p45 = (((_p41 & (v['_go']==0))) & (v['_go']==0))
            _p46 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p47(_p48,v):
                v = dict(v)
                v['i'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((v['myis'])+_p48*(jnp.int32(1))), v['i'])
                # WRF:3838 excts(i,k)=sorc(i,k,1)*(ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p45 & (v['_go']==0))) & (v['_go']==0)), _r((v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(1))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1]))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==353, 0, v['_go'])
                # WRF:3839 continue
                return v
            v = lax.fori_loop(0,_p46,_p47,v)
            v['i'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((v['myis'])+_p46*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p42,_p43,v)
        v['k'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p42*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3841 do i=myis,myie
        _p49 = ((jnp.bool_(True)) & (v['_go']==0))
        _p50 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p51(_p52,v):
            v = dict(v)
            v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p52*(jnp.int32(1))), v['i'])
            # WRF:3842 gxcts(i)=cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,1)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,1)-sorc(i,l,1)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(1))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(1))-1]))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==361, 0, v['_go'])
            # WRF:3846 continue
            return v
        v = lax.fori_loop(0,_p50,_p51,v)
        v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p50*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3855 do k=1,l
        _p53 = ((jnp.bool_(True)) & (v['_go']==0))
        _p54 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p55(_p56,v):
            v = dict(v)
            v['k'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p56*(jnp.int32(1))), v['k'])
            # WRF:3856 do i=myis,myie
            _p57 = (((_p53 & (v['_go']==0))) & (v['_go']==0))
            _p58 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p59(_p60,v):
                v = dict(v)
                v['i'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((v['myis'])+_p60*(jnp.int32(1))), v['i'])
                # WRF:3857 f(i,k)=h44194m2*(apcm(2)*x(i,k)+bpcm(2)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(2))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(2))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:3858 ff(i,k)=h44194m2*(atpcm(2)*x(i,k)+btpcm(2)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(2))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(2))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:3859 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:3860 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:3861 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:3862 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p57 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==401, 0, v['_go'])
                # WRF:3863 continue
                return v
            v = lax.fori_loop(0,_p58,_p59,v)
            v['i'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((v['myis'])+_p58*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p54,_p55,v)
        v['k'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p54*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3866 do i=myis,myie
        _p61 = ((jnp.bool_(True)) & (v['_go']==0))
        _p62 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p63(_p64,v):
            v = dict(v)
            v['i'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((v['myis'])+_p64*(jnp.int32(1))), v['i'])
            # WRF:3867 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3868 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==415, 0, v['_go'])
            # WRF:3869 continue
            return v
        v = lax.fori_loop(0,_p62,_p63,v)
        v['i'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((v['myis'])+_p62*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3870 do k=2,l
        _p65 = ((jnp.bool_(True)) & (v['_go']==0))
        _p66 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p67(_p68,v):
            v = dict(v)
            v['k'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p68*(jnp.int32(1))), v['k'])
            # WRF:3871 do i=myis,myie
            _p69 = (((_p65 & (v['_go']==0))) & (v['_go']==0))
            _p70 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p71(_p72,v):
                v = dict(v)
                v['i'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['myis'])+_p72*(jnp.int32(1))), v['i'])
                # WRF:3872 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p69 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:3873 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p69 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==417, 0, v['_go'])
                # WRF:3874 continue
                return v
            v = lax.fori_loop(0,_p70,_p71,v)
            v['i'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['myis'])+_p70*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==419, 0, v['_go'])
            # WRF:3875 continue
            return v
        v = lax.fori_loop(0,_p66,_p67,v)
        v['k'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p66*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3877 do k=1,l
        _p73 = ((jnp.bool_(True)) & (v['_go']==0))
        _p74 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p75(_p76,v):
            v = dict(v)
            v['k'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p76*(jnp.int32(1))), v['k'])
            # WRF:3878 do i=myis,myie
            _p77 = (((_p73 & (v['_go']==0))) & (v['_go']==0))
            _p78 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p79(_p80,v):
                v = dict(v)
                v['i'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((v['myis'])+_p80*(jnp.int32(1))), v['i'])
                # WRF:3879 fac1(i,k)=acomb(2)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p77 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(2))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:3880 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(2)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p77 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(2))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:3881 tt(i,k)=exp(hm1ez*fac1(i,k)/sqrt(1.+fac2(i,k)))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p77 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp(((v['hm1ez'] * v['fac1'][(v['i'])-1, (v['k'])-1]) / jnp.sqrt((_r(1.) + v['fac2'][(v['i'])-1, (v['k'])-1]))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:3882 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p77 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==421, 0, v['_go'])
                # WRF:3883 continue
                return v
            v = lax.fori_loop(0,_p78,_p79,v)
            v['i'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((v['myis'])+_p78*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p74,_p75,v)
        v['k'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p74*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3885 do k=1,l
        _p81 = ((jnp.bool_(True)) & (v['_go']==0))
        _p82 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p83(_p84,v):
            v = dict(v)
            v['k'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p84*(jnp.int32(1))), v['k'])
            # WRF:3886 do i=myis,myie
            _p85 = (((_p81 & (v['_go']==0))) & (v['_go']==0))
            _p86 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p87(_p88,v):
                v = dict(v)
                v['i'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((v['myis'])+_p88*(jnp.int32(1))), v['i'])
                # WRF:3887 excts(i,k)=excts(i,k)+sorc(i,k,2)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p85 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(2))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==453, 0, v['_go'])
                # WRF:3889 continue
                return v
            v = lax.fori_loop(0,_p86,_p87,v)
            v['i'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((v['myis'])+_p86*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p82,_p83,v)
        v['k'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p82*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3891 do i=myis,myie
        _p89 = ((jnp.bool_(True)) & (v['_go']==0))
        _p90 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p91(_p92,v):
            v = dict(v)
            v['i'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((v['myis'])+_p92*(jnp.int32(1))), v['i'])
            # WRF:3892 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,2)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,2)-sorc(i,l,2)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(2))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(2))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(2))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==461, 0, v['_go'])
            # WRF:3896 continue
            return v
        v = lax.fori_loop(0,_p90,_p91,v)
        v['i'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((v['myis'])+_p90*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3904 do k=1,l
        _p93 = ((jnp.bool_(True)) & (v['_go']==0))
        _p94 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p95(_p96,v):
            v = dict(v)
            v['k'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p96*(jnp.int32(1))), v['k'])
            # WRF:3905 do i=myis,myie
            _p97 = (((_p93 & (v['_go']==0))) & (v['_go']==0))
            _p98 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p99(_p100,v):
                v = dict(v)
                v['i'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((v['myis'])+_p100*(jnp.int32(1))), v['i'])
                # WRF:3906 f(i,k)=h44194m2*(apcm(3)*x(i,k)+bpcm(3)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(3))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(3))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:3907 ff(i,k)=h44194m2*(atpcm(3)*x(i,k)+btpcm(3)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(3))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(3))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:3908 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:3909 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:3910 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:3911 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==501, 0, v['_go'])
                # WRF:3912 continue
                return v
            v = lax.fori_loop(0,_p98,_p99,v)
            v['i'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((v['myis'])+_p98*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p94,_p95,v)
        v['k'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p94*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3915 do i=myis,myie
        _p101 = ((jnp.bool_(True)) & (v['_go']==0))
        _p102 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p103(_p104,v):
            v = dict(v)
            v['i'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((v['myis'])+_p104*(jnp.int32(1))), v['i'])
            # WRF:3916 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3917 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p101 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==515, 0, v['_go'])
            # WRF:3918 continue
            return v
        v = lax.fori_loop(0,_p102,_p103,v)
        v['i'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((v['myis'])+_p102*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3919 do k=2,l
        _p105 = ((jnp.bool_(True)) & (v['_go']==0))
        _p106 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p107(_p108,v):
            v = dict(v)
            v['k'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p108*(jnp.int32(1))), v['k'])
            # WRF:3920 do i=myis,myie
            _p109 = (((_p105 & (v['_go']==0))) & (v['_go']==0))
            _p110 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p111(_p112,v):
                v = dict(v)
                v['i'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((v['myis'])+_p112*(jnp.int32(1))), v['i'])
                # WRF:3921 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p109 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:3922 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p109 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==517, 0, v['_go'])
                # WRF:3923 continue
                return v
            v = lax.fori_loop(0,_p110,_p111,v)
            v['i'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((v['myis'])+_p110*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==519, 0, v['_go'])
            # WRF:3924 continue
            return v
        v = lax.fori_loop(0,_p106,_p107,v)
        v['k'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p106*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3926 do k=1,l
        _p113 = ((jnp.bool_(True)) & (v['_go']==0))
        _p114 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p115(_p116,v):
            v = dict(v)
            v['k'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p116*(jnp.int32(1))), v['k'])
            # WRF:3927 do i=myis,myie
            _p117 = (((_p113 & (v['_go']==0))) & (v['_go']==0))
            _p118 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p119(_p120,v):
                v = dict(v)
                v['i'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((v['myis'])+_p120*(jnp.int32(1))), v['i'])
                # WRF:3928 fac1(i,k)=acomb(3)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p117 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(3))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:3929 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(3)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p117 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(3))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:3930 tt(i,k)=exp(hm1ez*fac1(i,k)/sqrt(1.+fac2(i,k)))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p117 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp(((v['hm1ez'] * v['fac1'][(v['i'])-1, (v['k'])-1]) / jnp.sqrt((_r(1.) + v['fac2'][(v['i'])-1, (v['k'])-1]))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:3931 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p117 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==521, 0, v['_go'])
                # WRF:3932 continue
                return v
            v = lax.fori_loop(0,_p118,_p119,v)
            v['i'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((v['myis'])+_p118*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p114,_p115,v)
        v['k'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p114*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3934 do k=1,l
        _p121 = ((jnp.bool_(True)) & (v['_go']==0))
        _p122 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p123(_p124,v):
            v = dict(v)
            v['k'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p124*(jnp.int32(1))), v['k'])
            # WRF:3935 do i=myis,myie
            _p125 = (((_p121 & (v['_go']==0))) & (v['_go']==0))
            _p126 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p127(_p128,v):
                v = dict(v)
                v['i'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((v['myis'])+_p128*(jnp.int32(1))), v['i'])
                # WRF:3936 excts(i,k)=excts(i,k)+sorc(i,k,3)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p125 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(3))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==553, 0, v['_go'])
                # WRF:3938 continue
                return v
            v = lax.fori_loop(0,_p126,_p127,v)
            v['i'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((v['myis'])+_p126*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p122,_p123,v)
        v['k'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p122*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3940 do i=myis,myie
        _p129 = ((jnp.bool_(True)) & (v['_go']==0))
        _p130 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p131(_p132,v):
            v = dict(v)
            v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p132*(jnp.int32(1))), v['i'])
            # WRF:3941 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,3)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,3)-sorc(i,l,3)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(3))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(3))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(3))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==561, 0, v['_go'])
            # WRF:3945 continue
            return v
        v = lax.fori_loop(0,_p130,_p131,v)
        v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p130*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3953 do k=1,l
        _p133 = ((jnp.bool_(True)) & (v['_go']==0))
        _p134 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p135(_p136,v):
            v = dict(v)
            v['k'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p136*(jnp.int32(1))), v['k'])
            # WRF:3954 do i=myis,myie
            _p137 = (((_p133 & (v['_go']==0))) & (v['_go']==0))
            _p138 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p139(_p140,v):
                v = dict(v)
                v['i'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((v['myis'])+_p140*(jnp.int32(1))), v['i'])
                # WRF:3955 f(i,k)=h44194m2*(apcm(4)*x(i,k)+bpcm(4)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(4))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(4))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:3956 ff(i,k)=h44194m2*(atpcm(4)*x(i,k)+btpcm(4)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(4))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(4))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:3957 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:3958 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:3959 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:3960 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p137 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==601, 0, v['_go'])
                # WRF:3961 continue
                return v
            v = lax.fori_loop(0,_p138,_p139,v)
            v['i'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((v['myis'])+_p138*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p134,_p135,v)
        v['k'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p134*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3964 do i=myis,myie
        _p141 = ((jnp.bool_(True)) & (v['_go']==0))
        _p142 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p143(_p144,v):
            v = dict(v)
            v['i'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((v['myis'])+_p144*(jnp.int32(1))), v['i'])
            # WRF:3965 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p141 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:3966 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p141 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==615, 0, v['_go'])
            # WRF:3967 continue
            return v
        v = lax.fori_loop(0,_p142,_p143,v)
        v['i'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((v['myis'])+_p142*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3968 do k=2,l
        _p145 = ((jnp.bool_(True)) & (v['_go']==0))
        _p146 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p147(_p148,v):
            v = dict(v)
            v['k'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p148*(jnp.int32(1))), v['k'])
            # WRF:3969 do i=myis,myie
            _p149 = (((_p145 & (v['_go']==0))) & (v['_go']==0))
            _p150 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p151(_p152,v):
                v = dict(v)
                v['i'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((v['myis'])+_p152*(jnp.int32(1))), v['i'])
                # WRF:3970 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:3971 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p149 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==617, 0, v['_go'])
                # WRF:3972 continue
                return v
            v = lax.fori_loop(0,_p150,_p151,v)
            v['i'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((v['myis'])+_p150*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==619, 0, v['_go'])
            # WRF:3973 continue
            return v
        v = lax.fori_loop(0,_p146,_p147,v)
        v['k'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p146*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3975 do k=1,l
        _p153 = ((jnp.bool_(True)) & (v['_go']==0))
        _p154 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p155(_p156,v):
            v = dict(v)
            v['k'] = jnp.where((_p153 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p156*(jnp.int32(1))), v['k'])
            # WRF:3976 do i=myis,myie
            _p157 = (((_p153 & (v['_go']==0))) & (v['_go']==0))
            _p158 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p159(_p160,v):
                v = dict(v)
                v['i'] = jnp.where((_p157 & (v['_go']==0)), jnp.int32((v['myis'])+_p160*(jnp.int32(1))), v['i'])
                # WRF:3977 fac1(i,k)=acomb(4)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(4))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:3978 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(4)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(4))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:3979 tt(i,k)=exp(hm1ez*fac1(i,k)/sqrt(1.+fac2(i,k)))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp(((v['hm1ez'] * v['fac1'][(v['i'])-1, (v['k'])-1]) / jnp.sqrt((_r(1.) + v['fac2'][(v['i'])-1, (v['k'])-1]))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:3980 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p157 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==621, 0, v['_go'])
                # WRF:3981 continue
                return v
            v = lax.fori_loop(0,_p158,_p159,v)
            v['i'] = jnp.where((_p157 & (v['_go']==0)), jnp.int32((v['myis'])+_p158*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p154,_p155,v)
        v['k'] = jnp.where((_p153 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p154*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3983 do k=1,l
        _p161 = ((jnp.bool_(True)) & (v['_go']==0))
        _p162 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p163(_p164,v):
            v = dict(v)
            v['k'] = jnp.where((_p161 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p164*(jnp.int32(1))), v['k'])
            # WRF:3984 do i=myis,myie
            _p165 = (((_p161 & (v['_go']==0))) & (v['_go']==0))
            _p166 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p167(_p168,v):
                v = dict(v)
                v['i'] = jnp.where((_p165 & (v['_go']==0)), jnp.int32((v['myis'])+_p168*(jnp.int32(1))), v['i'])
                # WRF:3985 excts(i,k)=excts(i,k)+sorc(i,k,4)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p165 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(4))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==653, 0, v['_go'])
                # WRF:3987 continue
                return v
            v = lax.fori_loop(0,_p166,_p167,v)
            v['i'] = jnp.where((_p165 & (v['_go']==0)), jnp.int32((v['myis'])+_p166*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p162,_p163,v)
        v['k'] = jnp.where((_p161 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p162*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:3989 do i=myis,myie
        _p169 = ((jnp.bool_(True)) & (v['_go']==0))
        _p170 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p171(_p172,v):
            v = dict(v)
            v['i'] = jnp.where((_p169 & (v['_go']==0)), jnp.int32((v['myis'])+_p172*(jnp.int32(1))), v['i'])
            # WRF:3990 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,4)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,4)-sorc(i,l,4)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p169 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(4))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(4))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(4))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==661, 0, v['_go'])
            # WRF:3994 continue
            return v
        v = lax.fori_loop(0,_p170,_p171,v)
        v['i'] = jnp.where((_p169 & (v['_go']==0)), jnp.int32((v['myis'])+_p170*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4002 do k=1,l
        _p173 = ((jnp.bool_(True)) & (v['_go']==0))
        _p174 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p175(_p176,v):
            v = dict(v)
            v['k'] = jnp.where((_p173 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p176*(jnp.int32(1))), v['k'])
            # WRF:4003 do i=myis,myie
            _p177 = (((_p173 & (v['_go']==0))) & (v['_go']==0))
            _p178 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p179(_p180,v):
                v = dict(v)
                v['i'] = jnp.where((_p177 & (v['_go']==0)), jnp.int32((v['myis'])+_p180*(jnp.int32(1))), v['i'])
                # WRF:4004 f(i,k)=h44194m2*(apcm(5)*x(i,k)+bpcm(5)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(5))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(5))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4005 ff(i,k)=h44194m2*(atpcm(5)*x(i,k)+btpcm(5)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(5))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(5))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4006 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4007 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4008 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4009 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p177 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==701, 0, v['_go'])
                # WRF:4010 continue
                return v
            v = lax.fori_loop(0,_p178,_p179,v)
            v['i'] = jnp.where((_p177 & (v['_go']==0)), jnp.int32((v['myis'])+_p178*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p174,_p175,v)
        v['k'] = jnp.where((_p173 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p174*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4013 do i=myis,myie
        _p181 = ((jnp.bool_(True)) & (v['_go']==0))
        _p182 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p183(_p184,v):
            v = dict(v)
            v['i'] = jnp.where((_p181 & (v['_go']==0)), jnp.int32((v['myis'])+_p184*(jnp.int32(1))), v['i'])
            # WRF:4014 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p181 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4015 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p181 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==715, 0, v['_go'])
            # WRF:4016 continue
            return v
        v = lax.fori_loop(0,_p182,_p183,v)
        v['i'] = jnp.where((_p181 & (v['_go']==0)), jnp.int32((v['myis'])+_p182*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4017 do k=2,l
        _p185 = ((jnp.bool_(True)) & (v['_go']==0))
        _p186 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p187(_p188,v):
            v = dict(v)
            v['k'] = jnp.where((_p185 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p188*(jnp.int32(1))), v['k'])
            # WRF:4018 do i=myis,myie
            _p189 = (((_p185 & (v['_go']==0))) & (v['_go']==0))
            _p190 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p191(_p192,v):
                v = dict(v)
                v['i'] = jnp.where((_p189 & (v['_go']==0)), jnp.int32((v['myis'])+_p192*(jnp.int32(1))), v['i'])
                # WRF:4019 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p189 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4020 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p189 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==717, 0, v['_go'])
                # WRF:4021 continue
                return v
            v = lax.fori_loop(0,_p190,_p191,v)
            v['i'] = jnp.where((_p189 & (v['_go']==0)), jnp.int32((v['myis'])+_p190*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==719, 0, v['_go'])
            # WRF:4022 continue
            return v
        v = lax.fori_loop(0,_p186,_p187,v)
        v['k'] = jnp.where((_p185 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p186*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4024 do k=1,l
        _p193 = ((jnp.bool_(True)) & (v['_go']==0))
        _p194 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p195(_p196,v):
            v = dict(v)
            v['k'] = jnp.where((_p193 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p196*(jnp.int32(1))), v['k'])
            # WRF:4025 do i=myis,myie
            _p197 = (((_p193 & (v['_go']==0))) & (v['_go']==0))
            _p198 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p199(_p200,v):
                v = dict(v)
                v['i'] = jnp.where((_p197 & (v['_go']==0)), jnp.int32((v['myis'])+_p200*(jnp.int32(1))), v['i'])
                # WRF:4026 fac1(i,k)=acomb(5)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p197 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(5))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4027 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(5)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p197 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(5))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4028 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(5)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p197 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(5))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4030 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p197 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==721, 0, v['_go'])
                # WRF:4031 continue
                return v
            v = lax.fori_loop(0,_p198,_p199,v)
            v['i'] = jnp.where((_p197 & (v['_go']==0)), jnp.int32((v['myis'])+_p198*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p194,_p195,v)
        v['k'] = jnp.where((_p193 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p194*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4033 do k=1,l
        _p201 = ((jnp.bool_(True)) & (v['_go']==0))
        _p202 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p203(_p204,v):
            v = dict(v)
            v['k'] = jnp.where((_p201 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p204*(jnp.int32(1))), v['k'])
            # WRF:4034 do i=myis,myie
            _p205 = (((_p201 & (v['_go']==0))) & (v['_go']==0))
            _p206 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p207(_p208,v):
                v = dict(v)
                v['i'] = jnp.where((_p205 & (v['_go']==0)), jnp.int32((v['myis'])+_p208*(jnp.int32(1))), v['i'])
                # WRF:4035 excts(i,k)=excts(i,k)+sorc(i,k,5)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p205 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(5))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==753, 0, v['_go'])
                # WRF:4037 continue
                return v
            v = lax.fori_loop(0,_p206,_p207,v)
            v['i'] = jnp.where((_p205 & (v['_go']==0)), jnp.int32((v['myis'])+_p206*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p202,_p203,v)
        v['k'] = jnp.where((_p201 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p202*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4039 do i=myis,myie
        _p209 = ((jnp.bool_(True)) & (v['_go']==0))
        _p210 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p211(_p212,v):
            v = dict(v)
            v['i'] = jnp.where((_p209 & (v['_go']==0)), jnp.int32((v['myis'])+_p212*(jnp.int32(1))), v['i'])
            # WRF:4040 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,5)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,5)-sorc(i,l,5)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p209 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(5))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(5))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(5))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==761, 0, v['_go'])
            # WRF:4044 continue
            return v
        v = lax.fori_loop(0,_p210,_p211,v)
        v['i'] = jnp.where((_p209 & (v['_go']==0)), jnp.int32((v['myis'])+_p210*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4052 do k=1,l
        _p213 = ((jnp.bool_(True)) & (v['_go']==0))
        _p214 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p215(_p216,v):
            v = dict(v)
            v['k'] = jnp.where((_p213 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p216*(jnp.int32(1))), v['k'])
            # WRF:4053 do i=myis,myie
            _p217 = (((_p213 & (v['_go']==0))) & (v['_go']==0))
            _p218 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p219(_p220,v):
                v = dict(v)
                v['i'] = jnp.where((_p217 & (v['_go']==0)), jnp.int32((v['myis'])+_p220*(jnp.int32(1))), v['i'])
                # WRF:4054 f(i,k)=h44194m2*(apcm(6)*x(i,k)+bpcm(6)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(6))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(6))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4055 ff(i,k)=h44194m2*(atpcm(6)*x(i,k)+btpcm(6)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(6))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(6))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4056 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4057 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4058 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4059 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p217 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==801, 0, v['_go'])
                # WRF:4060 continue
                return v
            v = lax.fori_loop(0,_p218,_p219,v)
            v['i'] = jnp.where((_p217 & (v['_go']==0)), jnp.int32((v['myis'])+_p218*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p214,_p215,v)
        v['k'] = jnp.where((_p213 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p214*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4063 do i=myis,myie
        _p221 = ((jnp.bool_(True)) & (v['_go']==0))
        _p222 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p223(_p224,v):
            v = dict(v)
            v['i'] = jnp.where((_p221 & (v['_go']==0)), jnp.int32((v['myis'])+_p224*(jnp.int32(1))), v['i'])
            # WRF:4064 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p221 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4065 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p221 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==815, 0, v['_go'])
            # WRF:4066 continue
            return v
        v = lax.fori_loop(0,_p222,_p223,v)
        v['i'] = jnp.where((_p221 & (v['_go']==0)), jnp.int32((v['myis'])+_p222*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4067 do k=2,l
        _p225 = ((jnp.bool_(True)) & (v['_go']==0))
        _p226 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p227(_p228,v):
            v = dict(v)
            v['k'] = jnp.where((_p225 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p228*(jnp.int32(1))), v['k'])
            # WRF:4068 do i=myis,myie
            _p229 = (((_p225 & (v['_go']==0))) & (v['_go']==0))
            _p230 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p231(_p232,v):
                v = dict(v)
                v['i'] = jnp.where((_p229 & (v['_go']==0)), jnp.int32((v['myis'])+_p232*(jnp.int32(1))), v['i'])
                # WRF:4069 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p229 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4070 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p229 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==817, 0, v['_go'])
                # WRF:4071 continue
                return v
            v = lax.fori_loop(0,_p230,_p231,v)
            v['i'] = jnp.where((_p229 & (v['_go']==0)), jnp.int32((v['myis'])+_p230*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==819, 0, v['_go'])
            # WRF:4072 continue
            return v
        v = lax.fori_loop(0,_p226,_p227,v)
        v['k'] = jnp.where((_p225 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p226*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4074 do k=1,l
        _p233 = ((jnp.bool_(True)) & (v['_go']==0))
        _p234 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p235(_p236,v):
            v = dict(v)
            v['k'] = jnp.where((_p233 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p236*(jnp.int32(1))), v['k'])
            # WRF:4075 do i=myis,myie
            _p237 = (((_p233 & (v['_go']==0))) & (v['_go']==0))
            _p238 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p239(_p240,v):
                v = dict(v)
                v['i'] = jnp.where((_p237 & (v['_go']==0)), jnp.int32((v['myis'])+_p240*(jnp.int32(1))), v['i'])
                # WRF:4076 fac1(i,k)=acomb(6)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p237 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(6))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4077 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(6)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p237 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(6))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4078 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(6)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p237 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(6))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4080 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p237 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==821, 0, v['_go'])
                # WRF:4081 continue
                return v
            v = lax.fori_loop(0,_p238,_p239,v)
            v['i'] = jnp.where((_p237 & (v['_go']==0)), jnp.int32((v['myis'])+_p238*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p234,_p235,v)
        v['k'] = jnp.where((_p233 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p234*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4083 do k=1,l
        _p241 = ((jnp.bool_(True)) & (v['_go']==0))
        _p242 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p243(_p244,v):
            v = dict(v)
            v['k'] = jnp.where((_p241 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p244*(jnp.int32(1))), v['k'])
            # WRF:4084 do i=myis,myie
            _p245 = (((_p241 & (v['_go']==0))) & (v['_go']==0))
            _p246 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p247(_p248,v):
                v = dict(v)
                v['i'] = jnp.where((_p245 & (v['_go']==0)), jnp.int32((v['myis'])+_p248*(jnp.int32(1))), v['i'])
                # WRF:4085 excts(i,k)=excts(i,k)+sorc(i,k,6)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p245 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(6))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==853, 0, v['_go'])
                # WRF:4087 continue
                return v
            v = lax.fori_loop(0,_p246,_p247,v)
            v['i'] = jnp.where((_p245 & (v['_go']==0)), jnp.int32((v['myis'])+_p246*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p242,_p243,v)
        v['k'] = jnp.where((_p241 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p242*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4089 do i=myis,myie
        _p249 = ((jnp.bool_(True)) & (v['_go']==0))
        _p250 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p251(_p252,v):
            v = dict(v)
            v['i'] = jnp.where((_p249 & (v['_go']==0)), jnp.int32((v['myis'])+_p252*(jnp.int32(1))), v['i'])
            # WRF:4090 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,6)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,6)-sorc(i,l,6)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p249 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(6))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(6))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(6))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==861, 0, v['_go'])
            # WRF:4094 continue
            return v
        v = lax.fori_loop(0,_p250,_p251,v)
        v['i'] = jnp.where((_p249 & (v['_go']==0)), jnp.int32((v['myis'])+_p250*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4102 do k=1,l
        _p253 = ((jnp.bool_(True)) & (v['_go']==0))
        _p254 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p255(_p256,v):
            v = dict(v)
            v['k'] = jnp.where((_p253 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p256*(jnp.int32(1))), v['k'])
            # WRF:4103 do i=myis,myie
            _p257 = (((_p253 & (v['_go']==0))) & (v['_go']==0))
            _p258 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p259(_p260,v):
                v = dict(v)
                v['i'] = jnp.where((_p257 & (v['_go']==0)), jnp.int32((v['myis'])+_p260*(jnp.int32(1))), v['i'])
                # WRF:4104 f(i,k)=h44194m2*(apcm(7)*x(i,k)+bpcm(7)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(7))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(7))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4105 ff(i,k)=h44194m2*(atpcm(7)*x(i,k)+btpcm(7)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(7))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(7))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4106 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4107 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4108 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4109 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p257 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==901, 0, v['_go'])
                # WRF:4110 continue
                return v
            v = lax.fori_loop(0,_p258,_p259,v)
            v['i'] = jnp.where((_p257 & (v['_go']==0)), jnp.int32((v['myis'])+_p258*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p254,_p255,v)
        v['k'] = jnp.where((_p253 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p254*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4113 do i=myis,myie
        _p261 = ((jnp.bool_(True)) & (v['_go']==0))
        _p262 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p263(_p264,v):
            v = dict(v)
            v['i'] = jnp.where((_p261 & (v['_go']==0)), jnp.int32((v['myis'])+_p264*(jnp.int32(1))), v['i'])
            # WRF:4114 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p261 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4115 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p261 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==915, 0, v['_go'])
            # WRF:4116 continue
            return v
        v = lax.fori_loop(0,_p262,_p263,v)
        v['i'] = jnp.where((_p261 & (v['_go']==0)), jnp.int32((v['myis'])+_p262*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4117 do k=2,l
        _p265 = ((jnp.bool_(True)) & (v['_go']==0))
        _p266 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p267(_p268,v):
            v = dict(v)
            v['k'] = jnp.where((_p265 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p268*(jnp.int32(1))), v['k'])
            # WRF:4118 do i=myis,myie
            _p269 = (((_p265 & (v['_go']==0))) & (v['_go']==0))
            _p270 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p271(_p272,v):
                v = dict(v)
                v['i'] = jnp.where((_p269 & (v['_go']==0)), jnp.int32((v['myis'])+_p272*(jnp.int32(1))), v['i'])
                # WRF:4119 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p269 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4120 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p269 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==917, 0, v['_go'])
                # WRF:4121 continue
                return v
            v = lax.fori_loop(0,_p270,_p271,v)
            v['i'] = jnp.where((_p269 & (v['_go']==0)), jnp.int32((v['myis'])+_p270*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==919, 0, v['_go'])
            # WRF:4122 continue
            return v
        v = lax.fori_loop(0,_p266,_p267,v)
        v['k'] = jnp.where((_p265 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p266*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4124 do k=1,l
        _p273 = ((jnp.bool_(True)) & (v['_go']==0))
        _p274 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p275(_p276,v):
            v = dict(v)
            v['k'] = jnp.where((_p273 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p276*(jnp.int32(1))), v['k'])
            # WRF:4125 do i=myis,myie
            _p277 = (((_p273 & (v['_go']==0))) & (v['_go']==0))
            _p278 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p279(_p280,v):
                v = dict(v)
                v['i'] = jnp.where((_p277 & (v['_go']==0)), jnp.int32((v['myis'])+_p280*(jnp.int32(1))), v['i'])
                # WRF:4126 fac1(i,k)=acomb(7)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p277 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(7))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4127 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(7)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p277 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(7))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4128 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(7)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p277 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(7))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4130 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p277 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==921, 0, v['_go'])
                # WRF:4131 continue
                return v
            v = lax.fori_loop(0,_p278,_p279,v)
            v['i'] = jnp.where((_p277 & (v['_go']==0)), jnp.int32((v['myis'])+_p278*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p274,_p275,v)
        v['k'] = jnp.where((_p273 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p274*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4133 do k=1,l
        _p281 = ((jnp.bool_(True)) & (v['_go']==0))
        _p282 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p283(_p284,v):
            v = dict(v)
            v['k'] = jnp.where((_p281 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p284*(jnp.int32(1))), v['k'])
            # WRF:4134 do i=myis,myie
            _p285 = (((_p281 & (v['_go']==0))) & (v['_go']==0))
            _p286 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p287(_p288,v):
                v = dict(v)
                v['i'] = jnp.where((_p285 & (v['_go']==0)), jnp.int32((v['myis'])+_p288*(jnp.int32(1))), v['i'])
                # WRF:4135 excts(i,k)=excts(i,k)+sorc(i,k,7)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p285 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(7))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==953, 0, v['_go'])
                # WRF:4137 continue
                return v
            v = lax.fori_loop(0,_p286,_p287,v)
            v['i'] = jnp.where((_p285 & (v['_go']==0)), jnp.int32((v['myis'])+_p286*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p282,_p283,v)
        v['k'] = jnp.where((_p281 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p282*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4139 do i=myis,myie
        _p289 = ((jnp.bool_(True)) & (v['_go']==0))
        _p290 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p291(_p292,v):
            v = dict(v)
            v['i'] = jnp.where((_p289 & (v['_go']==0)), jnp.int32((v['myis'])+_p292*(jnp.int32(1))), v['i'])
            # WRF:4140 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,7)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,7)-sorc(i,l,7)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p289 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(7))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(7))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(7))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==961, 0, v['_go'])
            # WRF:4144 continue
            return v
        v = lax.fori_loop(0,_p290,_p291,v)
        v['i'] = jnp.where((_p289 & (v['_go']==0)), jnp.int32((v['myis'])+_p290*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4152 do k=1,l
        _p293 = ((jnp.bool_(True)) & (v['_go']==0))
        _p294 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p295(_p296,v):
            v = dict(v)
            v['k'] = jnp.where((_p293 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p296*(jnp.int32(1))), v['k'])
            # WRF:4153 do i=myis,myie
            _p297 = (((_p293 & (v['_go']==0))) & (v['_go']==0))
            _p298 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p299(_p300,v):
                v = dict(v)
                v['i'] = jnp.where((_p297 & (v['_go']==0)), jnp.int32((v['myis'])+_p300*(jnp.int32(1))), v['i'])
                # WRF:4154 f(i,k)=h44194m2*(apcm(8)*x(i,k)+bpcm(8)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p297 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(8))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(8))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4155 ff(i,k)=h44194m2*(atpcm(8)*x(i,k)+btpcm(8)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p297 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(8))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(8))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4156 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p297 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4157 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p297 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4158 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p297 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4159 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p297 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1001, 0, v['_go'])
                # WRF:4160 continue
                return v
            v = lax.fori_loop(0,_p298,_p299,v)
            v['i'] = jnp.where((_p297 & (v['_go']==0)), jnp.int32((v['myis'])+_p298*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p294,_p295,v)
        v['k'] = jnp.where((_p293 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p294*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4163 do i=myis,myie
        _p301 = ((jnp.bool_(True)) & (v['_go']==0))
        _p302 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p303(_p304,v):
            v = dict(v)
            v['i'] = jnp.where((_p301 & (v['_go']==0)), jnp.int32((v['myis'])+_p304*(jnp.int32(1))), v['i'])
            # WRF:4164 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p301 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4165 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p301 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1015, 0, v['_go'])
            # WRF:4166 continue
            return v
        v = lax.fori_loop(0,_p302,_p303,v)
        v['i'] = jnp.where((_p301 & (v['_go']==0)), jnp.int32((v['myis'])+_p302*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4167 do k=2,l
        _p305 = ((jnp.bool_(True)) & (v['_go']==0))
        _p306 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p307(_p308,v):
            v = dict(v)
            v['k'] = jnp.where((_p305 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p308*(jnp.int32(1))), v['k'])
            # WRF:4168 do i=myis,myie
            _p309 = (((_p305 & (v['_go']==0))) & (v['_go']==0))
            _p310 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p311(_p312,v):
                v = dict(v)
                v['i'] = jnp.where((_p309 & (v['_go']==0)), jnp.int32((v['myis'])+_p312*(jnp.int32(1))), v['i'])
                # WRF:4169 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p309 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4170 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p309 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1017, 0, v['_go'])
                # WRF:4171 continue
                return v
            v = lax.fori_loop(0,_p310,_p311,v)
            v['i'] = jnp.where((_p309 & (v['_go']==0)), jnp.int32((v['myis'])+_p310*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1019, 0, v['_go'])
            # WRF:4172 continue
            return v
        v = lax.fori_loop(0,_p306,_p307,v)
        v['k'] = jnp.where((_p305 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p306*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4174 do k=1,l
        _p313 = ((jnp.bool_(True)) & (v['_go']==0))
        _p314 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p315(_p316,v):
            v = dict(v)
            v['k'] = jnp.where((_p313 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p316*(jnp.int32(1))), v['k'])
            # WRF:4175 do i=myis,myie
            _p317 = (((_p313 & (v['_go']==0))) & (v['_go']==0))
            _p318 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p319(_p320,v):
                v = dict(v)
                v['i'] = jnp.where((_p317 & (v['_go']==0)), jnp.int32((v['myis'])+_p320*(jnp.int32(1))), v['i'])
                # WRF:4176 fac1(i,k)=acomb(8)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p317 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(8))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4177 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(8)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p317 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(8))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4178 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(8)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p317 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(8))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4180 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p317 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1021, 0, v['_go'])
                # WRF:4181 continue
                return v
            v = lax.fori_loop(0,_p318,_p319,v)
            v['i'] = jnp.where((_p317 & (v['_go']==0)), jnp.int32((v['myis'])+_p318*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p314,_p315,v)
        v['k'] = jnp.where((_p313 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p314*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4183 do k=1,l
        _p321 = ((jnp.bool_(True)) & (v['_go']==0))
        _p322 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p323(_p324,v):
            v = dict(v)
            v['k'] = jnp.where((_p321 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p324*(jnp.int32(1))), v['k'])
            # WRF:4184 do i=myis,myie
            _p325 = (((_p321 & (v['_go']==0))) & (v['_go']==0))
            _p326 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p327(_p328,v):
                v = dict(v)
                v['i'] = jnp.where((_p325 & (v['_go']==0)), jnp.int32((v['myis'])+_p328*(jnp.int32(1))), v['i'])
                # WRF:4185 excts(i,k)=excts(i,k)+sorc(i,k,8)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p325 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(8))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1053, 0, v['_go'])
                # WRF:4187 continue
                return v
            v = lax.fori_loop(0,_p326,_p327,v)
            v['i'] = jnp.where((_p325 & (v['_go']==0)), jnp.int32((v['myis'])+_p326*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p322,_p323,v)
        v['k'] = jnp.where((_p321 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p322*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4189 do i=myis,myie
        _p329 = ((jnp.bool_(True)) & (v['_go']==0))
        _p330 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p331(_p332,v):
            v = dict(v)
            v['i'] = jnp.where((_p329 & (v['_go']==0)), jnp.int32((v['myis'])+_p332*(jnp.int32(1))), v['i'])
            # WRF:4190 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,8)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,8)-sorc(i,l,8)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p329 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(8))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(8))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(8))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1061, 0, v['_go'])
            # WRF:4194 continue
            return v
        v = lax.fori_loop(0,_p330,_p331,v)
        v['i'] = jnp.where((_p329 & (v['_go']==0)), jnp.int32((v['myis'])+_p330*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4202 do k=1,l
        _p333 = ((jnp.bool_(True)) & (v['_go']==0))
        _p334 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p335(_p336,v):
            v = dict(v)
            v['k'] = jnp.where((_p333 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p336*(jnp.int32(1))), v['k'])
            # WRF:4203 do i=myis,myie
            _p337 = (((_p333 & (v['_go']==0))) & (v['_go']==0))
            _p338 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p339(_p340,v):
                v = dict(v)
                v['i'] = jnp.where((_p337 & (v['_go']==0)), jnp.int32((v['myis'])+_p340*(jnp.int32(1))), v['i'])
                # WRF:4204 f(i,k)=h44194m2*(apcm(9)*x(i,k)+bpcm(9)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p337 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(9))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(9))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4205 ff(i,k)=h44194m2*(atpcm(9)*x(i,k)+btpcm(9)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p337 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(9))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(9))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4206 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p337 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4207 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p337 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4208 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p337 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4209 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p337 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1101, 0, v['_go'])
                # WRF:4210 continue
                return v
            v = lax.fori_loop(0,_p338,_p339,v)
            v['i'] = jnp.where((_p337 & (v['_go']==0)), jnp.int32((v['myis'])+_p338*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p334,_p335,v)
        v['k'] = jnp.where((_p333 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p334*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4213 do i=myis,myie
        _p341 = ((jnp.bool_(True)) & (v['_go']==0))
        _p342 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p343(_p344,v):
            v = dict(v)
            v['i'] = jnp.where((_p341 & (v['_go']==0)), jnp.int32((v['myis'])+_p344*(jnp.int32(1))), v['i'])
            # WRF:4214 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p341 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4215 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p341 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1115, 0, v['_go'])
            # WRF:4216 continue
            return v
        v = lax.fori_loop(0,_p342,_p343,v)
        v['i'] = jnp.where((_p341 & (v['_go']==0)), jnp.int32((v['myis'])+_p342*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4217 do k=2,l
        _p345 = ((jnp.bool_(True)) & (v['_go']==0))
        _p346 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p347(_p348,v):
            v = dict(v)
            v['k'] = jnp.where((_p345 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p348*(jnp.int32(1))), v['k'])
            # WRF:4218 do i=myis,myie
            _p349 = (((_p345 & (v['_go']==0))) & (v['_go']==0))
            _p350 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p351(_p352,v):
                v = dict(v)
                v['i'] = jnp.where((_p349 & (v['_go']==0)), jnp.int32((v['myis'])+_p352*(jnp.int32(1))), v['i'])
                # WRF:4219 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p349 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4220 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p349 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1117, 0, v['_go'])
                # WRF:4221 continue
                return v
            v = lax.fori_loop(0,_p350,_p351,v)
            v['i'] = jnp.where((_p349 & (v['_go']==0)), jnp.int32((v['myis'])+_p350*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1119, 0, v['_go'])
            # WRF:4222 continue
            return v
        v = lax.fori_loop(0,_p346,_p347,v)
        v['k'] = jnp.where((_p345 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p346*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4224 do k=1,l
        _p353 = ((jnp.bool_(True)) & (v['_go']==0))
        _p354 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p355(_p356,v):
            v = dict(v)
            v['k'] = jnp.where((_p353 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p356*(jnp.int32(1))), v['k'])
            # WRF:4225 do i=myis,myie
            _p357 = (((_p353 & (v['_go']==0))) & (v['_go']==0))
            _p358 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p359(_p360,v):
                v = dict(v)
                v['i'] = jnp.where((_p357 & (v['_go']==0)), jnp.int32((v['myis'])+_p360*(jnp.int32(1))), v['i'])
                # WRF:4226 fac1(i,k)=acomb(9)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p357 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(9))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4227 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(9)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p357 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(9))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4228 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(9)*totvo2(i,k+1)*sko2d))*co2sp1(i,k+1)
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p357 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(9))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d']))))) * v['co2sp1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4230 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p357 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1121, 0, v['_go'])
                # WRF:4231 continue
                return v
            v = lax.fori_loop(0,_p358,_p359,v)
            v['i'] = jnp.where((_p357 & (v['_go']==0)), jnp.int32((v['myis'])+_p358*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p354,_p355,v)
        v['k'] = jnp.where((_p353 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p354*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4233 do k=1,l
        _p361 = ((jnp.bool_(True)) & (v['_go']==0))
        _p362 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p363(_p364,v):
            v = dict(v)
            v['k'] = jnp.where((_p361 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p364*(jnp.int32(1))), v['k'])
            # WRF:4234 do i=myis,myie
            _p365 = (((_p361 & (v['_go']==0))) & (v['_go']==0))
            _p366 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p367(_p368,v):
                v = dict(v)
                v['i'] = jnp.where((_p365 & (v['_go']==0)), jnp.int32((v['myis'])+_p368*(jnp.int32(1))), v['i'])
                # WRF:4235 excts(i,k)=excts(i,k)+sorc(i,k,9)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p365 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(9))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1153, 0, v['_go'])
                # WRF:4237 continue
                return v
            v = lax.fori_loop(0,_p366,_p367,v)
            v['i'] = jnp.where((_p365 & (v['_go']==0)), jnp.int32((v['myis'])+_p366*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p362,_p363,v)
        v['k'] = jnp.where((_p361 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p362*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4239 do i=myis,myie
        _p369 = ((jnp.bool_(True)) & (v['_go']==0))
        _p370 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p371(_p372,v):
            v = dict(v)
            v['i'] = jnp.where((_p369 & (v['_go']==0)), jnp.int32((v['myis'])+_p372*(jnp.int32(1))), v['i'])
            # WRF:4240 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,9)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,9)-sorc(i,l,9)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p369 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(9))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(9))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(9))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1161, 0, v['_go'])
            # WRF:4244 continue
            return v
        v = lax.fori_loop(0,_p370,_p371,v)
        v['i'] = jnp.where((_p369 & (v['_go']==0)), jnp.int32((v['myis'])+_p370*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4252 do k=1,l
        _p373 = ((jnp.bool_(True)) & (v['_go']==0))
        _p374 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p375(_p376,v):
            v = dict(v)
            v['k'] = jnp.where((_p373 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p376*(jnp.int32(1))), v['k'])
            # WRF:4253 do i=myis,myie
            _p377 = (((_p373 & (v['_go']==0))) & (v['_go']==0))
            _p378 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p379(_p380,v):
                v = dict(v)
                v['i'] = jnp.where((_p377 & (v['_go']==0)), jnp.int32((v['myis'])+_p380*(jnp.int32(1))), v['i'])
                # WRF:4254 f(i,k)=h44194m2*(apcm(10)*x(i,k)+bpcm(10)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p377 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(10))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(10))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4255 ff(i,k)=h44194m2*(atpcm(10)*x(i,k)+btpcm(10)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p377 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(10))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(10))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4256 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p377 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4257 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p377 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4258 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p377 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4259 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p377 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1201, 0, v['_go'])
                # WRF:4260 continue
                return v
            v = lax.fori_loop(0,_p378,_p379,v)
            v['i'] = jnp.where((_p377 & (v['_go']==0)), jnp.int32((v['myis'])+_p378*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p374,_p375,v)
        v['k'] = jnp.where((_p373 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p374*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4263 do i=myis,myie
        _p381 = ((jnp.bool_(True)) & (v['_go']==0))
        _p382 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p383(_p384,v):
            v = dict(v)
            v['i'] = jnp.where((_p381 & (v['_go']==0)), jnp.int32((v['myis'])+_p384*(jnp.int32(1))), v['i'])
            # WRF:4264 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p381 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4265 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p381 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1215, 0, v['_go'])
            # WRF:4266 continue
            return v
        v = lax.fori_loop(0,_p382,_p383,v)
        v['i'] = jnp.where((_p381 & (v['_go']==0)), jnp.int32((v['myis'])+_p382*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4267 do k=2,l
        _p385 = ((jnp.bool_(True)) & (v['_go']==0))
        _p386 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p387(_p388,v):
            v = dict(v)
            v['k'] = jnp.where((_p385 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p388*(jnp.int32(1))), v['k'])
            # WRF:4268 do i=myis,myie
            _p389 = (((_p385 & (v['_go']==0))) & (v['_go']==0))
            _p390 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p391(_p392,v):
                v = dict(v)
                v['i'] = jnp.where((_p389 & (v['_go']==0)), jnp.int32((v['myis'])+_p392*(jnp.int32(1))), v['i'])
                # WRF:4269 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p389 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4270 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p389 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1217, 0, v['_go'])
                # WRF:4271 continue
                return v
            v = lax.fori_loop(0,_p390,_p391,v)
            v['i'] = jnp.where((_p389 & (v['_go']==0)), jnp.int32((v['myis'])+_p390*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1219, 0, v['_go'])
            # WRF:4272 continue
            return v
        v = lax.fori_loop(0,_p386,_p387,v)
        v['k'] = jnp.where((_p385 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p386*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4274 do k=1,l
        _p393 = ((jnp.bool_(True)) & (v['_go']==0))
        _p394 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p395(_p396,v):
            v = dict(v)
            v['k'] = jnp.where((_p393 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p396*(jnp.int32(1))), v['k'])
            # WRF:4275 do i=myis,myie
            _p397 = (((_p393 & (v['_go']==0))) & (v['_go']==0))
            _p398 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p399(_p400,v):
                v = dict(v)
                v['i'] = jnp.where((_p397 & (v['_go']==0)), jnp.int32((v['myis'])+_p400*(jnp.int32(1))), v['i'])
                # WRF:4276 fac1(i,k)=acomb(10)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p397 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(10))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4277 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(10)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p397 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(10))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4278 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(10)*totvo2(i,k+1)*sko2d))*co2sp2(i,k+1)
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p397 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(10))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d']))))) * v['co2sp2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4280 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p397 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1221, 0, v['_go'])
                # WRF:4281 continue
                return v
            v = lax.fori_loop(0,_p398,_p399,v)
            v['i'] = jnp.where((_p397 & (v['_go']==0)), jnp.int32((v['myis'])+_p398*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p394,_p395,v)
        v['k'] = jnp.where((_p393 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p394*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4283 do k=1,l
        _p401 = ((jnp.bool_(True)) & (v['_go']==0))
        _p402 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p403(_p404,v):
            v = dict(v)
            v['k'] = jnp.where((_p401 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p404*(jnp.int32(1))), v['k'])
            # WRF:4284 do i=myis,myie
            _p405 = (((_p401 & (v['_go']==0))) & (v['_go']==0))
            _p406 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p407(_p408,v):
                v = dict(v)
                v['i'] = jnp.where((_p405 & (v['_go']==0)), jnp.int32((v['myis'])+_p408*(jnp.int32(1))), v['i'])
                # WRF:4285 excts(i,k)=excts(i,k)+sorc(i,k,10)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p405 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(10))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1253, 0, v['_go'])
                # WRF:4287 continue
                return v
            v = lax.fori_loop(0,_p406,_p407,v)
            v['i'] = jnp.where((_p405 & (v['_go']==0)), jnp.int32((v['myis'])+_p406*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p402,_p403,v)
        v['k'] = jnp.where((_p401 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p402*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4289 do i=myis,myie
        _p409 = ((jnp.bool_(True)) & (v['_go']==0))
        _p410 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p411(_p412,v):
            v = dict(v)
            v['i'] = jnp.where((_p409 & (v['_go']==0)), jnp.int32((v['myis'])+_p412*(jnp.int32(1))), v['i'])
            # WRF:4290 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,10)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,10)-sorc(i,l,10)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p409 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(10))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(10))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(10))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1261, 0, v['_go'])
            # WRF:4294 continue
            return v
        v = lax.fori_loop(0,_p410,_p411,v)
        v['i'] = jnp.where((_p409 & (v['_go']==0)), jnp.int32((v['myis'])+_p410*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4302 do k=1,l
        _p413 = ((jnp.bool_(True)) & (v['_go']==0))
        _p414 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p415(_p416,v):
            v = dict(v)
            v['k'] = jnp.where((_p413 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p416*(jnp.int32(1))), v['k'])
            # WRF:4303 do i=myis,myie
            _p417 = (((_p413 & (v['_go']==0))) & (v['_go']==0))
            _p418 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p419(_p420,v):
                v = dict(v)
                v['i'] = jnp.where((_p417 & (v['_go']==0)), jnp.int32((v['myis'])+_p420*(jnp.int32(1))), v['i'])
                # WRF:4304 f(i,k)=h44194m2*(apcm(11)*x(i,k)+bpcm(11)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p417 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(11))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(11))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4305 ff(i,k)=h44194m2*(atpcm(11)*x(i,k)+btpcm(11)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p417 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(11))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(11))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4306 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p417 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4307 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p417 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4308 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p417 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4309 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p417 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1301, 0, v['_go'])
                # WRF:4310 continue
                return v
            v = lax.fori_loop(0,_p418,_p419,v)
            v['i'] = jnp.where((_p417 & (v['_go']==0)), jnp.int32((v['myis'])+_p418*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p414,_p415,v)
        v['k'] = jnp.where((_p413 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p414*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4313 do i=myis,myie
        _p421 = ((jnp.bool_(True)) & (v['_go']==0))
        _p422 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p423(_p424,v):
            v = dict(v)
            v['i'] = jnp.where((_p421 & (v['_go']==0)), jnp.int32((v['myis'])+_p424*(jnp.int32(1))), v['i'])
            # WRF:4314 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p421 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4315 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p421 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1315, 0, v['_go'])
            # WRF:4316 continue
            return v
        v = lax.fori_loop(0,_p422,_p423,v)
        v['i'] = jnp.where((_p421 & (v['_go']==0)), jnp.int32((v['myis'])+_p422*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4317 do k=2,l
        _p425 = ((jnp.bool_(True)) & (v['_go']==0))
        _p426 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p427(_p428,v):
            v = dict(v)
            v['k'] = jnp.where((_p425 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p428*(jnp.int32(1))), v['k'])
            # WRF:4318 do i=myis,myie
            _p429 = (((_p425 & (v['_go']==0))) & (v['_go']==0))
            _p430 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p431(_p432,v):
                v = dict(v)
                v['i'] = jnp.where((_p429 & (v['_go']==0)), jnp.int32((v['myis'])+_p432*(jnp.int32(1))), v['i'])
                # WRF:4319 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p429 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4320 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p429 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1317, 0, v['_go'])
                # WRF:4321 continue
                return v
            v = lax.fori_loop(0,_p430,_p431,v)
            v['i'] = jnp.where((_p429 & (v['_go']==0)), jnp.int32((v['myis'])+_p430*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1319, 0, v['_go'])
            # WRF:4322 continue
            return v
        v = lax.fori_loop(0,_p426,_p427,v)
        v['k'] = jnp.where((_p425 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p426*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4324 do k=1,l
        _p433 = ((jnp.bool_(True)) & (v['_go']==0))
        _p434 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p435(_p436,v):
            v = dict(v)
            v['k'] = jnp.where((_p433 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p436*(jnp.int32(1))), v['k'])
            # WRF:4325 do i=myis,myie
            _p437 = (((_p433 & (v['_go']==0))) & (v['_go']==0))
            _p438 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p439(_p440,v):
                v = dict(v)
                v['i'] = jnp.where((_p437 & (v['_go']==0)), jnp.int32((v['myis'])+_p440*(jnp.int32(1))), v['i'])
                # WRF:4326 fac1(i,k)=acomb(11)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p437 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(11))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4327 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(11)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p437 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(11))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4328 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(11)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p437 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(11))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4330 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p437 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1321, 0, v['_go'])
                # WRF:4331 continue
                return v
            v = lax.fori_loop(0,_p438,_p439,v)
            v['i'] = jnp.where((_p437 & (v['_go']==0)), jnp.int32((v['myis'])+_p438*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p434,_p435,v)
        v['k'] = jnp.where((_p433 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p434*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4333 do k=1,l
        _p441 = ((jnp.bool_(True)) & (v['_go']==0))
        _p442 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p443(_p444,v):
            v = dict(v)
            v['k'] = jnp.where((_p441 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p444*(jnp.int32(1))), v['k'])
            # WRF:4334 do i=myis,myie
            _p445 = (((_p441 & (v['_go']==0))) & (v['_go']==0))
            _p446 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p447(_p448,v):
                v = dict(v)
                v['i'] = jnp.where((_p445 & (v['_go']==0)), jnp.int32((v['myis'])+_p448*(jnp.int32(1))), v['i'])
                # WRF:4335 excts(i,k)=excts(i,k)+sorc(i,k,11)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p445 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(11))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1353, 0, v['_go'])
                # WRF:4337 continue
                return v
            v = lax.fori_loop(0,_p446,_p447,v)
            v['i'] = jnp.where((_p445 & (v['_go']==0)), jnp.int32((v['myis'])+_p446*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p442,_p443,v)
        v['k'] = jnp.where((_p441 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p442*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4339 do i=myis,myie
        _p449 = ((jnp.bool_(True)) & (v['_go']==0))
        _p450 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p451(_p452,v):
            v = dict(v)
            v['i'] = jnp.where((_p449 & (v['_go']==0)), jnp.int32((v['myis'])+_p452*(jnp.int32(1))), v['i'])
            # WRF:4340 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,11)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,11)-sorc(i,l,11)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p449 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(11))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(11))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(11))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1361, 0, v['_go'])
            # WRF:4344 continue
            return v
        v = lax.fori_loop(0,_p450,_p451,v)
        v['i'] = jnp.where((_p449 & (v['_go']==0)), jnp.int32((v['myis'])+_p450*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4352 do k=1,l
        _p453 = ((jnp.bool_(True)) & (v['_go']==0))
        _p454 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p455(_p456,v):
            v = dict(v)
            v['k'] = jnp.where((_p453 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p456*(jnp.int32(1))), v['k'])
            # WRF:4353 do i=myis,myie
            _p457 = (((_p453 & (v['_go']==0))) & (v['_go']==0))
            _p458 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p459(_p460,v):
                v = dict(v)
                v['i'] = jnp.where((_p457 & (v['_go']==0)), jnp.int32((v['myis'])+_p460*(jnp.int32(1))), v['i'])
                # WRF:4354 f(i,k)=h44194m2*(apcm(12)*x(i,k)+bpcm(12)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(12))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(12))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4355 ff(i,k)=h44194m2*(atpcm(12)*x(i,k)+btpcm(12)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(12))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(12))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4356 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4357 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4358 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4359 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1401, 0, v['_go'])
                # WRF:4360 continue
                return v
            v = lax.fori_loop(0,_p458,_p459,v)
            v['i'] = jnp.where((_p457 & (v['_go']==0)), jnp.int32((v['myis'])+_p458*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p454,_p455,v)
        v['k'] = jnp.where((_p453 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p454*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4363 do i=myis,myie
        _p461 = ((jnp.bool_(True)) & (v['_go']==0))
        _p462 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p463(_p464,v):
            v = dict(v)
            v['i'] = jnp.where((_p461 & (v['_go']==0)), jnp.int32((v['myis'])+_p464*(jnp.int32(1))), v['i'])
            # WRF:4364 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p461 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4365 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p461 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1415, 0, v['_go'])
            # WRF:4366 continue
            return v
        v = lax.fori_loop(0,_p462,_p463,v)
        v['i'] = jnp.where((_p461 & (v['_go']==0)), jnp.int32((v['myis'])+_p462*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4367 do k=2,l
        _p465 = ((jnp.bool_(True)) & (v['_go']==0))
        _p466 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p467(_p468,v):
            v = dict(v)
            v['k'] = jnp.where((_p465 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p468*(jnp.int32(1))), v['k'])
            # WRF:4368 do i=myis,myie
            _p469 = (((_p465 & (v['_go']==0))) & (v['_go']==0))
            _p470 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p471(_p472,v):
                v = dict(v)
                v['i'] = jnp.where((_p469 & (v['_go']==0)), jnp.int32((v['myis'])+_p472*(jnp.int32(1))), v['i'])
                # WRF:4369 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p469 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4370 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p469 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1417, 0, v['_go'])
                # WRF:4371 continue
                return v
            v = lax.fori_loop(0,_p470,_p471,v)
            v['i'] = jnp.where((_p469 & (v['_go']==0)), jnp.int32((v['myis'])+_p470*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1419, 0, v['_go'])
            # WRF:4372 continue
            return v
        v = lax.fori_loop(0,_p466,_p467,v)
        v['k'] = jnp.where((_p465 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p466*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4374 do k=1,l
        _p473 = ((jnp.bool_(True)) & (v['_go']==0))
        _p474 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p475(_p476,v):
            v = dict(v)
            v['k'] = jnp.where((_p473 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p476*(jnp.int32(1))), v['k'])
            # WRF:4375 do i=myis,myie
            _p477 = (((_p473 & (v['_go']==0))) & (v['_go']==0))
            _p478 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p479(_p480,v):
                v = dict(v)
                v['i'] = jnp.where((_p477 & (v['_go']==0)), jnp.int32((v['myis'])+_p480*(jnp.int32(1))), v['i'])
                # WRF:4376 fac1(i,k)=acomb(12)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p477 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(12))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4377 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(12)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p477 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(12))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4378 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(12)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p477 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(12))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4380 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p477 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1421, 0, v['_go'])
                # WRF:4381 continue
                return v
            v = lax.fori_loop(0,_p478,_p479,v)
            v['i'] = jnp.where((_p477 & (v['_go']==0)), jnp.int32((v['myis'])+_p478*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p474,_p475,v)
        v['k'] = jnp.where((_p473 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p474*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4383 do k=1,l
        _p481 = ((jnp.bool_(True)) & (v['_go']==0))
        _p482 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p483(_p484,v):
            v = dict(v)
            v['k'] = jnp.where((_p481 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p484*(jnp.int32(1))), v['k'])
            # WRF:4384 do i=myis,myie
            _p485 = (((_p481 & (v['_go']==0))) & (v['_go']==0))
            _p486 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p487(_p488,v):
                v = dict(v)
                v['i'] = jnp.where((_p485 & (v['_go']==0)), jnp.int32((v['myis'])+_p488*(jnp.int32(1))), v['i'])
                # WRF:4385 excts(i,k)=excts(i,k)+sorc(i,k,12)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p485 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(12))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1453, 0, v['_go'])
                # WRF:4387 continue
                return v
            v = lax.fori_loop(0,_p486,_p487,v)
            v['i'] = jnp.where((_p485 & (v['_go']==0)), jnp.int32((v['myis'])+_p486*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p482,_p483,v)
        v['k'] = jnp.where((_p481 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p482*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4389 do i=myis,myie
        _p489 = ((jnp.bool_(True)) & (v['_go']==0))
        _p490 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p491(_p492,v):
            v = dict(v)
            v['i'] = jnp.where((_p489 & (v['_go']==0)), jnp.int32((v['myis'])+_p492*(jnp.int32(1))), v['i'])
            # WRF:4390 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,12)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,12)-sorc(i,l,12)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p489 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(12))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(12))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(12))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1461, 0, v['_go'])
            # WRF:4394 continue
            return v
        v = lax.fori_loop(0,_p490,_p491,v)
        v['i'] = jnp.where((_p489 & (v['_go']==0)), jnp.int32((v['myis'])+_p490*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4402 do k=1,l
        _p493 = ((jnp.bool_(True)) & (v['_go']==0))
        _p494 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p495(_p496,v):
            v = dict(v)
            v['k'] = jnp.where((_p493 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p496*(jnp.int32(1))), v['k'])
            # WRF:4403 do i=myis,myie
            _p497 = (((_p493 & (v['_go']==0))) & (v['_go']==0))
            _p498 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p499(_p500,v):
                v = dict(v)
                v['i'] = jnp.where((_p497 & (v['_go']==0)), jnp.int32((v['myis'])+_p500*(jnp.int32(1))), v['i'])
                # WRF:4404 f(i,k)=h44194m2*(apcm(13)*x(i,k)+bpcm(13)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p497 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(13))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(13))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4405 ff(i,k)=h44194m2*(atpcm(13)*x(i,k)+btpcm(13)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p497 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(13))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(13))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4406 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p497 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4407 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p497 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4408 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p497 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4409 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p497 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1501, 0, v['_go'])
                # WRF:4410 continue
                return v
            v = lax.fori_loop(0,_p498,_p499,v)
            v['i'] = jnp.where((_p497 & (v['_go']==0)), jnp.int32((v['myis'])+_p498*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p494,_p495,v)
        v['k'] = jnp.where((_p493 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p494*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4413 do i=myis,myie
        _p501 = ((jnp.bool_(True)) & (v['_go']==0))
        _p502 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p503(_p504,v):
            v = dict(v)
            v['i'] = jnp.where((_p501 & (v['_go']==0)), jnp.int32((v['myis'])+_p504*(jnp.int32(1))), v['i'])
            # WRF:4414 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p501 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4415 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p501 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1515, 0, v['_go'])
            # WRF:4416 continue
            return v
        v = lax.fori_loop(0,_p502,_p503,v)
        v['i'] = jnp.where((_p501 & (v['_go']==0)), jnp.int32((v['myis'])+_p502*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4417 do k=2,l
        _p505 = ((jnp.bool_(True)) & (v['_go']==0))
        _p506 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p507(_p508,v):
            v = dict(v)
            v['k'] = jnp.where((_p505 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p508*(jnp.int32(1))), v['k'])
            # WRF:4418 do i=myis,myie
            _p509 = (((_p505 & (v['_go']==0))) & (v['_go']==0))
            _p510 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p511(_p512,v):
                v = dict(v)
                v['i'] = jnp.where((_p509 & (v['_go']==0)), jnp.int32((v['myis'])+_p512*(jnp.int32(1))), v['i'])
                # WRF:4419 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p509 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4420 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p509 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1517, 0, v['_go'])
                # WRF:4421 continue
                return v
            v = lax.fori_loop(0,_p510,_p511,v)
            v['i'] = jnp.where((_p509 & (v['_go']==0)), jnp.int32((v['myis'])+_p510*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1519, 0, v['_go'])
            # WRF:4422 continue
            return v
        v = lax.fori_loop(0,_p506,_p507,v)
        v['k'] = jnp.where((_p505 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p506*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4424 do k=1,l
        _p513 = ((jnp.bool_(True)) & (v['_go']==0))
        _p514 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p515(_p516,v):
            v = dict(v)
            v['k'] = jnp.where((_p513 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p516*(jnp.int32(1))), v['k'])
            # WRF:4425 do i=myis,myie
            _p517 = (((_p513 & (v['_go']==0))) & (v['_go']==0))
            _p518 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p519(_p520,v):
                v = dict(v)
                v['i'] = jnp.where((_p517 & (v['_go']==0)), jnp.int32((v['myis'])+_p520*(jnp.int32(1))), v['i'])
                # WRF:4426 fac1(i,k)=acomb(13)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p517 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(13))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4427 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(13)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p517 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(13))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4428 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(13)*totvo2(i,k+1)*sko2d+to3spc(i,k)))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p517 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * (((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(13))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d']))) + v['to3spc'][(v['i'])-1, (v['k'])-1])))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4430 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p517 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1521, 0, v['_go'])
                # WRF:4431 continue
                return v
            v = lax.fori_loop(0,_p518,_p519,v)
            v['i'] = jnp.where((_p517 & (v['_go']==0)), jnp.int32((v['myis'])+_p518*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p514,_p515,v)
        v['k'] = jnp.where((_p513 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p514*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4433 do k=1,l
        _p521 = ((jnp.bool_(True)) & (v['_go']==0))
        _p522 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p523(_p524,v):
            v = dict(v)
            v['k'] = jnp.where((_p521 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p524*(jnp.int32(1))), v['k'])
            # WRF:4434 do i=myis,myie
            _p525 = (((_p521 & (v['_go']==0))) & (v['_go']==0))
            _p526 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p527(_p528,v):
                v = dict(v)
                v['i'] = jnp.where((_p525 & (v['_go']==0)), jnp.int32((v['myis'])+_p528*(jnp.int32(1))), v['i'])
                # WRF:4435 excts(i,k)=excts(i,k)+sorc(i,k,13)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p525 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(13))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1553, 0, v['_go'])
                # WRF:4437 continue
                return v
            v = lax.fori_loop(0,_p526,_p527,v)
            v['i'] = jnp.where((_p525 & (v['_go']==0)), jnp.int32((v['myis'])+_p526*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p522,_p523,v)
        v['k'] = jnp.where((_p521 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p522*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4439 do i=myis,myie
        _p529 = ((jnp.bool_(True)) & (v['_go']==0))
        _p530 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p531(_p532,v):
            v = dict(v)
            v['i'] = jnp.where((_p529 & (v['_go']==0)), jnp.int32((v['myis'])+_p532*(jnp.int32(1))), v['i'])
            # WRF:4440 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,13)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,13)-sorc(i,l,13)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p529 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(13))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(13))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(13))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1561, 0, v['_go'])
            # WRF:4444 continue
            return v
        v = lax.fori_loop(0,_p530,_p531,v)
        v['i'] = jnp.where((_p529 & (v['_go']==0)), jnp.int32((v['myis'])+_p530*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4452 do k=1,l
        _p533 = ((jnp.bool_(True)) & (v['_go']==0))
        _p534 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p535(_p536,v):
            v = dict(v)
            v['k'] = jnp.where((_p533 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p536*(jnp.int32(1))), v['k'])
            # WRF:4453 do i=myis,myie
            _p537 = (((_p533 & (v['_go']==0))) & (v['_go']==0))
            _p538 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p539(_p540,v):
                v = dict(v)
                v['i'] = jnp.where((_p537 & (v['_go']==0)), jnp.int32((v['myis'])+_p540*(jnp.int32(1))), v['i'])
                # WRF:4454 f(i,k)=h44194m2*(apcm(14)*x(i,k)+bpcm(14)*y(i,k))
                v['f'] = v['f'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['apcm'][(jnp.int32(14))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['bpcm'][(jnp.int32(14))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['f'][(v['i'])-1, (v['k'])-1]))
                # WRF:4455 ff(i,k)=h44194m2*(atpcm(14)*x(i,k)+btpcm(14)*y(i,k))
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), _r((v['h44194m2'] * ((v['atpcm'][(jnp.int32(14))-1] * v['x'][(v['i'])-1, (v['k'])-1]) + (v['btpcm'][(jnp.int32(14))-1] * v['y'][(v['i'])-1, (v['k'])-1])))), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:4456 ag(i,k)=(h1p41819+f(i,k))*f(i,k)+one
                v['ag'] = v['ag'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['f'][(v['i'])-1, (v['k'])-1]) * v['f'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['ag'][(v['i'])-1, (v['k'])-1]))
                # WRF:4457 agg(i,k)=(h1p41819+ff(i,k))*ff(i,k)+one
                v['agg'] = v['agg'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), _r((((v['h1p41819'] + v['ff'][(v['i'])-1, (v['k'])-1]) * v['ff'][(v['i'])-1, (v['k'])-1]) + v['one'])), v['agg'][(v['i'])-1, (v['k'])-1]))
                # WRF:4458 phitmp(i,k)=var1(i,k)*(((( ag(i,k)*ag(i,k))**2)**2)**2)
                v['phitmp'] = v['phitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), _r((v['var1'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['ag'][(v['i'])-1, (v['k'])-1] * v['ag'][(v['i'])-1, (v['k'])-1])))))), v['phitmp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4459 psitmp(i,k)=var2(i,k)*(((( agg(i,k)*agg(i,k))**2)**2)**2)
                v['psitmp'] = v['psitmp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), _r((v['var2'][(v['i'])-1, (v['k'])-1] * jnp.square(jnp.square(jnp.square((v['agg'][(v['i'])-1, (v['k'])-1] * v['agg'][(v['i'])-1, (v['k'])-1])))))), v['psitmp'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1601, 0, v['_go'])
                # WRF:4460 continue
                return v
            v = lax.fori_loop(0,_p538,_p539,v)
            v['i'] = jnp.where((_p537 & (v['_go']==0)), jnp.int32((v['myis'])+_p538*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p534,_p535,v)
        v['k'] = jnp.where((_p533 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p534*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4463 do i=myis,myie
        _p541 = ((jnp.bool_(True)) & (v['_go']==0))
        _p542 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p543(_p544,v):
            v = dict(v)
            v['i'] = jnp.where((_p541 & (v['_go']==0)), jnp.int32((v['myis'])+_p544*(jnp.int32(1))), v['i'])
            # WRF:4464 topm(i,1)=phitmp(i,1)
            v['topm'] = v['topm'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p541 & (v['_go']==0))) & (v['_go']==0)), _r(v['phitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topm'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4465 topphi(i,1)=psitmp(i,1)
            v['topphi'] = v['topphi'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p541 & (v['_go']==0))) & (v['_go']==0)), _r(v['psitmp'][(v['i'])-1, (jnp.int32(1))-1]), v['topphi'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==1615, 0, v['_go'])
            # WRF:4466 continue
            return v
        v = lax.fori_loop(0,_p542,_p543,v)
        v['i'] = jnp.where((_p541 & (v['_go']==0)), jnp.int32((v['myis'])+_p542*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4467 do k=2,l
        _p545 = ((jnp.bool_(True)) & (v['_go']==0))
        _p546 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p547(_p548,v):
            v = dict(v)
            v['k'] = jnp.where((_p545 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p548*(jnp.int32(1))), v['k'])
            # WRF:4468 do i=myis,myie
            _p549 = (((_p545 & (v['_go']==0))) & (v['_go']==0))
            _p550 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p551(_p552,v):
                v = dict(v)
                v['i'] = jnp.where((_p549 & (v['_go']==0)), jnp.int32((v['myis'])+_p552*(jnp.int32(1))), v['i'])
                # WRF:4469 topm(i,k)=topm(i,k-1)+phitmp(i,k)
                v['topm'] = v['topm'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p549 & (v['_go']==0))) & (v['_go']==0)), _r((v['topm'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['phitmp'][(v['i'])-1, (v['k'])-1])), v['topm'][(v['i'])-1, (v['k'])-1]))
                # WRF:4470 topphi(i,k)=topphi(i,k-1)+psitmp(i,k)
                v['topphi'] = v['topphi'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p549 & (v['_go']==0))) & (v['_go']==0)), _r((v['topphi'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + v['psitmp'][(v['i'])-1, (v['k'])-1])), v['topphi'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1617, 0, v['_go'])
                # WRF:4471 continue
                return v
            v = lax.fori_loop(0,_p550,_p551,v)
            v['i'] = jnp.where((_p549 & (v['_go']==0)), jnp.int32((v['myis'])+_p550*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1619, 0, v['_go'])
            # WRF:4472 continue
            return v
        v = lax.fori_loop(0,_p546,_p547,v)
        v['k'] = jnp.where((_p545 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p546*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4474 do k=1,l
        _p553 = ((jnp.bool_(True)) & (v['_go']==0))
        _p554 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p555(_p556,v):
            v = dict(v)
            v['k'] = jnp.where((_p553 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p556*(jnp.int32(1))), v['k'])
            # WRF:4475 do i=myis,myie
            _p557 = (((_p553 & (v['_go']==0))) & (v['_go']==0))
            _p558 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p559(_p560,v):
                v = dict(v)
                v['i'] = jnp.where((_p557 & (v['_go']==0)), jnp.int32((v['myis'])+_p560*(jnp.int32(1))), v['i'])
                # WRF:4476 fac1(i,k)=acomb(14)*topm(i,k)
                v['fac1'] = v['fac1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p557 & (v['_go']==0))) & (v['_go']==0)), _r((v['acomb'][(jnp.int32(14))-1] * v['topm'][(v['i'])-1, (v['k'])-1])), v['fac1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4477 fac2(i,k)=fac1(i,k)*topm(i,k)/(bcomb(14)*topphi(i,k))
                v['fac2'] = v['fac2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p557 & (v['_go']==0))) & (v['_go']==0)), _r(((v['fac1'][(v['i'])-1, (v['k'])-1] * v['topm'][(v['i'])-1, (v['k'])-1]) / (v['bcomb'][(jnp.int32(14))-1] * v['topphi'][(v['i'])-1, (v['k'])-1]))), v['fac2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4478 tt(i,k)=exp(hm1ez*(fac1(i,k)/sqrt(one+fac2(i,k))+ betacm(14)*totvo2(i,k+1)*sko2d))
                v['tt'] = v['tt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p557 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * ((v['fac1'][(v['i'])-1, (v['k'])-1] / jnp.sqrt((v['one'] + v['fac2'][(v['i'])-1, (v['k'])-1]))) + ((v['betacm'][(jnp.int32(14))-1] * v['totvo2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * _r(_TABLES['sko2d'])))))), v['tt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4480 ctmp(i,k+1)=tt(i,k)*cldfac(i,k+1,1)
                v['ctmp'] = v['ctmp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p557 & (v['_go']==0))) & (v['_go']==0)), _r((v['tt'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1621, 0, v['_go'])
                # WRF:4481 continue
                return v
            v = lax.fori_loop(0,_p558,_p559,v)
            v['i'] = jnp.where((_p557 & (v['_go']==0)), jnp.int32((v['myis'])+_p558*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p554,_p555,v)
        v['k'] = jnp.where((_p553 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p554*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4483 do k=1,l
        _p561 = ((jnp.bool_(True)) & (v['_go']==0))
        _p562 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p563(_p564,v):
            v = dict(v)
            v['k'] = jnp.where((_p561 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p564*(jnp.int32(1))), v['k'])
            # WRF:4484 do i=myis,myie
            _p565 = (((_p561 & (v['_go']==0))) & (v['_go']==0))
            _p566 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p567(_p568,v):
                v = dict(v)
                v['i'] = jnp.where((_p565 & (v['_go']==0)), jnp.int32((v['myis'])+_p568*(jnp.int32(1))), v['i'])
                # WRF:4485 excts(i,k)=excts(i,k)+sorc(i,k,14)* (ctmp(i,k+1)-ctmp(i,k))
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p565 & (v['_go']==0))) & (v['_go']==0)), _r((v['excts'][(v['i'])-1, (v['k'])-1] + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(14))-1] * (v['ctmp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp'][(v['i'])-1, (v['k'])-1])))), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1653, 0, v['_go'])
                # WRF:4487 continue
                return v
            v = lax.fori_loop(0,_p566,_p567,v)
            v['i'] = jnp.where((_p565 & (v['_go']==0)), jnp.int32((v['myis'])+_p566*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p562,_p563,v)
        v['k'] = jnp.where((_p561 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p562*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4489 do i=myis,myie
        _p569 = ((jnp.bool_(True)) & (v['_go']==0))
        _p570 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p571(_p572,v):
            v = dict(v)
            v['i'] = jnp.where((_p569 & (v['_go']==0)), jnp.int32((v['myis'])+_p572*(jnp.int32(1))), v['i'])
            # WRF:4490 gxcts(i)=gxcts(i)+cldfac(i,lp1,1)*(tt(i,l)*sorc(i,l,14)+ (haf*delp(i,l)*(tt(i,lm1)*(p(i,lp1)-press(i,l)) + tt(i,l)*(p(i,lp1)+press(i,l)-two*p(i,l)))) * (sorc(i,lp1,14)-sorc(i,l,14)))
            v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p569 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] + (v['cldfac'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(1))-1] * ((v['tt'][(v['i'])-1, (v['l'])-1] * v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(14))-1]) + (((v['haf'] * v['delp'][(v['i'])-1, (v['l'])-1]) * ((v['tt'][(v['i'])-1, (v['lm1'])-1] * (v['p'][(v['i'])-1, (v['lp1'])-1] - v['press'][(v['i'])-1, (v['l'])-1])) + (v['tt'][(v['i'])-1, (v['l'])-1] * ((v['p'][(v['i'])-1, (v['lp1'])-1] + v['press'][(v['i'])-1, (v['l'])-1]) - (v['two'] * v['p'][(v['i'])-1, (v['l'])-1]))))) * (v['sorc'][(v['i'])-1, (v['lp1'])-1, (jnp.int32(14))-1] - v['sorc'][(v['i'])-1, (v['l'])-1, (jnp.int32(14))-1])))))), v['gxcts'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==1661, 0, v['_go'])
            # WRF:4494 continue
            return v
        v = lax.fori_loop(0,_p570,_p571,v)
        v['i'] = jnp.where((_p569 & (v['_go']==0)), jnp.int32((v['myis'])+_p570*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4502 do k=1,l
        _p573 = ((jnp.bool_(True)) & (v['_go']==0))
        _p574 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p575(_p576,v):
            v = dict(v)
            v['k'] = jnp.where((_p573 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p576*(jnp.int32(1))), v['k'])
            # WRF:4503 do i=myis,myie
            _p577 = (((_p573 & (v['_go']==0))) & (v['_go']==0))
            _p578 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p579(_p580,v):
                v = dict(v)
                v['i'] = jnp.where((_p577 & (v['_go']==0)), jnp.int32((v['myis'])+_p580*(jnp.int32(1))), v['i'])
                # WRF:4504 gxcts(i)=gxcts(i)-excts(i,k)
                v['gxcts'] = v['gxcts'].at[(v['i'])-1].set(jnp.where((((_p577 & (v['_go']==0))) & (v['_go']==0)), _r((v['gxcts'][(v['i'])-1] - v['excts'][(v['i'])-1, (v['k'])-1])), v['gxcts'][(v['i'])-1]))
                v['_go'] = jnp.where(v['_go']==1731, 0, v['_go'])
                # WRF:4505 continue
                return v
            v = lax.fori_loop(0,_p578,_p579,v)
            v['i'] = jnp.where((_p577 & (v['_go']==0)), jnp.int32((v['myis'])+_p578*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p574,_p575,v)
        v['k'] = jnp.where((_p573 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p574*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4509 do k=1,l
        _p581 = ((jnp.bool_(True)) & (v['_go']==0))
        _p582 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p583(_p584,v):
            v = dict(v)
            v['k'] = jnp.where((_p581 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p584*(jnp.int32(1))), v['k'])
            # WRF:4510 do i=myis,myie
            _p585 = (((_p581 & (v['_go']==0))) & (v['_go']==0))
            _p586 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p587(_p588,v):
                v = dict(v)
                v['i'] = jnp.where((_p585 & (v['_go']==0)), jnp.int32((v['myis'])+_p588*(jnp.int32(1))), v['i'])
                # WRF:4511 excts(i,k)=excts(i,k)*radcon*delp(i,k)
                v['excts'] = v['excts'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p585 & (v['_go']==0))) & (v['_go']==0)), _r(((v['excts'][(v['i'])-1, (v['k'])-1] * v['radcon']) * v['delp'][(v['i'])-1, (v['k'])-1])), v['excts'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1741, 0, v['_go'])
                # WRF:4512 continue
                return v
            v = lax.fori_loop(0,_p586,_p587,v)
            v['i'] = jnp.where((_p585 & (v['_go']==0)), jnp.int32((v['myis'])+_p586*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p582,_p583,v)
        v['k'] = jnp.where((_p581 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p582*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4518 do k=1,l
        _p589 = ((jnp.bool_(True)) & (v['_go']==0))
        _p590 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p591(_p592,v):
            v = dict(v)
            v['k'] = jnp.where((_p589 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p592*(jnp.int32(1))), v['k'])
            # WRF:4519 do i=myis,myie
            _p593 = (((_p589 & (v['_go']==0))) & (v['_go']==0))
            _p594 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p595(_p596,v):
                v = dict(v)
                v['i'] = jnp.where((_p593 & (v['_go']==0)), jnp.int32((v['myis'])+_p596*(jnp.int32(1))), v['i'])
                # WRF:4520 ctmp2(i,k+1)=co2sp(i,k+1)*cldfac(i,k+1,1)
                v['ctmp2'] = v['ctmp2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p593 & (v['_go']==0))) & (v['_go']==0)), _r((v['co2sp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:4521 ctmp3(i,k+1)=to3sp(i,k)*cldfac(i,k+1,1)
                v['ctmp3'] = v['ctmp3'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p593 & (v['_go']==0))) & (v['_go']==0)), _r((v['to3sp'][(v['i'])-1, (v['k'])-1] * v['cldfac'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1, (jnp.int32(1))-1])), v['ctmp3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==1711, 0, v['_go'])
                # WRF:4522 continue
                return v
            v = lax.fori_loop(0,_p594,_p595,v)
            v['i'] = jnp.where((_p593 & (v['_go']==0)), jnp.int32((v['myis'])+_p594*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p590,_p591,v)
        v['k'] = jnp.where((_p589 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p590*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4523 do k=1,l
        _p597 = ((jnp.bool_(True)) & (v['_go']==0))
        _p598 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p599(_p600,v):
            v = dict(v)
            v['k'] = jnp.where((_p597 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p600*(jnp.int32(1))), v['k'])
            # WRF:4524 do i=myis,myie
            _p601 = (((_p597 & (v['_go']==0))) & (v['_go']==0))
            _p602 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p603(_p604,v):
                v = dict(v)
                v['i'] = jnp.where((_p601 & (v['_go']==0)), jnp.int32((v['myis'])+_p604*(jnp.int32(1))), v['i'])
                # WRF:4525 ctso3(i,k)=radcon*delp(i,k)* (csour(i,k)*(ctmp2(i,k+1)-ctmp2(i,k)) + sorc(i,k,13)*(ctmp3(i,k+1)-ctmp3(i,k)))
                v['ctso3'] = v['ctso3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p601 & (v['_go']==0))) & (v['_go']==0)), _r(((v['radcon'] * v['delp'][(v['i'])-1, (v['k'])-1]) * ((v['csour'][(v['i'])-1, (v['k'])-1] * (v['ctmp2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp2'][(v['i'])-1, (v['k'])-1])) + (v['sorc'][(v['i'])-1, (v['k'])-1, (jnp.int32(13))-1] * (v['ctmp3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['ctmp3'][(v['i'])-1, (v['k'])-1]))))), v['ctso3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1701, 0, v['_go'])
                # WRF:4528 continue
                return v
            v = lax.fori_loop(0,_p602,_p603,v)
            v['i'] = jnp.where((_p601 & (v['_go']==0)), jnp.int32((v['myis'])+_p602*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p598,_p599,v)
        v['k'] = jnp.where((_p597 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p598*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['excts'], v['ctso3'], v['gxcts'], v['sorc'], v['csour'], v['cldfac'], v['temp'], v['press'], v['var1'], v['var2'], v['p'], v['delp'], v['delp2'], v['totvo2'], v['to3sp'], v['to3spc'], v['co2sp1'], v['co2sp2'], v['co2sp'], v['apcm'], v['bpcm'], v['atpcm'], v['btpcm'], v['acomb'], v['bcomb'], v['betacm'], v['h25e2'], v['one'], v['h44194m2'], v['h1p41819'], v['haf'], v['hm1ez'], v['two'], v['radcon'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)

    def _e290(emissb, emiss, avephi, klen, fxoe2, dte2, h16e1, hp1, h28e1, haf, ten, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(emissb).dtype == jnp.dtype("float32") else jnp.float64
        v = {'emissb': emissb, 'emiss': emiss, 'avephi': avephi, 'klen': klen, 'fxoe2': fxoe2, 'dte2': dte2, 'h16e1': h16e1, 'hp1': hp1, 'h28e1': h28e1, 'haf': haf, 'ten': ten, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['dt'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['du'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['fyo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['ival'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=jnp.int32)
        v['k'] = jnp.int32(0)
        v['l'] = jnp.int32(0)
        v['ll'] = jnp.int32(0)
        v['llm1'] = jnp.int32(0)
        v['llm2'] = jnp.int32(0)
        v['llp1'] = jnp.int32(0)
        v['lm1'] = jnp.int32(0)
        v['lm2'] = jnp.int32(0)
        v['lm3'] = jnp.int32(0)
        v['lp1'] = jnp.int32(0)
        v['lp2'] = jnp.int32(0)
        v['lp3'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['tmp3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        # WRF:4565 l=kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:4566 lp1=l+1
        v['lp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(1))), v['lp1'])
        # WRF:4566 lp2=l+2
        v['lp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(2))), v['lp2'])
        # WRF:4566 lp3=l+3
        v['lp3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(3))), v['lp3'])
        # WRF:4566 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:4567 lm1=l-1
        v['lm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(1))), v['lm1'])
        # WRF:4567 lm2=l-2
        v['lm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(2))), v['lm2'])
        # WRF:4567 lm3=l-3
        v['lm3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(3))), v['lm3'])
        # WRF:4567 ll = 2*l
        v['ll'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((jnp.int32(2) * v['l'])), v['ll'])
        # WRF:4568 llm2 = ll-2
        v['llm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(2))), v['llm2'])
        # WRF:4568 llm1=ll-1
        v['llm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['ll'] - jnp.int32(1))), v['llm1'])
        # WRF:4569 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:4569 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:4581 do k=1,lp2-klen
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), (((v['lp2'] - v['klen']))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p4*(jnp.int32(1))), v['k'])
            # WRF:4582 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:4583 tmp3(i,k)=log10(avephi(i,klen+k-1))+h16e1
                v['tmp3'] = v['tmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.log10(v['avephi'][(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1]) + v['h16e1'])), v['tmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:4584 fyo(i,k)=aint(tmp3(i,k)*ten)
                v['fyo'] = v['fyo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['tmp3'][(v['i'])-1, (v['k'])-1] * v['ten']))), v['fyo'][(v['i'])-1, (v['k'])-1]))
                # WRF:4585 du(i,k)=tmp3(i,k)-hp1*fyo(i,k)
                v['du'] = v['du'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['tmp3'][(v['i'])-1, (v['k'])-1] - (v['hp1'] * v['fyo'][(v['i'])-1, (v['k'])-1]))), v['du'][(v['i'])-1, (v['k'])-1]))
                # WRF:4586 fyo(i,k)=h28e1*fyo(i,k)
                v['fyo'] = v['fyo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['h28e1'] * v['fyo'][(v['i'])-1, (v['k'])-1])), v['fyo'][(v['i'])-1, (v['k'])-1]))
                # WRF:4587 ival(i,k)=fyo(i,k)+fxoe2(i,klen+k-1)
                v['ival'] = v['ival'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fyo'][(v['i'])-1, (v['k'])-1] + v['fxoe2'][(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1])), v['ival'][(v['i'])-1, (v['k'])-1]))
                # WRF:4588 emiss(i,klen+k-1)=t1(ival(i,k))+du(i,k)*t2(ival(i,k)) +dte2(i,klen+k-1)*t4(ival(i,k))
                v['emiss'] = v['emiss'].at[(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(((_r(_TABLES['t1'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1] + (v['du'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t2'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1])) + (v['dte2'][(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1] * _r(_TABLES['t4'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1]))), v['emiss'][(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==132, 0, v['_go'])
                # WRF:4590 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4593 do i=myis,myie
        _p9 = ((jnp.bool_(True)) & (v['_go']==0))
        _p10 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p11(_p12,v):
            v = dict(v)
            v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p12*(jnp.int32(1))), v['i'])
            # WRF:4594 emiss(i,l)=haf*(emiss(i,l)+emiss(i,lp1))
            v['emiss'] = v['emiss'].at[(v['i'])-1, (v['l'])-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['emiss'][(v['i'])-1, (v['l'])-1] + v['emiss'][(v['i'])-1, (v['lp1'])-1]))), v['emiss'][(v['i'])-1, (v['l'])-1]))
            v['_go'] = jnp.where(v['_go']==1344, 0, v['_go'])
            # WRF:4595 continue
            return v
        v = lax.fori_loop(0,_p10,_p11,v)
        v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p10*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4605 do k=1,lp1-klen
        _p13 = ((jnp.bool_(True)) & (v['_go']==0))
        _p14 = jnp.maximum(jnp.int32(0), (((v['lp1'] - v['klen']))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p15(_p16,v):
            v = dict(v)
            v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p16*(jnp.int32(1))), v['k'])
            # WRF:4606 do i=myis,myie
            _p17 = (((_p13 & (v['_go']==0))) & (v['_go']==0))
            _p18 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p19(_p20,v):
                v = dict(v)
                v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p20*(jnp.int32(1))), v['i'])
                # WRF:4607 dt(i,k)=dte2(i,klen-1)
                v['dt'] = v['dt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r(v['dte2'][(v['i'])-1, ((v['klen'] - jnp.int32(1)))-1]), v['dt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4608 ival(i,k)=fyo(i,k)+fxoe2(i,klen-1)
                v['ival'] = v['ival'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fyo'][(v['i'])-1, (v['k'])-1] + v['fxoe2'][(v['i'])-1, ((v['klen'] - jnp.int32(1)))-1])), v['ival'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==142, 0, v['_go'])
                # WRF:4609 continue
                return v
            v = lax.fori_loop(0,_p18,_p19,v)
            v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p18*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p14,_p15,v)
        v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p14*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4611 do k=1,lp1-klen
        _p21 = ((jnp.bool_(True)) & (v['_go']==0))
        _p22 = jnp.maximum(jnp.int32(0), (((v['lp1'] - v['klen']))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p23(_p24,v):
            v = dict(v)
            v['k'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p24*(jnp.int32(1))), v['k'])
            # WRF:4612 do i=myis,myie
            _p25 = (((_p21 & (v['_go']==0))) & (v['_go']==0))
            _p26 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p27(_p28,v):
                v = dict(v)
                v['i'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((v['myis'])+_p28*(jnp.int32(1))), v['i'])
                # WRF:4613 emissb(i,klen+k-1)=t1(ival(i,k))+du(i,k)*t2(ival(i,k)) +dt(i,k)*t4(ival(i,k))
                v['emissb'] = v['emissb'].at[(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1].set(jnp.where((((_p25 & (v['_go']==0))) & (v['_go']==0)), _r(((_r(_TABLES['t1'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1] + (v['du'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t2'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1])) + (v['dt'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t4'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1]))), v['emissb'][(v['i'])-1, (((v['klen'] + v['k']) - jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==234, 0, v['_go'])
                # WRF:4615 continue
                return v
            v = lax.fori_loop(0,_p26,_p27,v)
            v['i'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((v['myis'])+_p26*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p22,_p23,v)
        v['k'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p22*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['emissb'], v['emiss'], v['avephi'], v['klen'], v['fxoe2'], v['dte2'], v['h16e1'], v['hp1'], v['h28e1'], v['haf'], v['ten'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)

    def _e2spec(emiss, avephi, fxosp, dtsp, h16e1, ten, h28e1, hp1, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(emiss).dtype == jnp.dtype("float32") else jnp.float64
        v = {'emiss': emiss, 'avephi': avephi, 'fxosp': fxosp, 'dtsp': dtsp, 'h16e1': h16e1, 'ten': ten, 'h28e1': h28e1, 'hp1': hp1, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['du'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['fyo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['ival'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=jnp.int32)
        v['k'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['tmp3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        # WRF:4649 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:4650 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:4652 do k=1,2
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((jnp.int32(2))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p4*(jnp.int32(1))), v['k'])
            # WRF:4653 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:4654 tmp3(i,k)=log10(avephi(i,k))+h16e1
                v['tmp3'] = v['tmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.log10(v['avephi'][(v['i'])-1, (v['k'])-1]) + v['h16e1'])), v['tmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:4655 fyo(i,k)=aint(tmp3(i,k)*ten)
                v['fyo'] = v['fyo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['tmp3'][(v['i'])-1, (v['k'])-1] * v['ten']))), v['fyo'][(v['i'])-1, (v['k'])-1]))
                # WRF:4656 du(i,k)=tmp3(i,k)-hp1*fyo(i,k)
                v['du'] = v['du'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['tmp3'][(v['i'])-1, (v['k'])-1] - (v['hp1'] * v['fyo'][(v['i'])-1, (v['k'])-1]))), v['du'][(v['i'])-1, (v['k'])-1]))
                # WRF:4657 ival(i,k)=h28e1*fyo(i,k)+fxosp(i,k)
                v['ival'] = v['ival'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(((v['h28e1'] * v['fyo'][(v['i'])-1, (v['k'])-1]) + v['fxosp'][(v['i'])-1, (v['k'])-1])), v['ival'][(v['i'])-1, (v['k'])-1]))
                # WRF:4658 emiss(i,k)=t1(ival(i,k))+du(i,k)*t2(ival(i,k))+ dtsp(i,k)*t4(ival(i,k))
                v['emiss'] = v['emiss'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(((_r(_TABLES['t1'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1] + (v['du'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t2'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1])) + (v['dtsp'][(v['i'])-1, (v['k'])-1] * _r(_TABLES['t4'])[(v['ival'][(v['i'])-1, (v['k'])-1])-1]))), v['emiss'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==132, 0, v['_go'])
                # WRF:4660 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['emiss'], v['avephi'], v['fxosp'], v['dtsp'], v['h16e1'], v['ten'], v['h28e1'], v['hp1'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)

    def _e3v88(emv, tv, av, ten, hp1, h28e1, h16e1, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(emv).dtype == jnp.dtype("float32") else jnp.float64
        v = {'emv': emv, 'tv': tv, 'av': av, 'ten': ten, 'hp1': hp1, 'h28e1': h28e1, 'h16e1': h16e1, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['dt'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['du'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['fxo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['fyo'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['i'] = jnp.int32(0)
        v['it'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=jnp.int32)
        v['k'] = jnp.int32(0)
        v['l'] = jnp.int32(0)
        v['llp1'] = jnp.int32(0)
        v['myie'] = jnp.int32(0)
        v['myis'] = jnp.int32(0)
        v['tmp3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['ww1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        v['ww2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+1)-((1))+1,), dtype=_r)
        # WRF:4693 l = kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:4694 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:4695 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:4695 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:4700 do k=1,llp1
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((v['llp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p4*(jnp.int32(1))), v['k'])
            # WRF:4701 do i=myis,myie
            _p5 = (((_p1 & (v['_go']==0))) & (v['_go']==0))
            _p6 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p7(_p8,v):
                v = dict(v)
                v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p8*(jnp.int32(1))), v['i'])
                # WRF:4702 fxo(i,k)=aint(tv(i,k)*hp1)
                v['fxo'] = v['fxo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['tv'][(v['i'])-1, (v['k'])-1] * v['hp1']))), v['fxo'][(v['i'])-1, (v['k'])-1]))
                # WRF:4703 tmp3(i,k)=log10(av(i,k))+h16e1
                v['tmp3'] = v['tmp3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.log10(v['av'][(v['i'])-1, (v['k'])-1]) + v['h16e1'])), v['tmp3'][(v['i'])-1, (v['k'])-1]))
                # WRF:4704 dt(i,k)=tv(i,k)-ten*fxo(i,k)
                v['dt'] = v['dt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['tv'][(v['i'])-1, (v['k'])-1] - (v['ten'] * v['fxo'][(v['i'])-1, (v['k'])-1]))), v['dt'][(v['i'])-1, (v['k'])-1]))
                # WRF:4705 fyo(i,k)=aint(tmp3(i,k)*ten)
                v['fyo'] = v['fyo'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.trunc((v['tmp3'][(v['i'])-1, (v['k'])-1] * v['ten']))), v['fyo'][(v['i'])-1, (v['k'])-1]))
                # WRF:4706 du(i,k)=tmp3(i,k)-hp1*fyo(i,k)
                v['du'] = v['du'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['tmp3'][(v['i'])-1, (v['k'])-1] - (v['hp1'] * v['fyo'][(v['i'])-1, (v['k'])-1]))), v['du'][(v['i'])-1, (v['k'])-1]))
                # WRF:4709 it(i,k)=fxo(i,k)+fyo(i,k)*h28e1
                v['it'] = v['it'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), jnp.int32((v['fxo'][(v['i'])-1, (v['k'])-1] + (v['fyo'][(v['i'])-1, (v['k'])-1] * v['h28e1']))), v['it'][(v['i'])-1, (v['k'])-1]))
                # WRF:4710 ww1(i,k)=ten-dt(i,k)
                v['ww1'] = v['ww1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['ten'] - v['dt'][(v['i'])-1, (v['k'])-1])), v['ww1'][(v['i'])-1, (v['k'])-1]))
                # WRF:4711 ww2(i,k)=hp1-du(i,k)
                v['ww2'] = v['ww2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((v['hp1'] - v['du'][(v['i'])-1, (v['k'])-1])), v['ww2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4712 emv(i,k)=ww1(i,k)*ww2(i,k)*em3v(it(i,k)-9)+ ww2(i,k)*dt(i,k)*em3v(it(i,k)-8)+ ww1(i,k)*du(i,k)*em3v(it(i,k)+19)+ dt(i,k)*du(i,k)*em3v(it(i,k)+20)
                v['emv'] = v['emv'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p5 & (v['_go']==0))) & (v['_go']==0)), _r((((((v['ww1'][(v['i'])-1, (v['k'])-1] * v['ww2'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em3v'])[((v['it'][(v['i'])-1, (v['k'])-1] - jnp.int32(9)))-1]) + ((v['ww2'][(v['i'])-1, (v['k'])-1] * v['dt'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em3v'])[((v['it'][(v['i'])-1, (v['k'])-1] - jnp.int32(8)))-1])) + ((v['ww1'][(v['i'])-1, (v['k'])-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em3v'])[((v['it'][(v['i'])-1, (v['k'])-1] + jnp.int32(19)))-1])) + ((v['dt'][(v['i'])-1, (v['k'])-1] * v['du'][(v['i'])-1, (v['k'])-1]) * _r(_TABLES['em3v'])[((v['it'][(v['i'])-1, (v['k'])-1] + jnp.int32(20)))-1]))), v['emv'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==203, 0, v['_go'])
                # WRF:4716 continue
                return v
            v = lax.fori_loop(0,_p6,_p7,v)
            v['i'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((v['myis'])+_p6*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['k'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p2*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['emv'], v['tv'], v['av'], v['ten'], v['hp1'], v['h28e1'], v['h16e1'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)
    return _lwr88
