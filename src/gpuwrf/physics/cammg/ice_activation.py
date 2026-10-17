"""Generated offline from pristine WRF CAM-MG; see proofs/s2_cammg/port.py.
Qualified against original WRF public/staged calls; native GPU qualification pending.
"""
import jax.numpy as jnp
from ._support import _r,_d,_int,_get,_put,_slice,_loop,_cond,_cast,_dtype,_pow,_powi,_retry,_zeros,_literal_array,_static,_neg

from .error_function import erf
from .cloud import polysvp
def hf(t,ww,rh,subgrid,na,ni,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('microp_aero',{}))
    v['t']=t
    v['ww']=ww
    v['rh']=rh
    v['subgrid']=subgrid
    v['na']=na
    v['ni']=ni
    v['_jump']=_int(0)
    v['a1_fast']=_zeros((),'f64',xp)
    v['a21_fast']=_zeros((),'f64',xp)
    v['a22_fast']=_zeros((),'f64',xp)
    v['b1_fast']=_zeros((),'f64',xp)
    v['b21_fast']=_zeros((),'f64',xp)
    v['b22_fast']=_zeros((),'f64',xp)
    v['a2_fast']=_zeros((),'f64',xp)
    v['b2_fast']=_zeros((),'f64',xp)
    v['c1_fast']=_zeros((),'f64',xp)
    v['c2_fast']=_zeros((),'f64',xp)
    v['k1_fast']=_zeros((),'f64',xp)
    v['k2_fast']=_zeros((),'f64',xp)
    v['a1_slow']=_zeros((),'f64',xp)
    v['a2_slow']=_zeros((),'f64',xp)
    v['b1_slow']=_zeros((),'f64',xp)
    v['b2_slow']=_zeros((),'f64',xp)
    v['b3_slow']=_zeros((),'f64',xp)
    v['c1_slow']=_zeros((),'f64',xp)
    v['c2_slow']=_zeros((),'f64',xp)
    v['k1_slow']=_zeros((),'f64',xp)
    v['k2_slow']=_zeros((),'f64',xp)
    v['regm']=_zeros((),'f64',xp)
    v['a']=_zeros((),'f64',xp)
    v['b']=_zeros((),'f64',xp)
    v['c']=_zeros((),'f64',xp)
    v['rhw']=_zeros((),'f64',xp)
    _put(v,'a1_fast',None,_d(0.0231))  # WRF:module_cam_mp_microp_aero.F:1447
    _put(v,'a21_fast',None,_neg(_d(1.6387)))  # WRF:module_cam_mp_microp_aero.F:1448
    _put(v,'a22_fast',None,_neg(_d(6.045)))  # WRF:module_cam_mp_microp_aero.F:1449
    _put(v,'b1_fast',None,_neg(_d(0.008)))  # WRF:module_cam_mp_microp_aero.F:1450
    _put(v,'b21_fast',None,_neg(_d(0.042)))  # WRF:module_cam_mp_microp_aero.F:1451
    _put(v,'b22_fast',None,_neg(_d(0.112)))  # WRF:module_cam_mp_microp_aero.F:1452
    _put(v,'c1_fast',None,_d(0.0739))  # WRF:module_cam_mp_microp_aero.F:1453
    _put(v,'c2_fast',None,_d(1.2372))  # WRF:module_cam_mp_microp_aero.F:1454
    _put(v,'a1_slow',None,_neg(_d(0.3949)))  # WRF:module_cam_mp_microp_aero.F:1456
    _put(v,'a2_slow',None,_d(1.282))  # WRF:module_cam_mp_microp_aero.F:1457
    _put(v,'b1_slow',None,_neg(_d(0.0156)))  # WRF:module_cam_mp_microp_aero.F:1458
    _put(v,'b2_slow',None,_d(0.0111))  # WRF:module_cam_mp_microp_aero.F:1459
    _put(v,'b3_slow',None,_d(0.0217))  # WRF:module_cam_mp_microp_aero.F:1460
    _put(v,'c1_slow',None,_d(0.12))  # WRF:module_cam_mp_microp_aero.F:1461
    _put(v,'c2_slow',None,_d(2.312))  # WRF:module_cam_mp_microp_aero.F:1462
    _put(v,'ni',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1464
    _put(v,'a',None,_d(0.0006) * xp.log(v['ww']) + _d(0.0066))  # WRF:module_cam_mp_microp_aero.F:1468
    _put(v,'b',None,_d(0.06) * xp.log(v['ww']) + _d(1.052))  # WRF:module_cam_mp_microp_aero.F:1469
    _put(v,'c',None,_d(1.68) * xp.log(v['ww']) + _d(129.35))  # WRF:module_cam_mp_microp_aero.F:1470
    _put(v,'rhw',None,(v['a'] * v['t'] * v['t'] + v['b'] * v['t'] + v['c']) * _d(0.01))  # WRF:module_cam_mp_microp_aero.F:1471
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni', 'regm'),_carry))
        _put(v,'regm',None,_d(6.07) * xp.log(v['ww']) - _d(55.0))  # WRF:module_cam_mp_microp_aero.F:1475
        _outer2=dict(v)
        def _yes2(_carry):
            v=dict(_outer2);v.update(zip(('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni'),_carry))
            _outer3=dict(v)
            def _yes3(_carry):
                v=dict(_outer3);v.update(zip(('a2_fast', 'b2_fast'),_carry))
                _put(v,'a2_fast',None,v['a21_fast'])  # WRF:module_cam_mp_microp_aero.F:1480
                _put(v,'b2_fast',None,v['b21_fast'])  # WRF:module_cam_mp_microp_aero.F:1481
                return tuple(v[_n] for _n in ('a2_fast', 'b2_fast'))
            def _no3(_carry):
                v=dict(_outer3);v.update(zip(('a2_fast', 'b2_fast'),_carry))
                _put(v,'a2_fast',None,v['a22_fast'])  # WRF:module_cam_mp_microp_aero.F:1483
                _put(v,'b2_fast',None,v['b22_fast'])  # WRF:module_cam_mp_microp_aero.F:1484
                return tuple(v[_n] for _n in ('a2_fast', 'b2_fast'))
            _result=_cond(v['t'] > _neg(_d(64.0)),_yes3,_no3,tuple(v[_n] for _n in ('a2_fast', 'b2_fast')),xp)  # WRF:module_cam_mp_microp_aero.F:1479
            v.update(zip(('a2_fast', 'b2_fast'),_result))
            _put(v,'k1_fast',None,xp.exp(v['a2_fast'] + v['b2_fast'] * v['t'] + v['c2_fast'] * xp.log(v['ww'])))  # WRF:module_cam_mp_microp_aero.F:1487
            _put(v,'k2_fast',None,v['a1_fast'] + v['b1_fast'] * v['t'] + v['c1_fast'] * xp.log(v['ww']))  # WRF:module_cam_mp_microp_aero.F:1488
            _put(v,'ni',None,v['k1_fast'] * _pow(v['na'], v['k2_fast'], xp))  # WRF:module_cam_mp_microp_aero.F:1490
            _put(v,'ni',None,xp.minimum(v['ni'], v['na']))  # WRF:module_cam_mp_microp_aero.F:1491
            return tuple(v[_n] for _n in ('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni'))
        def _no2(_carry):
            v=dict(_outer2);v.update(zip(('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni'),_carry))
            _put(v,'k1_slow',None,xp.exp(v['a2_slow'] + (v['b2_slow'] + v['b3_slow'] * xp.log(v['ww'])) * v['t'] + v['c2_slow'] * xp.log(v['ww'])))  # WRF:module_cam_mp_microp_aero.F:1495
            _put(v,'k2_slow',None,v['a1_slow'] + v['b1_slow'] * v['t'] + v['c1_slow'] * xp.log(v['ww']))  # WRF:module_cam_mp_microp_aero.F:1496
            _put(v,'ni',None,v['k1_slow'] * _pow(v['na'], v['k2_slow'], xp))  # WRF:module_cam_mp_microp_aero.F:1498
            _put(v,'ni',None,xp.minimum(v['ni'], v['na']))  # WRF:module_cam_mp_microp_aero.F:1499
            return tuple(v[_n] for _n in ('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni'))
        _result=_cond(v['t'] >= v['regm'],_yes2,_no2,tuple(v[_n] for _n in ('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni')),xp)  # WRF:module_cam_mp_microp_aero.F:1477
        v.update(zip(('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni'),_result))
        return tuple(v[_n] for _n in ('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni', 'regm'))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni', 'regm'),_carry))
        return tuple(v[_n] for _n in ('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni', 'regm'))
    _result=_cond(xp.logical_and(v['t'] <= _neg(_d(37.0)), v['rh'] * v['subgrid'] >= v['rhw']),_yes1,_no1,tuple(v[_n] for _n in ('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni', 'regm')),xp)  # WRF:module_cam_mp_microp_aero.F:1473
    v.update(zip(('a2_fast', 'b2_fast', 'k1_fast', 'k1_slow', 'k2_fast', 'k2_slow', 'ni', 'regm'),_result))
    return (v['ni'],)

def hetero(t,ww,ns,nis,nid,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('microp_aero',{}))
    v['t']=t
    v['ww']=ww
    v['ns']=ns
    v['nis']=nis
    v['nid']=nid
    v['_jump']=_int(0)
    v['a11']=_zeros((),'f64',xp)
    v['a12']=_zeros((),'f64',xp)
    v['a21']=_zeros((),'f64',xp)
    v['a22']=_zeros((),'f64',xp)
    v['b11']=_zeros((),'f64',xp)
    v['b12']=_zeros((),'f64',xp)
    v['b21']=_zeros((),'f64',xp)
    v['b22']=_zeros((),'f64',xp)
    v['b']=_zeros((),'f64',xp)
    v['c']=_zeros((),'f64',xp)
    _put(v,'a11',None,_d(0.0263))  # WRF:module_cam_mp_microp_aero.F:1407
    _put(v,'a12',None,_neg(_d(0.0185)))  # WRF:module_cam_mp_microp_aero.F:1408
    _put(v,'a21',None,_d(2.758))  # WRF:module_cam_mp_microp_aero.F:1409
    _put(v,'a22',None,_d(1.3221))  # WRF:module_cam_mp_microp_aero.F:1410
    _put(v,'b11',None,_neg(_d(0.008)))  # WRF:module_cam_mp_microp_aero.F:1411
    _put(v,'b12',None,_neg(_d(0.0468)))  # WRF:module_cam_mp_microp_aero.F:1412
    _put(v,'b21',None,_neg(_d(0.2667)))  # WRF:module_cam_mp_microp_aero.F:1413
    _put(v,'b22',None,_neg(_d(1.4588)))  # WRF:module_cam_mp_microp_aero.F:1414
    _put(v,'b',None,(v['a11'] + v['b11'] * xp.log(v['ns'])) * xp.log(v['ww']) + (v['a12'] + v['b12'] * xp.log(v['ns'])))  # WRF:module_cam_mp_microp_aero.F:1418
    _put(v,'c',None,v['a21'] + v['b21'] * xp.log(v['ns']))  # WRF:module_cam_mp_microp_aero.F:1419
    _put(v,'nis',None,xp.exp(v['a22']) * _pow(v['ns'], v['b22'], xp) * xp.exp(v['b'] * v['t']) * _pow(v['ww'], v['c'], xp))  # WRF:module_cam_mp_microp_aero.F:1421
    _put(v,'nis',None,xp.minimum(v['nis'], v['ns']))  # WRF:module_cam_mp_microp_aero.F:1422
    _put(v,'nid',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1424
    return (v['nis'],v['nid'],)

def nucleati(wbar,tair,relhum,cldn,qc,nfice,rhoair,qaerpt,dgnum,ptype,naer_all,nuci,onihf,oniimm,onidep,onimey,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('microp_aero',{}))
    v['wbar']=wbar
    v['tair']=tair
    v['relhum']=relhum
    v['cldn']=cldn
    v['qc']=qc
    v['nfice']=nfice
    v['rhoair']=rhoair
    v['qaerpt']=qaerpt
    v['dgnum']=dgnum
    v['ptype']=ptype
    v['naer_all']=naer_all
    v['nuci']=nuci
    v['onihf']=onihf
    v['oniimm']=oniimm
    v['onidep']=onidep
    v['onimey']=onimey
    v['_jump']=_int(0)
    v['so4_num']=_zeros((),'f64',xp)
    v['soot_num']=_zeros((),'f64',xp)
    v['dst1_num']=_zeros((),'f64',xp)
    v['dst2_num']=_zeros((),'f64',xp)
    v['dst3_num']=_zeros((),'f64',xp)
    v['dst4_num']=_zeros((),'f64',xp)
    v['dst_num']=_zeros((),'f64',xp)
    v['nihf']=_zeros((),'f64',xp)
    v['niimm']=_zeros((),'f64',xp)
    v['nidep']=_zeros((),'f64',xp)
    v['nimey']=_zeros((),'f64',xp)
    v['n1']=_zeros((),'f64',xp)
    v['ni']=_zeros((),'f64',xp)
    v['tc']=_zeros((),'f64',xp)
    v['a']=_zeros((),'f64',xp)
    v['b']=_zeros((),'f64',xp)
    v['c']=_zeros((),'f64',xp)
    v['regm']=_zeros((),'f64',xp)
    v['rhw']=_zeros((),'f64',xp)
    v['esl']=_zeros((),'f64',xp)
    v['esi']=_zeros((),'f64',xp)
    v['deles']=_zeros((),'f64',xp)
    v['subgrid']=_zeros((),'f64',xp)
    v['dmc']=_zeros((),'f64',xp)
    v['ssmc']=_zeros((),'f64',xp)
    _put(v,'so4_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1227
    _put(v,'soot_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1228
    _put(v,'dst_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1229
    _put(v,'dst1_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1230
    _put(v,'dst2_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1231
    _put(v,'dst3_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1232
    _put(v,'dst4_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1233
    _put(v,'soot_num',None,_get(v['qaerpt'], (_get(v['numptr_amode'], (v['modeptr_accum'],)),)) * _d(1e-06))  # WRF:module_cam_mp_microp_aero.F:1242
    _put(v,'dmc',None,_get(v['qaerpt'], (_get(v['lptr_dust_a_amode'], (v['modeptr_coarse'],)),)))  # WRF:module_cam_mp_microp_aero.F:1247
    _put(v,'ssmc',None,_get(v['qaerpt'], (_get(v['lptr_nacl_a_amode'], (v['modeptr_coarse'],)),)))  # WRF:module_cam_mp_microp_aero.F:1248
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('dst_num',),_carry))
        _put(v,'dst_num',None,v['dmc'] / (v['ssmc'] + v['dmc']) * _get(v['qaerpt'], (_get(v['numptr_amode'], (v['modeptr_coarse'],)),)) * _d(1e-06))  # WRF:module_cam_mp_microp_aero.F:1250
        return tuple(v[_n] for _n in ('dst_num',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('dst_num',),_carry))
        _put(v,'dst_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1252
        return tuple(v[_n] for _n in ('dst_num',))
    _result=_cond(v['dmc'] > _d(0.0),_yes1,_no1,tuple(v[_n] for _n in ('dst_num',)),xp)  # WRF:module_cam_mp_microp_aero.F:1249
    v.update(zip(('dst_num',),_result))
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('so4_num',),_carry))
        _put(v,'so4_num',None,_get(v['qaerpt'], (_get(v['numptr_amode'], (v['modeptr_aitken'],)),)) * _d(1e-06) * (_d(0.5) - _d(0.5) * erf(xp.log(_d(1e-07) / _get(v['dgnum'], (v['modeptr_aitken'],))) / (_pow(_d(2.0), _d(0.5), xp) * xp.log(_get(v['sigmag_amode'], (v['modeptr_aitken'],)))), xp=xp, _globals=_globals)))  # WRF:module_cam_mp_microp_aero.F:1257
        return tuple(v[_n] for _n in ('so4_num',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('so4_num',),_carry))
        _put(v,'so4_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1261
        return tuple(v[_n] for _n in ('so4_num',))
    _result=_cond(_get(v['dgnum'], (v['modeptr_aitken'],)) > _d(0.0),_yes2,_no2,tuple(v[_n] for _n in ('so4_num',)),xp)  # WRF:module_cam_mp_microp_aero.F:1256
    v.update(zip(('so4_num',),_result))
    _put(v,'so4_num',None,xp.maximum(_d(0.0), v['so4_num']))  # WRF:module_cam_mp_microp_aero.F:1263
    _put(v,'soot_num',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1296
    _put(v,'ni',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1298
    _put(v,'tc',None,v['tair'] - _d(273.15))  # WRF:module_cam_mp_microp_aero.F:1299
    _put(v,'niimm',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1302
    _put(v,'nidep',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1303
    _put(v,'nihf',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1304
    _outer3=dict(v)
    def _yes3(_carry):
        v=dict(_outer3);v.update(zip(('a', 'b', 'c', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm', 'rhw', 'subgrid'),_carry))
        _put(v,'a',None,_d(0.0073))  # WRF:module_cam_mp_microp_aero.F:1310
        _put(v,'b',None,_d(1.477))  # WRF:module_cam_mp_microp_aero.F:1311
        _put(v,'c',None,_d(131.74))  # WRF:module_cam_mp_microp_aero.F:1312
        _put(v,'rhw',None,(v['a'] * v['tc'] * v['tc'] + v['b'] * v['tc'] + v['c']) * _d(0.01))  # WRF:module_cam_mp_microp_aero.F:1313
        _put(v,'subgrid',None,_d(1.2))  # WRF:module_cam_mp_microp_aero.F:1315
        _outer4=dict(v)
        def _yes4(_carry):
            v=dict(_outer4);v.update(zip(('a', 'b', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm'),_carry))
            _put(v,'a',None,_neg(_d(1.4938)) * xp.log(v['soot_num'] + v['dst_num']) + _d(12.884))  # WRF:module_cam_mp_microp_aero.F:1319
            _put(v,'b',None,_neg(_d(10.41)) * xp.log(v['soot_num'] + v['dst_num']) - _d(67.69))  # WRF:module_cam_mp_microp_aero.F:1320
            _put(v,'regm',None,v['a'] * xp.log(v['wbar']) + v['b'])  # WRF:module_cam_mp_microp_aero.F:1321
            _outer5=dict(v)
            def _yes5(_carry):
                v=dict(_outer5);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                _outer6=dict(v)
                def _yes6(_carry):
                    v=dict(_outer6);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                    _call=hf(v['tc'],v['wbar'],v['relhum'],v['subgrid'],v['so4_num'],v['nihf'],_globals=_globals,xp=xp)
                    _put(v,'nihf',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:1325
                    _put(v,'niimm',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1326
                    _put(v,'nidep',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1327
                    _put(v,'n1',None,v['nihf'])  # WRF:module_cam_mp_microp_aero.F:1328
                    return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
                def _no6(_carry):
                    v=dict(_outer6);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                    _call=hetero(v['tc'],v['wbar'],v['soot_num'] + v['dst_num'],v['niimm'],v['nidep'],_globals=_globals,xp=xp)
                    _put(v,'niimm',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:1330
                    _put(v,'nidep',None,_call[1])  # WRF:module_cam_mp_microp_aero.F:1330
                    _put(v,'nihf',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1331
                    _put(v,'n1',None,v['niimm'] + v['nidep'])  # WRF:module_cam_mp_microp_aero.F:1332
                    return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
                _result=_cond(xp.logical_and(v['tc'] < _neg(_d(40.0)), v['wbar'] > _d(1.0)),_yes6,_no6,tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm')),xp)  # WRF:module_cam_mp_microp_aero.F:1324
                v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_result))
                return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
            def _no5(_carry):
                v=dict(_outer5);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                _outer7=dict(v)
                def _yes7(_carry):
                    v=dict(_outer7);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                    _call=hf(v['tc'],v['wbar'],v['relhum'],v['subgrid'],v['so4_num'],v['nihf'],_globals=_globals,xp=xp)
                    _put(v,'nihf',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:1335
                    _put(v,'niimm',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1336
                    _put(v,'nidep',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1337
                    _put(v,'n1',None,v['nihf'])  # WRF:module_cam_mp_microp_aero.F:1338
                    return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
                def _no7(_carry):
                    v=dict(_outer7);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                    _outer8=dict(v)
                    def _yes8(_carry):
                        v=dict(_outer8);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                        _call=hf(v['tc'],v['wbar'],v['relhum'],v['subgrid'],v['so4_num'],v['nihf'],_globals=_globals,xp=xp)
                        _put(v,'nihf',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:1341
                        _put(v,'niimm',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1342
                        _put(v,'nidep',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1343
                        _put(v,'n1',None,v['nihf'])  # WRF:module_cam_mp_microp_aero.F:1344
                        return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
                    def _no8(_carry):
                        v=dict(_outer8);v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_carry))
                        _call=hf(v['regm'] - _d(5.0),v['wbar'],v['relhum'],v['subgrid'],v['so4_num'],v['nihf'],_globals=_globals,xp=xp)
                        _put(v,'nihf',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:1347
                        _call=hetero(v['regm'],v['wbar'],v['soot_num'] + v['dst_num'],v['niimm'],v['nidep'],_globals=_globals,xp=xp)
                        _put(v,'niimm',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:1348
                        _put(v,'nidep',None,_call[1])  # WRF:module_cam_mp_microp_aero.F:1348
                        _outer9=dict(v)
                        def _yes9(_carry):
                            v=dict(_outer9);v.update(zip(('n1',),_carry))
                            _put(v,'n1',None,v['nihf'])  # WRF:module_cam_mp_microp_aero.F:1351
                            return tuple(v[_n] for _n in ('n1',))
                        def _no9(_carry):
                            v=dict(_outer9);v.update(zip(('n1',),_carry))
                            _put(v,'n1',None,(v['niimm'] + v['nidep']) * _pow((v['niimm'] + v['nidep']) / v['nihf'], (v['tc'] - v['regm']) / _d(5.0), xp))  # WRF:module_cam_mp_microp_aero.F:1353
                            return tuple(v[_n] for _n in ('n1',))
                        _result=_cond(v['nihf'] <= v['niimm'] + v['nidep'],_yes9,_no9,tuple(v[_n] for _n in ('n1',)),xp)  # WRF:module_cam_mp_microp_aero.F:1350
                        v.update(zip(('n1',),_result))
                        return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
                    _result=_cond(xp.logical_and(v['tc'] < _neg(_d(40.0)), v['wbar'] > _d(1.0)),_yes8,_no8,tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm')),xp)  # WRF:module_cam_mp_microp_aero.F:1340
                    v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_result))
                    return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
                _result=_cond(v['tc'] < v['regm'] - _d(5.0),_yes7,_no7,tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm')),xp)  # WRF:module_cam_mp_microp_aero.F:1334
                v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_result))
                return tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm'))
            _result=_cond(v['tc'] > v['regm'],_yes5,_no5,tuple(v[_n] for _n in ('n1', 'nidep', 'nihf', 'niimm')),xp)  # WRF:module_cam_mp_microp_aero.F:1323
            v.update(zip(('n1', 'nidep', 'nihf', 'niimm'),_result))
            _put(v,'ni',None,v['n1'])  # WRF:module_cam_mp_microp_aero.F:1358
            return tuple(v[_n] for _n in ('a', 'b', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm'))
        def _no4(_carry):
            v=dict(_outer4);v.update(zip(('a', 'b', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm'),_carry))
            return tuple(v[_n] for _n in ('a', 'b', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm'))
        _result=_cond(xp.logical_and(v['tc'] <= _neg(_d(35.0)), v['relhum'] * polysvp(v['tair'], 0, xp=xp, _globals=_globals) / polysvp(v['tair'], 1, xp=xp, _globals=_globals) * v['subgrid'] >= _d(1.2)),_yes4,_no4,tuple(v[_n] for _n in ('a', 'b', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm')),xp)  # WRF:module_cam_mp_microp_aero.F:1317
        v.update(zip(('a', 'b', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm'),_result))
        return tuple(v[_n] for _n in ('a', 'b', 'c', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm', 'rhw', 'subgrid'))
    def _no3(_carry):
        v=dict(_outer3);v.update(zip(('a', 'b', 'c', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm', 'rhw', 'subgrid'),_carry))
        return tuple(v[_n] for _n in ('a', 'b', 'c', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm', 'rhw', 'subgrid'))
    _result=_cond(xp.logical_and(xp.logical_and(v['so4_num'] >= _d(1e-10), v['soot_num'] + v['dst_num'] >= _d(1e-10)), v['cldn'] > _d(0.0)),_yes3,_no3,tuple(v[_n] for _n in ('a', 'b', 'c', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm', 'rhw', 'subgrid')),xp)  # WRF:module_cam_mp_microp_aero.F:1306
    v.update(zip(('a', 'b', 'c', 'n1', 'ni', 'nidep', 'nihf', 'niimm', 'regm', 'rhw', 'subgrid'),_result))
    v['_jump']=xp.where(v['_jump']==1100,_int(0),v['_jump'])
    _outer10=dict(v)
    def _yes10(_carry):
        v=dict(_outer10);v.update(zip(('deles', 'esi', 'esl', 'nimey'),_carry))
        _put(v,'esl',None,polysvp(v['tair'], 0, xp=xp, _globals=_globals))  # WRF:module_cam_mp_microp_aero.F:1366
        _put(v,'esi',None,polysvp(v['tair'], 1, xp=xp, _globals=_globals))  # WRF:module_cam_mp_microp_aero.F:1367
        _put(v,'deles',None,v['esl'] - v['esi'])  # WRF:module_cam_mp_microp_aero.F:1368
        _put(v,'nimey',None,_d(0.001) * xp.exp(_d(12.96) * v['deles'] / v['esi'] - _d(0.639)))  # WRF:module_cam_mp_microp_aero.F:1369
        return tuple(v[_n] for _n in ('deles', 'esi', 'esl', 'nimey'))
    def _no10(_carry):
        v=dict(_outer10);v.update(zip(('deles', 'esi', 'esl', 'nimey'),_carry))
        _put(v,'nimey',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1371
        return tuple(v[_n] for _n in ('deles', 'esi', 'esl', 'nimey'))
    _result=_cond(xp.logical_and(xp.logical_and(v['tc'] < _d(0.0), v['tc'] > _neg(_d(37.0))), v['qc'] > _d(1e-12)),_yes10,_no10,tuple(v[_n] for _n in ('deles', 'esi', 'esl', 'nimey')),xp)  # WRF:module_cam_mp_microp_aero.F:1365
    v.update(zip(('deles', 'esi', 'esl', 'nimey'),_result))
    _put(v,'nuci',None,v['ni'] + v['nimey'])  # WRF:module_cam_mp_microp_aero.F:1374
    _outer11=dict(v)
    def _yes11(_carry):
        v=dict(_outer11);v.update(zip(('nuci',),_carry))
        _put(v,'nuci',None,_d(0.0))  # WRF:module_cam_mp_microp_aero.F:1384
        return tuple(v[_n] for _n in ('nuci',))
    def _no11(_carry):
        v=dict(_outer11);v.update(zip(('nuci',),_carry))
        return tuple(v[_n] for _n in ('nuci',))
    _result=_cond(xp.logical_or(v['nuci'] > _d(9999.0), v['nuci'] < _d(0.0)),_yes11,_no11,tuple(v[_n] for _n in ('nuci',)),xp)  # WRF:module_cam_mp_microp_aero.F:1375
    v.update(zip(('nuci',),_result))
    _put(v,'nuci',None,v['nuci'] * _d(1000000.0) / v['rhoair'])  # WRF:module_cam_mp_microp_aero.F:1387
    _put(v,'onimey',None,v['nimey'] * _d(1000000.0) / v['rhoair'])  # WRF:module_cam_mp_microp_aero.F:1388
    _put(v,'onidep',None,v['nidep'] * _d(1000000.0) / v['rhoair'])  # WRF:module_cam_mp_microp_aero.F:1389
    _put(v,'oniimm',None,v['niimm'] * _d(1000000.0) / v['rhoair'])  # WRF:module_cam_mp_microp_aero.F:1390
    _put(v,'onihf',None,v['nihf'] * _d(1000000.0) / v['rhoair'])  # WRF:module_cam_mp_microp_aero.F:1391
    return (v['nuci'],v['onihf'],v['oniimm'],v['onidep'],v['onimey'],)

ARGUMENTS={'hf': ['t', 'ww', 'rh', 'subgrid', 'na', 'ni'], 'hetero': ['t', 'ww', 'ns', 'nis', 'nid'], 'nucleati': ['wbar', 'tair', 'relhum', 'cldn', 'qc', 'nfice', 'rhoair', 'qaerpt', 'dgnum', 'ptype', 'naer_all', 'nuci', 'onihf', 'oniimm', 'onidep', 'onimey']}
OUTPUTS={'hf': ['ni'], 'hetero': ['nis', 'nid'], 'nucleati': ['nuci', 'onihf', 'oniimm', 'onidep', 'onimey']}
VARIABLES={'hf': {'t': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ww': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'rh': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'subgrid': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'na': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ni': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'a1_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a21_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a22_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b1_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b21_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b22_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a2_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b2_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c1_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c2_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k1_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k2_fast': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a1_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a2_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b1_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b2_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b3_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c1_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c2_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k1_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k2_slow': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'regm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'hetero': {'t': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ww': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ns': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'nis': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'nid': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'a11': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a12': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a21': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a22': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b11': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b12': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b21': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b22': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'nucleati': {'ptype': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'naer_all': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wbar': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'tair': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'relhum': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldn': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qc': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'nfice': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhoair': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qaerpt': {'kind': 'f64', 'dims': ['pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dgnum': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'nuci': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'onihf': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'oniimm': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'onidep': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'onimey': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'so4_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'soot_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dst1_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dst2_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dst3_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dst4_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dst_num': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nihf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'niimm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nidep': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nimey': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'n1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ni': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'a': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'b': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'c': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'regm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'esl': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'esi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'deles': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dst_scale': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'subgrid': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dmc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ssmc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}}
