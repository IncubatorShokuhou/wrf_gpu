"""Generated offline from pristine WRF CAM-MG; see proofs/s2_cammg/port.py.
Qualified against original WRF public/staged calls; native GPU qualification pending.
"""
import jax.numpy as jnp
from ._support import _r,_d,_int,_get,_put,_slice,_loop,_cond,_cast,_dtype,_pow,_powi,_retry,_zeros,_literal_array,_static,_neg

def polysvp(t,type,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('wv_saturation',{}))
    v['t']=t
    v['type']=type
    v['_jump']=_int(0)
    v['polysvp']=_zeros((),'f64',xp)
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('polysvp',),_carry))
        _put(v,'polysvp',None,_pow(_d(10.0), _neg(_d(9.09718)) * (_d(273.16) / v['t'] - _d(1.0)) - _d(3.56654) * xp.log10(_d(273.16) / v['t']) + _d(0.876793) * (_d(1.0) - v['t'] / _d(273.16)) + xp.log10(_d(6.1071)), xp) * _d(100.0))  # WRF:module_cam_wv_saturation.F:737
        return tuple(v[_n] for _n in ('polysvp',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('polysvp',),_carry))
        return tuple(v[_n] for _n in ('polysvp',))
    _result=_cond(v['type'] == 1,_yes1,_no1,tuple(v[_n] for _n in ('polysvp',)),xp)  # WRF:module_cam_wv_saturation.F:733
    v.update(zip(('polysvp',),_result))
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('polysvp',),_carry))
        _put(v,'polysvp',None,_pow(_d(10.0), _neg(_d(7.90298)) * (_d(373.16) / v['t'] - _d(1.0)) + _d(5.02808) * xp.log10(_d(373.16) / v['t']) - _d(1.3816e-07) * (_pow(_d(10.0), _d(11.344) * (_d(1.0) - v['t'] / _d(373.16)), xp) - _d(1.0)) + _d(0.0081328) * (_pow(_d(10.0), _neg(_d(3.49149)) * (_d(373.16) / v['t'] - _d(1.0)), xp) - _d(1.0)) + xp.log10(_d(1013.246)), xp) * _d(100.0))  # WRF:module_cam_wv_saturation.F:746
        return tuple(v[_n] for _n in ('polysvp',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('polysvp',),_carry))
        return tuple(v[_n] for _n in ('polysvp',))
    _result=_cond(v['type'] == 0,_yes2,_no2,tuple(v[_n] for _n in ('polysvp',)),xp)  # WRF:module_cam_wv_saturation.F:745
    v.update(zip(('polysvp',),_result))
    return v['polysvp']

def cal_cldfra_mp_1d(cldl,cldi,qqv,qqc,qqi,tt,pp,xland_pt,snowh_pt,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('module_mp_cammgmp_driver',{}))
    v['cldl']=cldl
    v['cldi']=cldi
    v['qqv']=qqv
    v['qqc']=qqc
    v['qqi']=qqi
    v['tt']=tt
    v['pp']=pp
    v['xland_pt']=xland_pt
    v['snowh_pt']=snowh_pt
    v['_jump']=_int(0)
    v['rhl']=_zeros((),'f64',xp)
    v['dv']=_zeros((),'f64',xp)
    v['esl']=_zeros((),'f64',xp)
    v['omeps']=_zeros((),'f64',xp)
    v['cldrh']=_zeros((),'f64',xp)
    v['qvl']=_zeros((),'f64',xp)
    v['cldcr']=_zeros((),'f64',xp)
    v['esi']=_zeros((),'f64',xp)
    v['qvi']=_zeros((),'f64',xp)
    v['rhi']=_zeros((),'f64',xp)
    v['rhdif']=_zeros((),'f64',xp)
    v['icimr']=_zeros((),'f64',xp)
    v['qv']=_zeros((),'f64',xp)
    v['qc']=_zeros((),'f64',xp)
    v['qi']=_zeros((),'f64',xp)
    _put(v,'qv',None,xp.maximum(v['qqv'], _d(0.0)))  # WRF:module_mp_cammgmp_driver.F:1671
    _put(v,'qc',None,xp.maximum(v['qqc'], _d(0.0)))  # WRF:module_mp_cammgmp_driver.F:1672
    _put(v,'qi',None,xp.maximum(v['qqi'], _d(0.0)))  # WRF:module_mp_cammgmp_driver.F:1673
    _put(v,'cldi',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1674
    _put(v,'cldl',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1675
    _put(v,'cldrh',None,_d(1.0))  # WRF:module_mp_cammgmp_driver.F:1678
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('cldcr',),_carry))
        _outer2=dict(v)
        def _yes2(_carry):
            v=dict(_outer2);v.update(zip(('cldcr',),_carry))
            _put(v,'cldcr',None,_d(0.7875))  # WRF:module_mp_cammgmp_driver.F:1682
            return tuple(v[_n] for _n in ('cldcr',))
        def _no2(_carry):
            v=dict(_outer2);v.update(zip(('cldcr',),_carry))
            _put(v,'cldcr',None,_d(0.8875))  # WRF:module_mp_cammgmp_driver.F:1684
            return tuple(v[_n] for _n in ('cldcr',))
        _result=_cond(xp.logical_and(v['xland_pt'] < _d(1.1), v['snowh_pt'] < _d(0.001)),_yes2,_no2,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1681
        v.update(zip(('cldcr',),_result))
        return tuple(v[_n] for _n in ('cldcr',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('cldcr',),_carry))
        _outer3=dict(v)
        def _yes3(_carry):
            v=dict(_outer3);v.update(zip(('cldcr',),_carry))
            _put(v,'cldcr',None,_d(0.8))  # WRF:module_mp_cammgmp_driver.F:1687
            return tuple(v[_n] for _n in ('cldcr',))
        def _no3(_carry):
            v=dict(_outer3);v.update(zip(('cldcr',),_carry))
            _put(v,'cldcr',None,(_d(0.8875) - _d(0.8)) / _d(30000.0) * (v['pp'] - _d(40000.0)) + _d(0.8))  # WRF:module_mp_cammgmp_driver.F:1689
            return tuple(v[_n] for _n in ('cldcr',))
        _result=_cond(v['pp'] <= _d(40000.0),_yes3,_no3,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1686
        v.update(zip(('cldcr',),_result))
        return tuple(v[_n] for _n in ('cldcr',))
    _result=_cond(v['pp'] >= _d(70000.0),_yes1,_no1,tuple(v[_n] for _n in ('cldcr',)),xp)  # WRF:module_mp_cammgmp_driver.F:1680
    v.update(zip(('cldcr',),_result))
    _put(v,'cldcr',None,xp.minimum(xp.maximum(_d(0.0), v['cldcr']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:1692
    _put(v,'dv',None,v['cldrh'] - v['cldcr'])  # WRF:module_mp_cammgmp_driver.F:1694
    _put(v,'omeps',None,_d(1.0) - _d(0.622))  # WRF:module_mp_cammgmp_driver.F:1695
    _put(v,'rhdif',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1696
    _put(v,'icimr',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1697
    _put(v,'esl',None,polysvp(v['tt'], 0, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:1699
    _put(v,'qvl',None,xp.maximum(_d(1e-12), v['epsqs'] * v['esl'] / (v['pp'] - (_d(1.0) - v['epsqs']) * v['esl'])))  # WRF:module_mp_cammgmp_driver.F:1700
    _put(v,'rhl',None,(v['qv'] + v['qc']) / v['qvl'])  # WRF:module_mp_cammgmp_driver.F:1701
    _put(v,'rhl',None,xp.minimum(_d(1.1), xp.maximum(_d(0.0), v['rhl'])))  # WRF:module_mp_cammgmp_driver.F:1702
    _outer4=dict(v)
    def _yes4(_carry):
        v=dict(_outer4);v.update(zip(('cldl',),_carry))
        _put(v,'cldl',None,_d(1.0))  # WRF:module_mp_cammgmp_driver.F:1704
        return tuple(v[_n] for _n in ('cldl',))
    def _no4(_carry):
        v=dict(_outer4);v.update(zip(('cldl',),_carry))
        _outer5=dict(v)
        def _yes5(_carry):
            v=dict(_outer5);v.update(zip(('cldl',),_carry))
            _put(v,'cldl',None,_d(1.0) - _pow(_neg(_d(3.0)) / xp.sqrt(_d(2.0)) * (v['rhl'] - v['cldrh']) / v['dv'], _d(2.0) / _d(3.0), xp))  # WRF:module_mp_cammgmp_driver.F:1706
            return tuple(v[_n] for _n in ('cldl',))
        def _no5(_carry):
            v=dict(_outer5);v.update(zip(('cldl',),_carry))
            _outer6=dict(v)
            def _yes6(_carry):
                v=dict(_outer6);v.update(zip(('cldl',),_carry))
                _put(v,'cldl',None,_d(4.0) * _pow(xp.cos(_d(1.0) / _d(3.0) * (xp.arccos(_d(3.0) / _d(2.0) / xp.sqrt(_d(2.0)) * (_d(1.0) + (v['rhl'] - v['cldrh']) / v['dv'])) - _d(2.0) * _d(3.14159))), _d(2.0), xp))  # WRF:module_mp_cammgmp_driver.F:1708
                return tuple(v[_n] for _n in ('cldl',))
            def _no6(_carry):
                v=dict(_outer6);v.update(zip(('cldl',),_carry))
                _outer7=dict(v)
                def _yes7(_carry):
                    v=dict(_outer7);v.update(zip(('cldl',),_carry))
                    _put(v,'cldl',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1711
                    return tuple(v[_n] for _n in ('cldl',))
                def _no7(_carry):
                    v=dict(_outer7);v.update(zip(('cldl',),_carry))
                    return tuple(v[_n] for _n in ('cldl',))
                _result=_cond(v['rhl'] <= v['cldrh'] - v['dv'],_yes7,_no7,tuple(v[_n] for _n in ('cldl',)),xp)  # WRF:module_mp_cammgmp_driver.F:1710
                v.update(zip(('cldl',),_result))
                return tuple(v[_n] for _n in ('cldl',))
            _result=_cond(xp.logical_and(v['rhl'] > v['cldrh'] - v['dv'], v['rhl'] <= v['cldrh'] - v['dv'] / _d(6.0)),_yes6,_no6,tuple(v[_n] for _n in ('cldl',)),xp)  # WRF:module_mp_cammgmp_driver.F:1707
            v.update(zip(('cldl',),_result))
            return tuple(v[_n] for _n in ('cldl',))
        _result=_cond(xp.logical_and(v['rhl'] > v['cldrh'] - v['dv'] / _d(6.0), v['rhl'] < _d(1.0)),_yes5,_no5,tuple(v[_n] for _n in ('cldl',)),xp)  # WRF:module_mp_cammgmp_driver.F:1705
        v.update(zip(('cldl',),_result))
        return tuple(v[_n] for _n in ('cldl',))
    _result=_cond(v['rhl'] >= _d(1.0),_yes4,_no4,tuple(v[_n] for _n in ('cldl',)),xp)  # WRF:module_mp_cammgmp_driver.F:1703
    v.update(zip(('cldl',),_result))
    _put(v,'esi',None,polysvp(v['tt'], 1, xp=xp, _globals=_globals))  # WRF:module_mp_cammgmp_driver.F:1714
    _outer8=dict(v)
    def _yes8(_carry):
        v=dict(_outer8);v.update(zip(('esi',),_carry))
        _put(v,'esi',None,v['esl'])  # WRF:module_mp_cammgmp_driver.F:1715
        return tuple(v[_n] for _n in ('esi',))
    def _no8(_carry):
        v=dict(_outer8);v.update(zip(('esi',),_carry))
        return tuple(v[_n] for _n in ('esi',))
    _result=_cond(v['tt'] > v['tmelt'],_yes8,_no8,tuple(v[_n] for _n in ('esi',)),xp)  # WRF:module_mp_cammgmp_driver.F:1715
    v.update(zip(('esi',),_result))
    _put(v,'qvi',None,xp.maximum(_d(1e-12), v['epsqs'] * v['esi'] / (v['pp'] - (_d(1.0) - v['epsqs']) * v['esi'])))  # WRF:module_mp_cammgmp_driver.F:1716
    _put(v,'rhi',None,(v['qv'] + v['qi']) / v['qvi'])  # WRF:module_mp_cammgmp_driver.F:1717
    _put(v,'rhi',None,xp.minimum(_d(1.1), xp.maximum(_d(0.0), v['rhi'])))  # WRF:module_mp_cammgmp_driver.F:1718
    _outer9=dict(v)
    def _yes9(_carry):
        v=dict(_outer9);v.update(zip(('cldi', 'icimr', 'rhdif'),_carry))
        _put(v,'rhdif',None,(v['rhi'] - _d(0.8)) / _d(0.3))  # WRF:module_mp_cammgmp_driver.F:1721
        _put(v,'cldi',None,xp.minimum(_d(1.0), _pow(xp.maximum(v['rhdif'], _d(0.0)), _d(2.0), xp)))  # WRF:module_mp_cammgmp_driver.F:1722
        _outer10=dict(v)
        def _yes10(_carry):
            v=dict(_outer10);v.update(zip(('cldi',),_carry))
            _put(v,'cldi',None,_d(0.0))  # WRF:module_mp_cammgmp_driver.F:1724
            return tuple(v[_n] for _n in ('cldi',))
        def _no10(_carry):
            v=dict(_outer10);v.update(zip(('cldi',),_carry))
            _put(v,'cldi',None,xp.maximum(_d(0.0001), v['cldi']))  # WRF:module_mp_cammgmp_driver.F:1726
            return tuple(v[_n] for _n in ('cldi',))
        _result=_cond(v['qi'] < _d(1e-12),_yes10,_no10,tuple(v[_n] for _n in ('cldi',)),xp)  # WRF:module_mp_cammgmp_driver.F:1723
        v.update(zip(('cldi',),_result))
        _outer11=dict(v)
        def _yes11(_carry):
            v=dict(_outer11);v.update(zip(('cldi', 'icimr'),_carry))
            _put(v,'icimr',None,v['qi'] / xp.maximum(v['cldi'], _d(1e-20)))  # WRF:module_mp_cammgmp_driver.F:1730
            _outer12=dict(v)
            def _yes12(_carry):
                v=dict(_outer12);v.update(zip(('cldi',),_carry))
                _put(v,'cldi',None,xp.maximum(_d(0.0), xp.minimum(_d(1.0), v['qi'] / _d(1e-07))))  # WRF:module_mp_cammgmp_driver.F:1732
                return tuple(v[_n] for _n in ('cldi',))
            def _no12(_carry):
                v=dict(_outer12);v.update(zip(('cldi',),_carry))
                return tuple(v[_n] for _n in ('cldi',))
            _result=_cond(v['icimr'] < _d(1e-07),_yes12,_no12,tuple(v[_n] for _n in ('cldi',)),xp)  # WRF:module_mp_cammgmp_driver.F:1731
            v.update(zip(('cldi',),_result))
            _outer13=dict(v)
            def _yes13(_carry):
                v=dict(_outer13);v.update(zip(('cldi',),_carry))
                _put(v,'cldi',None,xp.maximum(_d(0.0), xp.minimum(_d(1.0), v['qi'] / _d(0.005))))  # WRF:module_mp_cammgmp_driver.F:1735
                return tuple(v[_n] for _n in ('cldi',))
            def _no13(_carry):
                v=dict(_outer13);v.update(zip(('cldi',),_carry))
                return tuple(v[_n] for _n in ('cldi',))
            _result=_cond(v['icimr'] > _d(0.005),_yes13,_no13,tuple(v[_n] for _n in ('cldi',)),xp)  # WRF:module_mp_cammgmp_driver.F:1734
            v.update(zip(('cldi',),_result))
            return tuple(v[_n] for _n in ('cldi', 'icimr'))
        def _no11(_carry):
            v=dict(_outer11);v.update(zip(('cldi', 'icimr'),_carry))
            return tuple(v[_n] for _n in ('cldi', 'icimr'))
        _result=_cond(v['qi'] >= _d(1e-12),_yes11,_no11,tuple(v[_n] for _n in ('cldi', 'icimr')),xp)  # WRF:module_mp_cammgmp_driver.F:1729
        v.update(zip(('cldi', 'icimr'),_result))
        return tuple(v[_n] for _n in ('cldi', 'icimr', 'rhdif'))
    def _no9(_carry):
        v=dict(_outer9);v.update(zip(('cldi', 'icimr', 'rhdif'),_carry))
        return tuple(v[_n] for _n in ('cldi', 'icimr', 'rhdif'))
    _result=_cond(v['tt'] < _d(273.15),_yes9,_no9,tuple(v[_n] for _n in ('cldi', 'icimr', 'rhdif')),xp)  # WRF:module_mp_cammgmp_driver.F:1720
    v.update(zip(('cldi', 'icimr', 'rhdif'),_result))
    _put(v,'cldl',None,xp.minimum(xp.maximum(_d(0.0), v['cldl']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:1740
    _put(v,'cldi',None,xp.minimum(xp.maximum(_d(0.0), v['cldi']), _d(1.0)))  # WRF:module_mp_cammgmp_driver.F:1741
    return (v['cldl'],v['cldi'],)

def vqsatd_water(t,p,es,qs,gam,len,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('wv_saturation',{}))
    v['t']=t
    v['p']=p
    v['es']=es
    v['qs']=qs
    v['gam']=gam
    v['len']=len
    v['_jump']=_int(0)
    v['i']=_zeros((),'i32',xp)
    v['omeps']=_zeros((),'f64',xp)
    v['hltalt']=_zeros((),'f64',xp)
    v['hlatsb']=_zeros((),'f64',xp)
    v['hlatvp']=_zeros((),'f64',xp)
    v['desdt']=_zeros((),'f64',xp)
    _put(v,'omeps',None,_d(1.0) - v['epsqs'])  # WRF:module_cam_wv_saturation.F:675
    _outer1=dict(v)
    def _body1(_k,_carry):
        v=dict(_outer1);v.update(zip(('_jump', 'es', 'i', 'qs'),_carry));v['i']=_int(_k)
        _put(v,'es',(v['i'],),polysvp(_get(v['t'], (v['i'],)), 0, xp=xp, _globals=_globals))  # WRF:module_cam_wv_saturation.F:677
        _put(v,'qs',(v['i'],),v['epsqs'] * _get(v['es'], (v['i'],)) / (_get(v['p'], (v['i'],)) - v['omeps'] * _get(v['es'], (v['i'],))))  # WRF:module_cam_wv_saturation.F:681
        _put(v,'qs',(v['i'],),xp.minimum(_d(1.0), _get(v['qs'], (v['i'],))))  # WRF:module_cam_wv_saturation.F:686
        _outer2=dict(v)
        def _yes2(_carry):
            v=dict(_outer2);v.update(zip(('es', 'qs'),_carry))
            _put(v,'qs',(v['i'],),_d(1.0))  # WRF:module_cam_wv_saturation.F:689
            _put(v,'es',(v['i'],),_get(v['p'], (v['i'],)))  # WRF:module_cam_wv_saturation.F:690
            return tuple(v[_n] for _n in ('es', 'qs'))
        def _no2(_carry):
            v=dict(_outer2);v.update(zip(('es', 'qs'),_carry))
            return tuple(v[_n] for _n in ('es', 'qs'))
        _result=_cond(_get(v['qs'], (v['i'],)) < _d(0.0),_yes2,_no2,tuple(v[_n] for _n in ('es', 'qs')),xp)  # WRF:module_cam_wv_saturation.F:688
        v.update(zip(('es', 'qs'),_result))
        return tuple(v[_n] for _n in ('_jump', 'es', 'i', 'qs'))
    _result=_loop(1,v['len'],1,_body1,tuple(v[_n] for _n in ('_jump', 'es', 'i', 'qs')),xp,jump_index=0)
    v.update(zip(('_jump', 'es', 'i', 'qs'),_result))
    _outer3=dict(v)
    def _body3(_k,_carry):
        v=dict(_outer3);v.update(zip(('_jump', 'desdt', 'gam', 'hlatsb', 'hlatvp', 'hltalt', 'i'),_carry));v['i']=_int(_k)
        _put(v,'hlatvp',None,v['hlatv'] - _d(2369.0) * (_get(v['t'], (v['i'],)) - v['tmelt']))  # WRF:module_cam_wv_saturation.F:701
        _put(v,'hlatsb',None,v['hlatv'])  # WRF:module_cam_wv_saturation.F:702
        _outer4=dict(v)
        def _yes4(_carry):
            v=dict(_outer4);v.update(zip(('hltalt',),_carry))
            _put(v,'hltalt',None,v['hlatsb'])  # WRF:module_cam_wv_saturation.F:704
            return tuple(v[_n] for _n in ('hltalt',))
        def _no4(_carry):
            v=dict(_outer4);v.update(zip(('hltalt',),_carry))
            _put(v,'hltalt',None,v['hlatvp'])  # WRF:module_cam_wv_saturation.F:706
            return tuple(v[_n] for _n in ('hltalt',))
        _result=_cond(_get(v['t'], (v['i'],)) < v['tmelt'],_yes4,_no4,tuple(v[_n] for _n in ('hltalt',)),xp)  # WRF:module_cam_wv_saturation.F:703
        v.update(zip(('hltalt',),_result))
        _put(v,'desdt',None,v['hltalt'] * _get(v['es'], (v['i'],)) / (v['rgasv'] * _get(v['t'], (v['i'],)) * _get(v['t'], (v['i'],))))  # WRF:module_cam_wv_saturation.F:708
        _put(v,'gam',(v['i'],),v['hltalt'] * _get(v['qs'], (v['i'],)) * _get(v['p'], (v['i'],)) * v['desdt'] / (v['cp'] * _get(v['es'], (v['i'],)) * (_get(v['p'], (v['i'],)) - v['omeps'] * _get(v['es'], (v['i'],)))))  # WRF:module_cam_wv_saturation.F:709
        _outer5=dict(v)
        def _yes5(_carry):
            v=dict(_outer5);v.update(zip(('gam',),_carry))
            _put(v,'gam',(v['i'],),_d(0.0))  # WRF:module_cam_wv_saturation.F:710
            return tuple(v[_n] for _n in ('gam',))
        def _no5(_carry):
            v=dict(_outer5);v.update(zip(('gam',),_carry))
            return tuple(v[_n] for _n in ('gam',))
        _result=_cond(_get(v['qs'], (v['i'],)) == _d(1.0),_yes5,_no5,tuple(v[_n] for _n in ('gam',)),xp)  # WRF:module_cam_wv_saturation.F:710
        v.update(zip(('gam',),_result))
        return tuple(v[_n] for _n in ('_jump', 'desdt', 'gam', 'hlatsb', 'hlatvp', 'hltalt', 'i'))
    _result=_loop(1,v['len'],1,_body3,tuple(v[_n] for _n in ('_jump', 'desdt', 'gam', 'hlatsb', 'hlatvp', 'hltalt', 'i')),xp,jump_index=0)
    v.update(zip(('_jump', 'desdt', 'gam', 'hlatsb', 'hlatvp', 'hltalt', 'i'),_result))
    return (v['es'],v['qs'],v['gam'],)

ARGUMENTS={'polysvp': ['t', 'type'], 'cal_cldfra_mp_1d': ['cldl', 'cldi', 'qqv', 'qqc', 'qqi', 'tt', 'pp', 'xland_pt', 'snowh_pt'], 'vqsatd_water': ['t', 'p', 'es', 'qs', 'gam', 'len']}
OUTPUTS={'polysvp': [], 'cal_cldfra_mp_1d': ['cldl', 'cldi'], 'vqsatd_water': ['es', 'qs', 'gam']}
VARIABLES={'polysvp': {'dum': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 't': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'polysvp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'type': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'cal_cldfra_mp_1d': {'qqv': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qqc': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qqi': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'tt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pp': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'xland_pt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'snowh_pt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldl': {'kind': 'f64', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldi': {'kind': 'f64', 'dims': [], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhl': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'esl': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'omeps': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldrh': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qvl': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldcr': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'esi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qvi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhdif': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'icimr': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'vqsatd_water': {'len': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 't': {'kind': 'f64', 'dims': ['len'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'p': {'kind': 'f64', 'dims': ['len'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'es': {'kind': 'f64', 'dims': ['len'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'qs': {'kind': 'f64', 'dims': ['len'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'gam': {'kind': 'f64', 'dims': ['len'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'omeps': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hltalt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hlatsb': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hlatvp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'desdt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}}
