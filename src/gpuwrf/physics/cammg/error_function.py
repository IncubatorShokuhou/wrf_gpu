"""Generated offline from pristine WRF CAM-MG; see proofs/s2_cammg/port.py.
Qualified against original WRF public/staged calls; native GPU qualification pending.
"""
import jax.numpy as jnp
from ._support import _r,_d,_int,_get,_put,_slice,_loop,_cond,_cast,_dtype,_pow,_powi,_retry,_zeros,_literal_array,_static,_neg

def calerf_r8(arg,result,jint,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('error_function',{}))
    v['arg']=arg
    v['result']=result
    v['jint']=jint
    v['_jump']=_int(0)
    v['i']=_zeros((),'i32',xp)
    v['x']=_zeros((),'f64',xp)
    v['y']=_zeros((),'f64',xp)
    v['ysq']=_zeros((),'f64',xp)
    v['xnum']=_zeros((),'f64',xp)
    v['xden']=_zeros((),'f64',xp)
    v['del']=_zeros((),'f64',xp)
    v['zero']=_cast(_d(0.0),'f64',xp)
    v['four']=_cast(_d(4.0),'f64',xp)
    v['one']=_cast(_d(1.0),'f64',xp)
    v['half']=_cast(_d(0.5),'f64',xp)
    v['two']=_cast(_d(2.0),'f64',xp)
    v['sqrpi']=_cast(_d(0.5641895835477563),'f64',xp)
    v['thresh']=_cast(_d(0.46875),'f64',xp)
    v['sixten']=_cast(_d(16.0),'f64',xp)
    v['xinf']=_cast(_d(1.79e+308),'f64',xp)
    v['xneg']=_cast(_neg(_d(26.628)),'f64',xp)
    v['xsmall']=_cast(_d(1.11e-16),'f64',xp)
    v['xbig']=_cast(_d(26.543),'f64',xp)
    v['xhuge']=_cast(_d(67100000.0),'f64',xp)
    v['xmax']=_cast(_d(2.53e+307),'f64',xp)
    v['a']=_literal_array([_d(3.1611237438705655),_d(113.86415415105016),_d(377.485237685302),_d(3209.3775891384694),_d(0.18577770618460315)],'f64',xp)
    v['b']=_literal_array([_d(23.601290952344122),_d(244.02463793444417),_d(1282.6165260773723),_d(2844.236833439171)],'f64',xp)
    v['c']=_literal_array([_d(0.5641884969886701),_d(8.883149794388377),_d(66.11919063714163),_d(298.6351381974001),_d(881.952221241769),_d(1712.0476126340707),_d(2051.0783778260716),_d(1230.3393547979972),_d(2.1531153547440383e-08)],'f64',xp)
    v['d']=_literal_array([_d(15.744926110709835),_d(117.6939508913125),_d(537.1811018620099),_d(1621.3895745666903),_d(3290.7992357334597),_d(4362.619090143247),_d(3439.3676741437216),_d(1230.3393548037495)],'f64',xp)
    v['p']=_literal_array([_d(0.30532663496123236),_d(0.36034489994980445),_d(0.12578172611122926),_d(0.016083785148742275),_d(0.0006587491615298378),_d(0.016315387137302097)],'f64',xp)
    v['q']=_literal_array([_d(2.568520192289822),_d(1.8729528499234604),_d(0.5279051029514285),_d(0.06051834131244132),_d(0.0023352049762686918)],'f64',xp)
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('x',),_carry))
        _put(v,'x',None,v['arg'])  # WRF:module_cam_error_function.F:248
        return tuple(v[_n] for _n in ('x',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('x',),_carry))
        return tuple(v[_n] for _n in ('x',))
    _result=_cond(v['_jump'] == 0,_yes1,_no1,tuple(v[_n] for _n in ('x',)),xp)  # WRF:module_cam_error_function.F:248
    v.update(zip(('x',),_result))
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('y',),_carry))
        _put(v,'y',None,xp.abs(v['x']))  # WRF:module_cam_error_function.F:249
        return tuple(v[_n] for _n in ('y',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('y',),_carry))
        return tuple(v[_n] for _n in ('y',))
    _result=_cond(v['_jump'] == 0,_yes2,_no2,tuple(v[_n] for _n in ('y',)),xp)  # WRF:module_cam_error_function.F:249
    v.update(zip(('y',),_result))
    _outer3=dict(v)
    def _yes3(_carry):
        v=dict(_outer3);v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_carry))
        _outer4=dict(v)
        def _yes4(_carry):
            v=dict(_outer4);v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_carry))
            _put(v,'ysq',None,v['zero'])  # WRF:module_cam_error_function.F:254
            _outer5=dict(v)
            def _yes5(_carry):
                v=dict(_outer5);v.update(zip(('ysq',),_carry))
                _put(v,'ysq',None,v['y'] * v['y'])  # WRF:module_cam_error_function.F:255
                return tuple(v[_n] for _n in ('ysq',))
            def _no5(_carry):
                v=dict(_outer5);v.update(zip(('ysq',),_carry))
                return tuple(v[_n] for _n in ('ysq',))
            _result=_cond(v['y'] > v['xsmall'],_yes5,_no5,tuple(v[_n] for _n in ('ysq',)),xp)  # WRF:module_cam_error_function.F:255
            v.update(zip(('ysq',),_result))
            _put(v,'xnum',None,_get(v['a'], (5,)) * v['ysq'])  # WRF:module_cam_error_function.F:256
            _put(v,'xden',None,v['ysq'])  # WRF:module_cam_error_function.F:257
            _outer6=dict(v)
            def _body6(_k,_carry):
                v=dict(_outer6);v.update(zip(('_jump', 'i', 'xden', 'xnum'),_carry));v['i']=_int(_k)
                _put(v,'xnum',None,(v['xnum'] + _get(v['a'], (v['i'],))) * v['ysq'])  # WRF:module_cam_error_function.F:259
                _put(v,'xden',None,(v['xden'] + _get(v['b'], (v['i'],))) * v['ysq'])  # WRF:module_cam_error_function.F:260
                return tuple(v[_n] for _n in ('_jump', 'i', 'xden', 'xnum'))
            _result=_loop(1,3,1,_body6,tuple(v[_n] for _n in ('_jump', 'i', 'xden', 'xnum')),xp,jump_index=0)
            v.update(zip(('_jump', 'i', 'xden', 'xnum'),_result))
            _put(v,'result',None,v['x'] * (v['xnum'] + _get(v['a'], (4,))) / (v['xden'] + _get(v['b'], (4,))))  # WRF:module_cam_error_function.F:262
            _outer7=dict(v)
            def _yes7(_carry):
                v=dict(_outer7);v.update(zip(('result',),_carry))
                _put(v,'result',None,v['one'] - v['result'])  # WRF:module_cam_error_function.F:263
                return tuple(v[_n] for _n in ('result',))
            def _no7(_carry):
                v=dict(_outer7);v.update(zip(('result',),_carry))
                return tuple(v[_n] for _n in ('result',))
            _result=_cond(v['jint'] != 0,_yes7,_no7,tuple(v[_n] for _n in ('result',)),xp)  # WRF:module_cam_error_function.F:263
            v.update(zip(('result',),_result))
            _outer8=dict(v)
            def _yes8(_carry):
                v=dict(_outer8);v.update(zip(('result',),_carry))
                _put(v,'result',None,xp.exp(v['ysq']) * v['result'])  # WRF:module_cam_error_function.F:264
                return tuple(v[_n] for _n in ('result',))
            def _no8(_carry):
                v=dict(_outer8);v.update(zip(('result',),_carry))
                return tuple(v[_n] for _n in ('result',))
            _result=_cond(v['jint'] == 2,_yes8,_no8,tuple(v[_n] for _n in ('result',)),xp)  # WRF:module_cam_error_function.F:264
            v.update(zip(('result',),_result))
            v['_jump']=_int(80)  # WRF:module_cam_error_function.F:265
            return tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'))
        def _no4(_carry):
            v=dict(_outer4);v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_carry))
            _outer9=dict(v)
            def _yes9(_carry):
                v=dict(_outer9);v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_carry))
                _put(v,'xnum',None,_get(v['c'], (9,)) * v['y'])  # WRF:module_cam_error_function.F:270
                _put(v,'xden',None,v['y'])  # WRF:module_cam_error_function.F:271
                _outer10=dict(v)
                def _body10(_k,_carry):
                    v=dict(_outer10);v.update(zip(('_jump', 'i', 'xden', 'xnum'),_carry));v['i']=_int(_k)
                    _put(v,'xnum',None,(v['xnum'] + _get(v['c'], (v['i'],))) * v['y'])  # WRF:module_cam_error_function.F:273
                    _put(v,'xden',None,(v['xden'] + _get(v['d'], (v['i'],))) * v['y'])  # WRF:module_cam_error_function.F:274
                    return tuple(v[_n] for _n in ('_jump', 'i', 'xden', 'xnum'))
                _result=_loop(1,7,1,_body10,tuple(v[_n] for _n in ('_jump', 'i', 'xden', 'xnum')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'xden', 'xnum'),_result))
                _put(v,'result',None,(v['xnum'] + _get(v['c'], (8,))) / (v['xden'] + _get(v['d'], (8,))))  # WRF:module_cam_error_function.F:276
                _outer11=dict(v)
                def _yes11(_carry):
                    v=dict(_outer11);v.update(zip(('del', 'result', 'ysq'),_carry))
                    _put(v,'ysq',None,xp.trunc(v['y'] * v['sixten']) / v['sixten'])  # WRF:module_cam_error_function.F:278
                    _put(v,'del',None,(v['y'] - v['ysq']) * (v['y'] + v['ysq']))  # WRF:module_cam_error_function.F:279
                    _put(v,'result',None,xp.exp(_neg(v['ysq']) * v['ysq']) * xp.exp(_neg(v['del'])) * v['result'])  # WRF:module_cam_error_function.F:280
                    return tuple(v[_n] for _n in ('del', 'result', 'ysq'))
                def _no11(_carry):
                    v=dict(_outer11);v.update(zip(('del', 'result', 'ysq'),_carry))
                    return tuple(v[_n] for _n in ('del', 'result', 'ysq'))
                _result=_cond(v['jint'] != 2,_yes11,_no11,tuple(v[_n] for _n in ('del', 'result', 'ysq')),xp)  # WRF:module_cam_error_function.F:277
                v.update(zip(('del', 'result', 'ysq'),_result))
                return tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'))
            def _no9(_carry):
                v=dict(_outer9);v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_carry))
                _put(v,'result',None,v['zero'])  # WRF:module_cam_error_function.F:286
                _outer12=dict(v)
                def _yes12(_carry):
                    v=dict(_outer12);v.update(zip(('_jump', 'result'),_carry))
                    _outer13=dict(v)
                    def _yes13(_carry):
                        v=dict(_outer13);v.update(zip(('_jump',),_carry))
                        v['_jump']=_int(30)  # WRF:module_cam_error_function.F:288
                        return tuple(v[_n] for _n in ('_jump',))
                    def _no13(_carry):
                        v=dict(_outer13);v.update(zip(('_jump',),_carry))
                        return tuple(v[_n] for _n in ('_jump',))
                    _result=_cond(xp.logical_or(v['jint'] != 2, v['y'] >= v['xmax']),_yes13,_no13,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_error_function.F:288
                    v.update(zip(('_jump',),_result))
                    _outer14=dict(v)
                    def _yes14(_carry):
                        v=dict(_outer14);v.update(zip(('_jump', 'result'),_carry))
                        _put(v,'result',None,v['sqrpi'] / v['y'])  # WRF:module_cam_error_function.F:290
                        v['_jump']=_int(30)  # WRF:module_cam_error_function.F:291
                        return tuple(v[_n] for _n in ('_jump', 'result'))
                    def _no14(_carry):
                        v=dict(_outer14);v.update(zip(('_jump', 'result'),_carry))
                        return tuple(v[_n] for _n in ('_jump', 'result'))
                    _result=_cond(v['y'] >= v['xhuge'],_yes14,_no14,tuple(v[_n] for _n in ('_jump', 'result')),xp)  # WRF:module_cam_error_function.F:289
                    v.update(zip(('_jump', 'result'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'result'))
                def _no12(_carry):
                    v=dict(_outer12);v.update(zip(('_jump', 'result'),_carry))
                    return tuple(v[_n] for _n in ('_jump', 'result'))
                _result=_cond(v['y'] >= v['xbig'],_yes12,_no12,tuple(v[_n] for _n in ('_jump', 'result')),xp)  # WRF:module_cam_error_function.F:287
                v.update(zip(('_jump', 'result'),_result))
                _put(v,'ysq',None,v['one'] / (v['y'] * v['y']))  # WRF:module_cam_error_function.F:294
                _put(v,'xnum',None,_get(v['p'], (6,)) * v['ysq'])  # WRF:module_cam_error_function.F:295
                _put(v,'xden',None,v['ysq'])  # WRF:module_cam_error_function.F:296
                _outer15=dict(v)
                def _body15(_k,_carry):
                    v=dict(_outer15);v.update(zip(('_jump', 'i', 'xden', 'xnum'),_carry));v['i']=_int(_k)
                    _put(v,'xnum',None,(v['xnum'] + _get(v['p'], (v['i'],))) * v['ysq'])  # WRF:module_cam_error_function.F:298
                    _put(v,'xden',None,(v['xden'] + _get(v['q'], (v['i'],))) * v['ysq'])  # WRF:module_cam_error_function.F:299
                    return tuple(v[_n] for _n in ('_jump', 'i', 'xden', 'xnum'))
                _result=_loop(1,4,1,_body15,tuple(v[_n] for _n in ('_jump', 'i', 'xden', 'xnum')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'xden', 'xnum'),_result))
                _put(v,'result',None,v['ysq'] * (v['xnum'] + _get(v['p'], (5,))) / (v['xden'] + _get(v['q'], (5,))))  # WRF:module_cam_error_function.F:301
                _put(v,'result',None,(v['sqrpi'] - v['result']) / v['y'])  # WRF:module_cam_error_function.F:302
                _outer16=dict(v)
                def _yes16(_carry):
                    v=dict(_outer16);v.update(zip(('del', 'result', 'ysq'),_carry))
                    _put(v,'ysq',None,xp.trunc(v['y'] * v['sixten']) / v['sixten'])  # WRF:module_cam_error_function.F:304
                    _put(v,'del',None,(v['y'] - v['ysq']) * (v['y'] + v['ysq']))  # WRF:module_cam_error_function.F:305
                    _put(v,'result',None,xp.exp(_neg(v['ysq']) * v['ysq']) * xp.exp(_neg(v['del'])) * v['result'])  # WRF:module_cam_error_function.F:306
                    return tuple(v[_n] for _n in ('del', 'result', 'ysq'))
                def _no16(_carry):
                    v=dict(_outer16);v.update(zip(('del', 'result', 'ysq'),_carry))
                    return tuple(v[_n] for _n in ('del', 'result', 'ysq'))
                _result=_cond(v['jint'] != 2,_yes16,_no16,tuple(v[_n] for _n in ('del', 'result', 'ysq')),xp)  # WRF:module_cam_error_function.F:303
                v.update(zip(('del', 'result', 'ysq'),_result))
                return tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'))
            _result=_cond(v['y'] <= v['four'],_yes9,_no9,tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq')),xp)  # WRF:module_cam_error_function.F:266
            v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_result))
            return tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'))
        _result=_cond(v['y'] <= v['thresh'],_yes4,_no4,tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq')),xp)  # WRF:module_cam_error_function.F:250
        v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_result))
        return tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'))
    def _no3(_carry):
        v=dict(_outer3);v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_carry))
        return tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'))
    _result=_cond(v['_jump'] == 0,_yes3,_no3,tuple(v[_n] for _n in ('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq')),xp)  # WRF:module_cam_error_function.F:250
    v.update(zip(('_jump', 'del', 'i', 'result', 'xden', 'xnum', 'ysq'),_result))
    v['_jump']=xp.where(v['_jump']==30,_int(0),v['_jump'])
    _outer17=dict(v)
    def _yes17(_carry):
        v=dict(_outer17);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
        _outer18=dict(v)
        def _yes18(_carry):
            v=dict(_outer18);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
            _put(v,'result',None,v['half'] - v['result'] + v['half'])  # WRF:module_cam_error_function.F:314
            _outer19=dict(v)
            def _yes19(_carry):
                v=dict(_outer19);v.update(zip(('result',),_carry))
                _put(v,'result',None,_neg(v['result']))  # WRF:module_cam_error_function.F:315
                return tuple(v[_n] for _n in ('result',))
            def _no19(_carry):
                v=dict(_outer19);v.update(zip(('result',),_carry))
                return tuple(v[_n] for _n in ('result',))
            _result=_cond(v['x'] < v['zero'],_yes19,_no19,tuple(v[_n] for _n in ('result',)),xp)  # WRF:module_cam_error_function.F:315
            v.update(zip(('result',),_result))
            return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
        def _no18(_carry):
            v=dict(_outer18);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
            _outer20=dict(v)
            def _yes20(_carry):
                v=dict(_outer20);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
                _outer21=dict(v)
                def _yes21(_carry):
                    v=dict(_outer21);v.update(zip(('result',),_carry))
                    _put(v,'result',None,v['two'] - v['result'])  # WRF:module_cam_error_function.F:317
                    return tuple(v[_n] for _n in ('result',))
                def _no21(_carry):
                    v=dict(_outer21);v.update(zip(('result',),_carry))
                    return tuple(v[_n] for _n in ('result',))
                _result=_cond(v['x'] < v['zero'],_yes21,_no21,tuple(v[_n] for _n in ('result',)),xp)  # WRF:module_cam_error_function.F:317
                v.update(zip(('result',),_result))
                return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
            def _no20(_carry):
                v=dict(_outer20);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
                _outer22=dict(v)
                def _yes22(_carry):
                    v=dict(_outer22);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
                    _outer23=dict(v)
                    def _yes23(_carry):
                        v=dict(_outer23);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
                        _put(v,'result',None,v['xinf'])  # WRF:module_cam_error_function.F:321
                        return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
                    def _no23(_carry):
                        v=dict(_outer23);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
                        _put(v,'ysq',None,xp.trunc(v['x'] * v['sixten']) / v['sixten'])  # WRF:module_cam_error_function.F:323
                        _put(v,'del',None,(v['x'] - v['ysq']) * (v['x'] + v['ysq']))  # WRF:module_cam_error_function.F:324
                        _put(v,'y',None,xp.exp(v['ysq'] * v['ysq']) * xp.exp(v['del']))  # WRF:module_cam_error_function.F:325
                        _put(v,'result',None,v['y'] + v['y'] - v['result'])  # WRF:module_cam_error_function.F:326
                        return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
                    _result=_cond(v['x'] < v['xneg'],_yes23,_no23,tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq')),xp)  # WRF:module_cam_error_function.F:320
                    v.update(zip(('del', 'result', 'y', 'ysq'),_result))
                    return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
                def _no22(_carry):
                    v=dict(_outer22);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
                    return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
                _result=_cond(v['x'] < v['zero'],_yes22,_no22,tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq')),xp)  # WRF:module_cam_error_function.F:319
                v.update(zip(('del', 'result', 'y', 'ysq'),_result))
                return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
            _result=_cond(v['jint'] == 1,_yes20,_no20,tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq')),xp)  # WRF:module_cam_error_function.F:316
            v.update(zip(('del', 'result', 'y', 'ysq'),_result))
            return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
        _result=_cond(v['jint'] == 0,_yes18,_no18,tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq')),xp)  # WRF:module_cam_error_function.F:313
        v.update(zip(('del', 'result', 'y', 'ysq'),_result))
        return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
    def _no17(_carry):
        v=dict(_outer17);v.update(zip(('del', 'result', 'y', 'ysq'),_carry))
        return tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq'))
    _result=_cond(v['_jump'] == 0,_yes17,_no17,tuple(v[_n] for _n in ('del', 'result', 'y', 'ysq')),xp)  # WRF:module_cam_error_function.F:313
    v.update(zip(('del', 'result', 'y', 'ysq'),_result))
    v['_jump']=xp.where(v['_jump']==80,_int(0),v['_jump'])
    return (v['result'],)

def derf(x,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('error_function',{}))
    v['x']=x
    v['_jump']=_int(0)
    v['derf']=_zeros((),'f64',xp)
    v['jint']=_cast(0,'i32',xp)
    _call=calerf_r8(v['x'],v['derf'],v['jint'],_globals=_globals,xp=xp)
    _put(v,'derf',None,_call[0])  # WRF:module_cam_error_function.F:515
    return v['derf']

def derfc(x,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('error_function',{}))
    v['x']=x
    v['_jump']=_int(0)
    v['derfc']=_zeros((),'f64',xp)
    v['jint']=_cast(1,'i32',xp)
    _call=calerf_r8(v['x'],v['derfc'],v['jint'],_globals=_globals,xp=xp)
    _put(v,'derfc',None,_call[0])  # WRF:module_cam_error_function.F:567
    return v['derfc']

ARGUMENTS={'calerf_r8': ['arg', 'result', 'jint'], 'derf': ['x'], 'derfc': ['x']}
OUTPUTS={'calerf_r8': ['result'], 'derf': [], 'derfc': []}
VARIABLES={'calerf_r8': {'rk': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': 'r8', 'parameter': True, 'allocatable': False}, 'arg': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'jint': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'result': {'kind': 'f64', 'dims': [], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'x': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'y': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ysq': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xnum': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xden': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'del': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zero': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.0e0_r8', 'parameter': True, 'allocatable': False}, 'four': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '4.0e0_r8', 'parameter': True, 'allocatable': False}, 'one': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '1.0e0_r8', 'parameter': True, 'allocatable': False}, 'half': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.5e0_r8', 'parameter': True, 'allocatable': False}, 'two': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '2.0e0_r8', 'parameter': True, 'allocatable': False}, 'sqrpi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '5.6418958354775628695e-1_r8', 'parameter': True, 'allocatable': False}, 'thresh': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.46875e0_r8', 'parameter': True, 'allocatable': False}, 'sixten': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '16.0e0_r8', 'parameter': True, 'allocatable': False}, 'xinf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '1.79e308_r8', 'parameter': True, 'allocatable': False}, 'xneg': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '-26.628e0_r8', 'parameter': True, 'allocatable': False}, 'xsmall': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '1.11e-16_r8', 'parameter': True, 'allocatable': False}, 'xbig': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '26.543e0_r8', 'parameter': True, 'allocatable': False}, 'xhuge': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '6.71e7_r8', 'parameter': True, 'allocatable': False}, 'xmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '2.53e307_r8', 'parameter': True, 'allocatable': False}, 'a': {'kind': 'f64', 'dims': ['5'], 'intent': '', 'initial': '(/ 3.16112374387056560e00_r8, 1.13864154151050156e02_r8, 3.77485237685302021e02_r8, 3.20937758913846947e03_r8, 1.85777706184603153e-1_r8 /)', 'parameter': True, 'allocatable': False}, 'b': {'kind': 'f64', 'dims': ['4'], 'intent': '', 'initial': '(/ 2.36012909523441209e01_r8, 2.44024637934444173e02_r8, 1.28261652607737228e03_r8, 2.84423683343917062e03_r8 /)', 'parameter': True, 'allocatable': False}, 'c': {'kind': 'f64', 'dims': ['9'], 'intent': '', 'initial': '(/ 5.64188496988670089e-1_r8, 8.88314979438837594e00_r8, 6.61191906371416295e01_r8, 2.98635138197400131e02_r8, 8.81952221241769090e02_r8, 1.71204761263407058e03_r8, 2.05107837782607147e03_r8, 1.23033935479799725e03_r8, 2.15311535474403846e-8_r8 /)', 'parameter': True, 'allocatable': False}, 'd': {'kind': 'f64', 'dims': ['8'], 'intent': '', 'initial': '(/ 1.57449261107098347e01_r8, 1.17693950891312499e02_r8, 5.37181101862009858e02_r8, 1.62138957456669019e03_r8, 3.29079923573345963e03_r8, 4.36261909014324716e03_r8, 3.43936767414372164e03_r8, 1.23033935480374942e03_r8 /)', 'parameter': True, 'allocatable': False}, 'p': {'kind': 'f64', 'dims': ['6'], 'intent': '', 'initial': '(/ 3.05326634961232344e-1_r8, 3.60344899949804439e-1_r8, 1.25781726111229246e-1_r8, 1.60837851487422766e-2_r8, 6.58749161529837803e-4_r8, 1.63153871373020978e-2_r8 /)', 'parameter': True, 'allocatable': False}, 'q': {'kind': 'f64', 'dims': ['5'], 'intent': '', 'initial': '(/ 2.56852019228982242e00_r8, 1.87295284992346047e00_r8, 5.27905102951428412e-1_r8, 6.05183413124413191e-2_r8, 2.33520497626869185e-3_r8 /)', 'parameter': True, 'allocatable': False}}, 'derf': {'rk': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': 'r8', 'parameter': True, 'allocatable': False}, 'x': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'derf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'jint': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': '0', 'parameter': False, 'allocatable': False}}, 'derfc': {'rk': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': 'r8', 'parameter': True, 'allocatable': False}, 'x': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'derfc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'jint': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': '1', 'parameter': False, 'allocatable': False}}}
erf=derf
erfc=derfc
