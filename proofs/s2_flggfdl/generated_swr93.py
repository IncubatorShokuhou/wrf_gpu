"""Generated GFDL SWR93 column reference; CPU/native qualification pending.
Regenerate with proofs/s2_flggfdl/transcribe_gfdl.py. No runtime source parsing.
Pristine source SHA256: b6b2e9e006eff0b61f637a30e31a70f21b3c2bd8275182acb3c49aaf3fac0b4a
"""
import jax.numpy as jnp
from jax import lax

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

def reference_factory(nlayers):
    _NLAYERS=int(nlayers)
    if _NLAYERS<3:raise ValueError('WRF radiation needs at least three layers')
    def _swr93(fswc, hswc, ufswc, dfswc, fswl, hswl, ufswl, dfswl, press, coszro, taudar, rh2o, rrco2, ssolar, qo3, nclds, ktopsw, kbtmsw, camt, crr, ctt, alvb, alnb, alvd, alnd, gdfvb, gdfnb, gdfvd, gdfnd, abcff, pwts, h35e1, h1224e3, one, zero, haf, h69766e5, hp219, hp816, rrayav, ginv, cfco2, cfo3, two, h235m3, hp26, h129m2, h75826m4, h1036e2, h1p082, hmp805, h1386e2, h658m2, h2118m2, h42m2, h323m4, hm1ez, diffctr, o3difctr, fifty, radcon, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte):
        _r = jnp.float32 if jnp.asarray(fswc).dtype == jnp.dtype("float32") else jnp.float64
        v = {'fswc': fswc, 'hswc': hswc, 'ufswc': ufswc, 'dfswc': dfswc, 'fswl': fswl, 'hswl': hswl, 'ufswl': ufswl, 'dfswl': dfswl, 'press': press, 'coszro': coszro, 'taudar': taudar, 'rh2o': rh2o, 'rrco2': rrco2, 'ssolar': ssolar, 'qo3': qo3, 'nclds': nclds, 'ktopsw': ktopsw, 'kbtmsw': kbtmsw, 'camt': camt, 'crr': crr, 'ctt': ctt, 'alvb': alvb, 'alnb': alnb, 'alvd': alvd, 'alnd': alnd, 'gdfvb': gdfvb, 'gdfnb': gdfnb, 'gdfvd': gdfvd, 'gdfnd': gdfnd, 'abcff': abcff, 'pwts': pwts, 'h35e1': h35e1, 'h1224e3': h1224e3, 'one': one, 'zero': zero, 'haf': haf, 'h69766e5': h69766e5, 'hp219': hp219, 'hp816': hp816, 'rrayav': rrayav, 'ginv': ginv, 'cfco2': cfco2, 'cfo3': cfo3, 'two': two, 'h235m3': h235m3, 'hp26': hp26, 'h129m2': h129m2, 'h75826m4': h75826m4, 'h1036e2': h1036e2, 'h1p082': h1p082, 'hmp805': hmp805, 'h1386e2': h1386e2, 'h658m2': h658m2, 'h2118m2': h2118m2, 'h42m2': h42m2, 'h323m4': h323m4, 'hm1ez': hm1ez, 'diffctr': diffctr, 'o3difctr': o3difctr, 'fifty': fifty, 'radcon': radcon, 'ids': ids, 'ide': ide, 'jds': jds, 'jde': jde, 'kds': kds, 'kde': kde, 'ims': ims, 'ime': ime, 'jms': jms, 'jme': jme, 'kms': kms, 'kme': kme, 'its': its, 'ite': ite, 'jts': jts, 'jte': jte, 'kts': kts, 'kte': kte}
        v['_go'] = jnp.int32(0)
        v['alfa'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['alfau'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ccmax'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['cr'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ct'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['denom'] = _r(0)
        v['dfn'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dfnclu'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dfntop'] = jnp.zeros((((1))-((1))+1,(12),), dtype=_r)
        v['dfntrn'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['dpcld'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['du'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['duco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['duo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ff'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ffco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ffo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['htemp'] = _r(0)
        v['i'] = jnp.int32(0)
        v['ip'] = jnp.int32(0)
        v['j1'] = jnp.int32(0)
        v['j2'] = jnp.int32(0)
        v['j3'] = jnp.int32(0)
        v['jtop'] = jnp.int32(0)
        v['k'] = jnp.int32(0)
        v['kclds'] = jnp.int32(0)
        v['kk'] = jnp.int32(0)
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
        v['myis1'] = jnp.int32(0)
        v['n'] = jnp.int32(0)
        v['nnclds'] = jnp.int32(0)
        v['pp'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['pptop'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['pr2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['refl'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['refl2'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['rray'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['secz'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['tcld'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tclu'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+2)-((1))+1,), dtype=_r)
        v['tdcl1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tdcl2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tdco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tdo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tempf'] = _r(0)
        v['tempg'] = _r(0)
        v['tmp1'] = jnp.zeros((((1))-((1))+1,), dtype=_r)
        v['to3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+2)-((1))+1,), dtype=_r)
        v['ttd'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ttdb1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ttu'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ttub1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tucl1'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tuco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['tuo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['uco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+2)-((1))+1,), dtype=_r)
        v['ud'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['udco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['udo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ufn'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ufnclu'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['ufntrn'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['uo3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)*2+2)-((1))+1,), dtype=_r)
        v['ur'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['urco2'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['uro3'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        v['xamt'] = jnp.zeros((((1))-((1))+1,((_NLAYERS)+1)-((1))+1,), dtype=_r)
        # WRF:4803 l=kte
        v['l'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['kte']), v['l'])
        # WRF:4804 lp1=l+1
        v['lp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(1))), v['lp1'])
        # WRF:4804 lp2=l+2
        v['lp2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(2))), v['lp2'])
        # WRF:4804 lp3=l+3
        v['lp3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] + jnp.int32(3))), v['lp3'])
        # WRF:4804 llp1 = 2*l + 1
        v['llp1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(((jnp.int32(2) * v['l']) + jnp.int32(1))), v['llp1'])
        # WRF:4805 lm1=l-1
        v['lm1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(1))), v['lm1'])
        # WRF:4805 lm2=l-2
        v['lm2'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(2))), v['lm2'])
        # WRF:4805 lm3=l-3
        v['lm3'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['l'] - jnp.int32(3))), v['lm3'])
        # WRF:4805 ll = 2*l
        v['ll'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((jnp.int32(2) * v['l'])), v['ll'])
        # WRF:4806 myis=its
        v['myis'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['its']), v['myis'])
        # WRF:4806 myie=ite
        v['myie'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['ite']), v['myie'])
        # WRF:4807 myis1=myis+1
        v['myis1'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32((v['myis'] + jnp.int32(1))), v['myis1'])
        # WRF:4809 do i=myis,myie
        _p1 = ((jnp.bool_(True)) & (v['_go']==0))
        _p2 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p3(_p4,v):
            v = dict(v)
            v['i'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((v['myis'])+_p4*(jnp.int32(1))), v['i'])
            # WRF:4810 secz(i) = h35e1/sqrt(h1224e3*coszro(i)*coszro(i)+one)
            v['secz'] = v['secz'].at[(v['i'])-1].set(jnp.where((((_p1 & (v['_go']==0))) & (v['_go']==0)), _r((v['h35e1'] / jnp.sqrt((((v['h1224e3'] * v['coszro'][(v['i'])-1]) * v['coszro'][(v['i'])-1]) + v['one'])))), v['secz'][(v['i'])-1]))
            # WRF:4811 pp(i,1)   = zero
            v['pp'] = v['pp'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p1 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['pp'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4812 pp(i,lp1) = press(i,lp1)
            v['pp'] = v['pp'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p1 & (v['_go']==0))) & (v['_go']==0)), _r(v['press'][(v['i'])-1, (v['lp1'])-1]), v['pp'][(v['i'])-1, (v['lp1'])-1]))
            # WRF:4813 tmp1(i)  = one/press(i,lp1)
            v['tmp1'] = v['tmp1'].at[(v['i'])-1].set(jnp.where((((_p1 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['press'][(v['i'])-1, (v['lp1'])-1])), v['tmp1'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==100, 0, v['_go'])
            # WRF:4814 continue
            return v
        v = lax.fori_loop(0,_p2,_p3,v)
        v['i'] = jnp.where((_p1 & (v['_go']==0)), jnp.int32((v['myis'])+_p2*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4815 do k=1,lm1
        _p5 = ((jnp.bool_(True)) & (v['_go']==0))
        _p6 = jnp.maximum(jnp.int32(0), ((v['lm1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p7(_p8,v):
            v = dict(v)
            v['k'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p8*(jnp.int32(1))), v['k'])
            # WRF:4816 do i=myis,myie
            _p9 = (((_p5 & (v['_go']==0))) & (v['_go']==0))
            _p10 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p11(_p12,v):
                v = dict(v)
                v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p12*(jnp.int32(1))), v['i'])
                # WRF:4817 pp(i,k+1) = haf*(press(i,k+1)+press(i,k))
                v['pp'] = v['pp'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p9 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['press'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + v['press'][(v['i'])-1, (v['k'])-1]))), v['pp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==110, 0, v['_go'])
                # WRF:4818 continue
                return v
            v = lax.fori_loop(0,_p10,_p11,v)
            v['i'] = jnp.where((_p9 & (v['_go']==0)), jnp.int32((v['myis'])+_p10*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p6,_p7,v)
        v['k'] = jnp.where((_p5 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p6*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4819 do k=1,l
        _p13 = ((jnp.bool_(True)) & (v['_go']==0))
        _p14 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p15(_p16,v):
            v = dict(v)
            v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p16*(jnp.int32(1))), v['k'])
            # WRF:4820 do i=myis,myie
            _p17 = (((_p13 & (v['_go']==0))) & (v['_go']==0))
            _p18 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p19(_p20,v):
                v = dict(v)
                v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p20*(jnp.int32(1))), v['i'])
                # WRF:4821 dp (i,k) = pp(i,k+1)-pp(i,k)
                v['dp'] = v['dp'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['pp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['pp'][(v['i'])-1, (v['k'])-1])), v['dp'][(v['i'])-1, (v['k'])-1]))
                # WRF:4822 pr2(i,k) = haf*(pp(i,k)+pp(i,k+1))
                v['pr2'] = v['pr2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p17 & (v['_go']==0))) & (v['_go']==0)), _r((v['haf'] * (v['pp'][(v['i'])-1, (v['k'])-1] + v['pp'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))), v['pr2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==120, 0, v['_go'])
                # WRF:4823 continue
                return v
            v = lax.fori_loop(0,_p18,_p19,v)
            v['i'] = jnp.where((_p17 & (v['_go']==0)), jnp.int32((v['myis'])+_p18*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p14,_p15,v)
        v['k'] = jnp.where((_p13 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p14*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4824 do k=1,l
        _p21 = ((jnp.bool_(True)) & (v['_go']==0))
        _p22 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p23(_p24,v):
            v = dict(v)
            v['k'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p24*(jnp.int32(1))), v['k'])
            # WRF:4825 do i=myis,myie
            _p25 = (((_p21 & (v['_go']==0))) & (v['_go']==0))
            _p26 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p27(_p28,v):
                v = dict(v)
                v['i'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((v['myis'])+_p28*(jnp.int32(1))), v['i'])
                # WRF:4826 pr2(i,k) = pr2(i,k)*tmp1(i)
                v['pr2'] = v['pr2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p25 & (v['_go']==0))) & (v['_go']==0)), _r((v['pr2'][(v['i'])-1, (v['k'])-1] * v['tmp1'][(v['i'])-1])), v['pr2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==130, 0, v['_go'])
                # WRF:4827 continue
                return v
            v = lax.fori_loop(0,_p26,_p27,v)
            v['i'] = jnp.where((_p25 & (v['_go']==0)), jnp.int32((v['myis'])+_p26*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p22,_p23,v)
        v['k'] = jnp.where((_p21 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p22*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4829 do n=1,nb
        _p29 = ((jnp.bool_(True)) & (v['_go']==0))
        _p30 = jnp.maximum(jnp.int32(0), ((jnp.int32(jnp.int32(12)))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p31(_p32,v):
            v = dict(v)
            v['n'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p32*(jnp.int32(1))), v['n'])
            # WRF:4830 do ip=myis,myie
            _p33 = (((_p29 & (v['_go']==0))) & (v['_go']==0))
            _p34 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p35(_p36,v):
                v = dict(v)
                v['ip'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p36*(jnp.int32(1))), v['ip'])
                # WRF:4831 dfntop(ip,n) = ssolar*h69766e5*coszro(ip)*taudar(ip)*pwts(n)
                v['dfntop'] = v['dfntop'].at[(v['ip'])-1, (v['n'])-1].set(jnp.where((((_p33 & (v['_go']==0))) & (v['_go']==0)), _r(((((v['ssolar'] * v['h69766e5']) * v['coszro'][(v['ip'])-1]) * v['taudar'][(v['ip'])-1]) * v['pwts'][(v['n'])-1])), v['dfntop'][(v['ip'])-1, (v['n'])-1]))
                v['_go'] = jnp.where(v['_go']==140, 0, v['_go'])
                # WRF:4832 continue
                return v
            v = lax.fori_loop(0,_p34,_p35,v)
            v['ip'] = jnp.where((_p33 & (v['_go']==0)), jnp.int32((v['myis'])+_p34*(jnp.int32(1))), v['ip'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p30,_p31,v)
        v['n'] = jnp.where((_p29 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p30*(jnp.int32(1))), v['n'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4835 do i=myis,myie
        _p37 = ((jnp.bool_(True)) & (v['_go']==0))
        _p38 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p39(_p40,v):
            v = dict(v)
            v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p40*(jnp.int32(1))), v['i'])
            # WRF:4836 rray(i) = hp219/(one+hp816*coszro(i))
            v['rray'] = v['rray'].at[(v['i'])-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r((v['hp219'] / (v['one'] + (v['hp816'] * v['coszro'][(v['i'])-1])))), v['rray'][(v['i'])-1]))
            # WRF:4837 refl(i) = rray(i) + (one-rray(i))*(one-rrayav)*alvb(i)/ (one-alvd(i)*rrayav)
            v['refl'] = v['refl'].at[(v['i'])-1].set(jnp.where((((_p37 & (v['_go']==0))) & (v['_go']==0)), _r((v['rray'][(v['i'])-1] + ((((v['one'] - v['rray'][(v['i'])-1]) * (v['one'] - v['rrayav'])) * v['alvb'][(v['i'])-1]) / (v['one'] - (v['alvd'][(v['i'])-1] * v['rrayav']))))), v['refl'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==150, 0, v['_go'])
            # WRF:4839 continue
            return v
        v = lax.fori_loop(0,_p38,_p39,v)
        v['i'] = jnp.where((_p37 & (v['_go']==0)), jnp.int32((v['myis'])+_p38*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4840 do i=myis,myie
        _p41 = ((jnp.bool_(True)) & (v['_go']==0))
        _p42 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p43(_p44,v):
            v = dict(v)
            v['i'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((v['myis'])+_p44*(jnp.int32(1))), v['i'])
            # WRF:4841 rray(i) = 0.104/(one+4.8*coszro(i))
            v['rray'] = v['rray'].at[(v['i'])-1].set(jnp.where((((_p41 & (v['_go']==0))) & (v['_go']==0)), _r((_r(0.104) / (v['one'] + (_r(4.8) * v['coszro'][(v['i'])-1])))), v['rray'][(v['i'])-1]))
            # WRF:4842 refl2(i)= rray(i) + (one-rray(i))*(one-0.093)*alvb(i)/ (one-alvd(i)*0.093)
            v['refl2'] = v['refl2'].at[(v['i'])-1].set(jnp.where((((_p41 & (v['_go']==0))) & (v['_go']==0)), _r((v['rray'][(v['i'])-1] + ((((v['one'] - v['rray'][(v['i'])-1]) * (v['one'] - _r(0.093))) * v['alvb'][(v['i'])-1]) / (v['one'] - (v['alvd'][(v['i'])-1] * _r(0.093)))))), v['refl2'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==155, 0, v['_go'])
            # WRF:4844 continue
            return v
        v = lax.fori_loop(0,_p42,_p43,v)
        v['i'] = jnp.where((_p41 & (v['_go']==0)), jnp.int32((v['myis'])+_p42*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4848 do k=1,l
        _p45 = ((jnp.bool_(True)) & (v['_go']==0))
        _p46 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p47(_p48,v):
            v = dict(v)
            v['k'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p48*(jnp.int32(1))), v['k'])
            # WRF:4849 do i=myis,myie
            _p49 = (((_p45 & (v['_go']==0))) & (v['_go']==0))
            _p50 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p51(_p52,v):
                v = dict(v)
                v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p52*(jnp.int32(1))), v['i'])
                # WRF:4850 du   (i,k) = ginv*rh2o(i,k)*dp(i,k)*pr2(i,k)
                v['du'] = v['du'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((((v['ginv'] * v['rh2o'][(v['i'])-1, (v['k'])-1]) * v['dp'][(v['i'])-1, (v['k'])-1]) * v['pr2'][(v['i'])-1, (v['k'])-1])), v['du'][(v['i'])-1, (v['k'])-1]))
                # WRF:4851 duco2(i,k) = (rrco2*ginv*cfco2)*dp(i,k)*pr2(i,k)
                v['duco2'] = v['duco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r(((((v['rrco2'] * v['ginv']) * v['cfco2']) * v['dp'][(v['i'])-1, (v['k'])-1]) * v['pr2'][(v['i'])-1, (v['k'])-1])), v['duco2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4852 duo3 (i,k) = (ginv*cfo3)*qo3(i,k)*dp(i,k)
                v['duo3'] = v['duo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p49 & (v['_go']==0))) & (v['_go']==0)), _r((((v['ginv'] * v['cfo3']) * v['qo3'][(v['i'])-1, (v['k'])-1]) * v['dp'][(v['i'])-1, (v['k'])-1])), v['duo3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==160, 0, v['_go'])
                # WRF:4853 continue
                return v
            v = lax.fori_loop(0,_p50,_p51,v)
            v['i'] = jnp.where((_p49 & (v['_go']==0)), jnp.int32((v['myis'])+_p50*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p46,_p47,v)
        v['k'] = jnp.where((_p45 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p46*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4861 do ip=myis,myie
        _p53 = ((jnp.bool_(True)) & (v['_go']==0))
        _p54 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p55(_p56,v):
            v = dict(v)
            v['ip'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((v['myis'])+_p56*(jnp.int32(1))), v['ip'])
            # WRF:4862 ud   (ip,1) = zero
            v['ud'] = v['ud'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p53 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['ud'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:4863 udco2(ip,1) = zero
            v['udco2'] = v['udco2'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p53 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['udco2'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:4864 udo3 (ip,1) = zero
            v['udo3'] = v['udo3'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p53 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['udo3'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:4866 uo3  (ip,1) = udo3 (ip,1)
            v['uo3'] = v['uo3'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p53 & (v['_go']==0))) & (v['_go']==0)), _r(v['udo3'][(v['ip'])-1, (jnp.int32(1))-1]), v['uo3'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:4867 uco2 (ip,1) = udco2(ip,1)
            v['uco2'] = v['uco2'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p53 & (v['_go']==0))) & (v['_go']==0)), _r(v['udco2'][(v['ip'])-1, (jnp.int32(1))-1]), v['uco2'][(v['ip'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==200, 0, v['_go'])
            # WRF:4869 continue
            return v
        v = lax.fori_loop(0,_p54,_p55,v)
        v['ip'] = jnp.where((_p53 & (v['_go']==0)), jnp.int32((v['myis'])+_p54*(jnp.int32(1))), v['ip'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4870 do k=2,lp1
        _p57 = ((jnp.bool_(True)) & (v['_go']==0))
        _p58 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p59(_p60,v):
            v = dict(v)
            v['k'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p60*(jnp.int32(1))), v['k'])
            # WRF:4871 do i=myis,myie
            _p61 = (((_p57 & (v['_go']==0))) & (v['_go']==0))
            _p62 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p63(_p64,v):
                v = dict(v)
                v['i'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((v['myis'])+_p64*(jnp.int32(1))), v['i'])
                # WRF:4872 ud   (i,k) = ud   (i,k-1)+du   (i,k-1)*secz(i)
                v['ud'] = v['ud'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r((v['ud'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + (v['du'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] * v['secz'][(v['i'])-1]))), v['ud'][(v['i'])-1, (v['k'])-1]))
                # WRF:4873 udco2(i,k) = udco2(i,k-1)+duco2(i,k-1)*secz(i)
                v['udco2'] = v['udco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r((v['udco2'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + (v['duco2'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] * v['secz'][(v['i'])-1]))), v['udco2'][(v['i'])-1, (v['k'])-1]))
                # WRF:4874 udo3 (i,k) = udo3 (i,k-1)+duo3 (i,k-1)*secz(i)
                v['udo3'] = v['udo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r((v['udo3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + (v['duo3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] * v['secz'][(v['i'])-1]))), v['udo3'][(v['i'])-1, (v['k'])-1]))
                # WRF:4876 uo3  (i,k) = udo3 (i,k)
                v['uo3'] = v['uo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r(v['udo3'][(v['i'])-1, (v['k'])-1]), v['uo3'][(v['i'])-1, (v['k'])-1]))
                # WRF:4877 uco2 (i,k) = udco2(i,k)
                v['uco2'] = v['uco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p61 & (v['_go']==0))) & (v['_go']==0)), _r(v['udco2'][(v['i'])-1, (v['k'])-1]), v['uco2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==210, 0, v['_go'])
                # WRF:4879 continue
                return v
            v = lax.fori_loop(0,_p62,_p63,v)
            v['i'] = jnp.where((_p61 & (v['_go']==0)), jnp.int32((v['myis'])+_p62*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p58,_p59,v)
        v['k'] = jnp.where((_p57 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p58*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4880 do ip=myis,myie
        _p65 = ((jnp.bool_(True)) & (v['_go']==0))
        _p66 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p67(_p68,v):
            v = dict(v)
            v['ip'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((v['myis'])+_p68*(jnp.int32(1))), v['ip'])
            # WRF:4881 ur   (ip,lp1) = ud   (ip,lp1)
            v['ur'] = v['ur'].at[(v['ip'])-1, (v['lp1'])-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r(v['ud'][(v['ip'])-1, (v['lp1'])-1]), v['ur'][(v['ip'])-1, (v['lp1'])-1]))
            # WRF:4882 urco2(ip,lp1) = udco2(ip,lp1)
            v['urco2'] = v['urco2'].at[(v['ip'])-1, (v['lp1'])-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r(v['udco2'][(v['ip'])-1, (v['lp1'])-1]), v['urco2'][(v['ip'])-1, (v['lp1'])-1]))
            # WRF:4883 uro3 (ip,lp1) = udo3 (ip,lp1)
            v['uro3'] = v['uro3'].at[(v['ip'])-1, (v['lp1'])-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r(v['udo3'][(v['ip'])-1, (v['lp1'])-1]), v['uro3'][(v['ip'])-1, (v['lp1'])-1]))
            # WRF:4885 uo3  (ip,lp1+lp1) = uro3 (ip,lp1)
            v['uo3'] = v['uo3'].at[(v['ip'])-1, ((v['lp1'] + v['lp1']))-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r(v['uro3'][(v['ip'])-1, (v['lp1'])-1]), v['uo3'][(v['ip'])-1, ((v['lp1'] + v['lp1']))-1]))
            # WRF:4886 uco2 (ip,lp1+lp1) = urco2(ip,lp1)
            v['uco2'] = v['uco2'].at[(v['ip'])-1, ((v['lp1'] + v['lp1']))-1].set(jnp.where((((_p65 & (v['_go']==0))) & (v['_go']==0)), _r(v['urco2'][(v['ip'])-1, (v['lp1'])-1]), v['uco2'][(v['ip'])-1, ((v['lp1'] + v['lp1']))-1]))
            v['_go'] = jnp.where(v['_go']==220, 0, v['_go'])
            # WRF:4888 continue
            return v
        v = lax.fori_loop(0,_p66,_p67,v)
        v['ip'] = jnp.where((_p65 & (v['_go']==0)), jnp.int32((v['myis'])+_p66*(jnp.int32(1))), v['ip'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4889 do k=l,1,-1
        _p69 = ((jnp.bool_(True)) & (v['_go']==0))
        _p70 = jnp.maximum(jnp.int32(0), ((jnp.int32(1))-(v['l']))//(jnp.int32(-1))+jnp.int32(1))
        def _p71(_p72,v):
            v = dict(v)
            v['k'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['l'])+_p72*(jnp.int32(-1))), v['k'])
            # WRF:4890 do ip=myis,myie
            _p73 = (((_p69 & (v['_go']==0))) & (v['_go']==0))
            _p74 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p75(_p76,v):
                v = dict(v)
                v['ip'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((v['myis'])+_p76*(jnp.int32(1))), v['ip'])
                # WRF:4891 ur   (ip,k) = ur   (ip,k+1)+du   (ip,k)*diffctr
                v['ur'] = v['ur'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r((v['ur'][(v['ip'])-1, ((v['k'] + jnp.int32(1)))-1] + (v['du'][(v['ip'])-1, (v['k'])-1] * v['diffctr']))), v['ur'][(v['ip'])-1, (v['k'])-1]))
                # WRF:4892 urco2(ip,k) = urco2(ip,k+1)+duco2(ip,k)*diffctr
                v['urco2'] = v['urco2'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r((v['urco2'][(v['ip'])-1, ((v['k'] + jnp.int32(1)))-1] + (v['duco2'][(v['ip'])-1, (v['k'])-1] * v['diffctr']))), v['urco2'][(v['ip'])-1, (v['k'])-1]))
                # WRF:4893 uro3 (ip,k) = uro3 (ip,k+1)+duo3 (ip,k)*o3difctr
                v['uro3'] = v['uro3'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r((v['uro3'][(v['ip'])-1, ((v['k'] + jnp.int32(1)))-1] + (v['duo3'][(v['ip'])-1, (v['k'])-1] * v['o3difctr']))), v['uro3'][(v['ip'])-1, (v['k'])-1]))
                # WRF:4895 uo3 (ip,lp1+k) = uro3 (ip,k)
                v['uo3'] = v['uo3'].at[(v['ip'])-1, ((v['lp1'] + v['k']))-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r(v['uro3'][(v['ip'])-1, (v['k'])-1]), v['uo3'][(v['ip'])-1, ((v['lp1'] + v['k']))-1]))
                # WRF:4896 uco2(ip,lp1+k) = urco2(ip,k)
                v['uco2'] = v['uco2'].at[(v['ip'])-1, ((v['lp1'] + v['k']))-1].set(jnp.where((((_p73 & (v['_go']==0))) & (v['_go']==0)), _r(v['urco2'][(v['ip'])-1, (v['k'])-1]), v['uco2'][(v['ip'])-1, ((v['lp1'] + v['k']))-1]))
                v['_go'] = jnp.where(v['_go']==230, 0, v['_go'])
                # WRF:4898 continue
                return v
            v = lax.fori_loop(0,_p74,_p75,v)
            v['ip'] = jnp.where((_p73 & (v['_go']==0)), jnp.int32((v['myis'])+_p74*(jnp.int32(1))), v['ip'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p70,_p71,v)
        v['k'] = jnp.where((_p69 & (v['_go']==0)), jnp.int32((v['l'])+_p70*(jnp.int32(-1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4906 do k=1,ll
        _p77 = ((jnp.bool_(True)) & (v['_go']==0))
        _p78 = jnp.maximum(jnp.int32(0), ((v['ll'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p79(_p80,v):
            v = dict(v)
            v['k'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p80*(jnp.int32(1))), v['k'])
            # WRF:4907 do i=myis,myie
            _p81 = (((_p77 & (v['_go']==0))) & (v['_go']==0))
            _p82 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p83(_p84,v):
                v = dict(v)
                v['i'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((v['myis'])+_p84*(jnp.int32(1))), v['i'])
                # WRF:4908 tco2(i,k+1)=one-two*(h235m3*exp(hp26*log(uco2(i,k+1)+h129m2)) -h75826m4)
                v['tco2'] = v['tco2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p81 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - (v['two'] * ((v['h235m3'] * jnp.exp((v['hp26'] * jnp.log((v['uco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + v['h129m2']))))) - v['h75826m4'])))), v['tco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==240, 0, v['_go'])
                # WRF:4910 continue
                return v
            v = lax.fori_loop(0,_p82,_p83,v)
            v['i'] = jnp.where((_p81 & (v['_go']==0)), jnp.int32((v['myis'])+_p82*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p78,_p79,v)
        v['k'] = jnp.where((_p77 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p78*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4913 do k=1,l
        _p85 = ((jnp.bool_(True)) & (v['_go']==0))
        _p86 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p87(_p88,v):
            v = dict(v)
            v['k'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p88*(jnp.int32(1))), v['k'])
            # WRF:4914 do i=myis,myie
            _p89 = (((_p85 & (v['_go']==0))) & (v['_go']==0))
            _p90 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p91(_p92,v):
                v = dict(v)
                v['i'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((v['myis'])+_p92*(jnp.int32(1))), v['i'])
                # WRF:4915 tdco2(i,k+1)=tco2(i,k+1)
                v['tdco2'] = v['tdco2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p89 & (v['_go']==0))) & (v['_go']==0)), _r(v['tco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]), v['tdco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==241, 0, v['_go'])
                # WRF:4916 continue
                return v
            v = lax.fori_loop(0,_p90,_p91,v)
            v['i'] = jnp.where((_p89 & (v['_go']==0)), jnp.int32((v['myis'])+_p90*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p86,_p87,v)
        v['k'] = jnp.where((_p85 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p86*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4917 do k=1,l
        _p93 = ((jnp.bool_(True)) & (v['_go']==0))
        _p94 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p95(_p96,v):
            v = dict(v)
            v['k'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p96*(jnp.int32(1))), v['k'])
            # WRF:4918 do i=myis,myie
            _p97 = (((_p93 & (v['_go']==0))) & (v['_go']==0))
            _p98 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p99(_p100,v):
                v = dict(v)
                v['i'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((v['myis'])+_p100*(jnp.int32(1))), v['i'])
                # WRF:4919 tuco2(i,k)=tco2(i,lp1+k)
                v['tuco2'] = v['tuco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p97 & (v['_go']==0))) & (v['_go']==0)), _r(v['tco2'][(v['i'])-1, ((v['lp1'] + v['k']))-1]), v['tuco2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==242, 0, v['_go'])
                # WRF:4920 continue
                return v
            v = lax.fori_loop(0,_p98,_p99,v)
            v['i'] = jnp.where((_p97 & (v['_go']==0)), jnp.int32((v['myis'])+_p98*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p94,_p95,v)
        v['k'] = jnp.where((_p93 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p94*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4926 htemp = h1036e2*h1036e2*h1036e2
        v['htemp'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), _r(((v['h1036e2'] * v['h1036e2']) * v['h1036e2'])), v['htemp'])
        # WRF:4927 do k=1,ll
        _p101 = ((jnp.bool_(True)) & (v['_go']==0))
        _p102 = jnp.maximum(jnp.int32(0), ((v['ll'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p103(_p104,v):
            v = dict(v)
            v['k'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p104*(jnp.int32(1))), v['k'])
            # WRF:4928 do i=myis,myie
            _p105 = (((_p101 & (v['_go']==0))) & (v['_go']==0))
            _p106 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p107(_p108,v):
                v = dict(v)
                v['i'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((v['myis'])+_p108*(jnp.int32(1))), v['i'])
                # WRF:4929 to3(i,k+1)=one-two*uo3(i,k+1)* (h1p082*exp(hmp805*log(one+h1386e2*uo3(i,k+1)))+ h658m2/(one+htemp*uo3(i,k+1)*uo3(i,k+1)*uo3(i,k+1))+ h2118m2/(one+uo3(i,k+1)*(h42m2+h323m4*uo3(i,k+1))))
                v['to3'] = v['to3'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p105 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - ((v['two'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * (((v['h1p082'] * jnp.exp((v['hmp805'] * jnp.log((v['one'] + (v['h1386e2'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])))))) + (v['h658m2'] / (v['one'] + (((v['htemp'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])))) + (v['h2118m2'] / (v['one'] + (v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['h42m2'] + (v['h323m4'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))))))))), v['to3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==250, 0, v['_go'])
                # WRF:4933 continue
                return v
            v = lax.fori_loop(0,_p106,_p107,v)
            v['i'] = jnp.where((_p105 & (v['_go']==0)), jnp.int32((v['myis'])+_p106*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p102,_p103,v)
        v['k'] = jnp.where((_p101 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p102*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4936 do k=1,l
        _p109 = ((jnp.bool_(True)) & (v['_go']==0))
        _p110 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p111(_p112,v):
            v = dict(v)
            v['k'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p112*(jnp.int32(1))), v['k'])
            # WRF:4937 do i=myis,myie
            _p113 = (((_p109 & (v['_go']==0))) & (v['_go']==0))
            _p114 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p115(_p116,v):
                v = dict(v)
                v['i'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((v['myis'])+_p116*(jnp.int32(1))), v['i'])
                # WRF:4938 tdo3(i,k+1)=to3(i,k+1)
                v['tdo3'] = v['tdo3'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p113 & (v['_go']==0))) & (v['_go']==0)), _r(v['to3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]), v['tdo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==251, 0, v['_go'])
                # WRF:4939 continue
                return v
            v = lax.fori_loop(0,_p114,_p115,v)
            v['i'] = jnp.where((_p113 & (v['_go']==0)), jnp.int32((v['myis'])+_p114*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p110,_p111,v)
        v['k'] = jnp.where((_p109 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p110*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4940 do k=1,l
        _p117 = ((jnp.bool_(True)) & (v['_go']==0))
        _p118 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p119(_p120,v):
            v = dict(v)
            v['k'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p120*(jnp.int32(1))), v['k'])
            # WRF:4941 do i=myis,myie
            _p121 = (((_p117 & (v['_go']==0))) & (v['_go']==0))
            _p122 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p123(_p124,v):
                v = dict(v)
                v['i'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((v['myis'])+_p124*(jnp.int32(1))), v['i'])
                # WRF:4942 tuo3(i,k)=to3(i,lp1+k)
                v['tuo3'] = v['tuo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p121 & (v['_go']==0))) & (v['_go']==0)), _r(v['to3'][(v['i'])-1, ((v['lp1'] + v['k']))-1]), v['tuo3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==252, 0, v['_go'])
                # WRF:4943 continue
                return v
            v = lax.fori_loop(0,_p122,_p123,v)
            v['i'] = jnp.where((_p121 & (v['_go']==0)), jnp.int32((v['myis'])+_p122*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p118,_p119,v)
        v['k'] = jnp.where((_p117 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p118*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4949 do k=1,l
        _p125 = ((jnp.bool_(True)) & (v['_go']==0))
        _p126 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p127(_p128,v):
            v = dict(v)
            v['k'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p128*(jnp.int32(1))), v['k'])
            # WRF:4950 do i=myis,myie
            _p129 = (((_p125 & (v['_go']==0))) & (v['_go']==0))
            _p130 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p131(_p132,v):
                v = dict(v)
                v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p132*(jnp.int32(1))), v['i'])
                # WRF:4951 ttd(i,k+1) = exp(hm1ez*min(fifty,abcff(1)*ud(i,k+1)))
                v['ttd'] = v['ttd'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(jnp.int32(1))-1] * v['ud'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))))), v['ttd'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:4952 ttu(i,k) = exp(hm1ez*min(fifty,abcff(1)*ur(i,k)))
                v['ttu'] = v['ttu'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(jnp.int32(1))-1] * v['ur'][(v['i'])-1, (v['k'])-1]))))), v['ttu'][(v['i'])-1, (v['k'])-1]))
                # WRF:4953 dfn(i,k+1) = ttd(i,k+1)*tdo3(i,k+1)
                v['dfn'] = v['dfn'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttd'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['tdo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['dfn'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:4954 ufn(i,k) = ttu(i,k)*tuo3(i,k)
                v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p129 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttu'][(v['i'])-1, (v['k'])-1] * v['tuo3'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==260, 0, v['_go'])
                # WRF:4955 continue
                return v
            v = lax.fori_loop(0,_p130,_p131,v)
            v['i'] = jnp.where((_p129 & (v['_go']==0)), jnp.int32((v['myis'])+_p130*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p126,_p127,v)
        v['k'] = jnp.where((_p125 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p126*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4956 do i=myis,myie
        _p133 = ((jnp.bool_(True)) & (v['_go']==0))
        _p134 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p135(_p136,v):
            v = dict(v)
            v['i'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((v['myis'])+_p136*(jnp.int32(1))), v['i'])
            # WRF:4957 dfn(i,1)   = one
            v['dfn'] = v['dfn'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p133 & (v['_go']==0))) & (v['_go']==0)), _r(v['one']), v['dfn'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:4958 ufn(i,lp1) = dfn(i,lp1)
            v['ufn'] = v['ufn'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p133 & (v['_go']==0))) & (v['_go']==0)), _r(v['dfn'][(v['i'])-1, (v['lp1'])-1]), v['ufn'][(v['i'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==270, 0, v['_go'])
            # WRF:4959 continue
            return v
        v = lax.fori_loop(0,_p134,_p135,v)
        v['i'] = jnp.where((_p133 & (v['_go']==0)), jnp.int32((v['myis'])+_p134*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4963 do k=1,lp1
        _p137 = ((jnp.bool_(True)) & (v['_go']==0))
        _p138 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p139(_p140,v):
            v = dict(v)
            v['k'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p140*(jnp.int32(1))), v['k'])
            # WRF:4964 do i=myis,myie
            _p141 = (((_p137 & (v['_go']==0))) & (v['_go']==0))
            _p142 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p143(_p144,v):
                v = dict(v)
                v['i'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((v['myis'])+_p144*(jnp.int32(1))), v['i'])
                # WRF:4965 dfswl(i,k) =         dfn(i,k)*dfntop(i,1)
                v['dfswl'] = v['dfswl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p141 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfn'][(v['i'])-1, (v['k'])-1] * v['dfntop'][(v['i'])-1, (jnp.int32(1))-1])), v['dfswl'][(v['i'])-1, (v['k'])-1]))
                # WRF:4966 ufswl(i,k) = refl(i)*ufn(i,k)*dfntop(i,1)
                v['ufswl'] = v['ufswl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p141 & (v['_go']==0))) & (v['_go']==0)), _r(((v['refl'][(v['i'])-1] * v['ufn'][(v['i'])-1, (v['k'])-1]) * v['dfntop'][(v['i'])-1, (jnp.int32(1))-1])), v['ufswl'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==280, 0, v['_go'])
                # WRF:4967 continue
                return v
            v = lax.fori_loop(0,_p142,_p143,v)
            v['i'] = jnp.where((_p141 & (v['_go']==0)), jnp.int32((v['myis'])+_p142*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p138,_p139,v)
        v['k'] = jnp.where((_p137 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p138*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4968 do i=myis,myie
        _p145 = ((jnp.bool_(True)) & (v['_go']==0))
        _p146 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p147(_p148,v):
            v = dict(v)
            v['i'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32((v['myis'])+_p148*(jnp.int32(1))), v['i'])
            # WRF:4969 gdfvb(i) = dfswl(i,lp1)*exp(-0.15746*secz(i))
            v['gdfvb'] = v['gdfvb'].at[(v['i'])-1].set(jnp.where((((_p145 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfswl'][(v['i'])-1, (v['lp1'])-1] * jnp.exp(((-_r(0.15746)) * v['secz'][(v['i'])-1])))), v['gdfvb'][(v['i'])-1]))
            # WRF:4970 gdfvd(i) = ((one-refl2(i))*dfswl(i,lp1) - (one-alvb(i)) *gdfvb(i)) / (one-alvd(i))
            v['gdfvd'] = v['gdfvd'].at[(v['i'])-1].set(jnp.where((((_p145 & (v['_go']==0))) & (v['_go']==0)), _r(((((v['one'] - v['refl2'][(v['i'])-1]) * v['dfswl'][(v['i'])-1, (v['lp1'])-1]) - ((v['one'] - v['alvb'][(v['i'])-1]) * v['gdfvb'][(v['i'])-1])) / (v['one'] - v['alvd'][(v['i'])-1]))), v['gdfvd'][(v['i'])-1]))
            # WRF:4972 gdfnb(i) = zero
            v['gdfnb'] = v['gdfnb'].at[(v['i'])-1].set(jnp.where((((_p145 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['gdfnb'][(v['i'])-1]))
            # WRF:4973 gdfnd(i) = zero
            v['gdfnd'] = v['gdfnd'].at[(v['i'])-1].set(jnp.where((((_p145 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['gdfnd'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==285, 0, v['_go'])
            # WRF:4974 continue
            return v
        v = lax.fori_loop(0,_p146,_p147,v)
        v['i'] = jnp.where((_p145 & (v['_go']==0)), jnp.int32((v['myis'])+_p146*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:4979 do n=2,nb
        _p149 = ((jnp.bool_(True)) & (v['_go']==0))
        _p150 = jnp.maximum(jnp.int32(0), ((jnp.int32(jnp.int32(12)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p151(_p152,v):
            v = dict(v)
            v['n'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p152*(jnp.int32(1))), v['n'])
            # WRF:4980 __if__ n.eq.2
            _p153 = (((_p149 & (v['_go']==0))) & (v['_go']==0)) & ((v['n'] == jnp.int32(2)))
            _p154 = (((_p149 & (v['_go']==0))) & (v['_go']==0)) & ~((v['n'] == jnp.int32(2)))
            # WRF:4984 do k=1,l
            _p155 = ((_p153) & (v['_go']==0))
            _p156 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p157(_p158,v):
                v = dict(v)
                v['k'] = jnp.where((_p155 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p158*(jnp.int32(1))), v['k'])
                # WRF:4985 do i=myis,myie
                _p159 = (((_p155 & (v['_go']==0))) & (v['_go']==0))
                _p160 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p161(_p162,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p159 & (v['_go']==0)), jnp.int32((v['myis'])+_p162*(jnp.int32(1))), v['i'])
                    # WRF:4986 dfn(i,k+1) = ttd(i,k+1)*tdco2(i,k+1)
                    v['dfn'] = v['dfn'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p159 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttd'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['tdco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['dfn'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                    # WRF:4987 ufn(i,k) = ttu(i,k)*tuco2(i,k)
                    v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p159 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttu'][(v['i'])-1, (v['k'])-1] * v['tuco2'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==290, 0, v['_go'])
                    # WRF:4988 continue
                    return v
                v = lax.fori_loop(0,_p160,_p161,v)
                v['i'] = jnp.where((_p159 & (v['_go']==0)), jnp.int32((v['myis'])+_p160*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p156,_p157,v)
            v['k'] = jnp.where((_p155 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p156*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:4993 do k=1,l
            _p163 = ((_p154) & (v['_go']==0))
            _p164 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p165(_p166,v):
                v = dict(v)
                v['k'] = jnp.where((_p163 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p166*(jnp.int32(1))), v['k'])
                # WRF:4994 do i=myis,myie
                _p167 = (((_p163 & (v['_go']==0))) & (v['_go']==0))
                _p168 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p169(_p170,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p167 & (v['_go']==0)), jnp.int32((v['myis'])+_p170*(jnp.int32(1))), v['i'])
                    # WRF:4995 dfn(i,k+1)=exp(hm1ez*min(fifty,abcff(n)*ud(i,k+1))) *tdco2(i,k+1)
                    v['dfn'] = v['dfn'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p167 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(v['n'])-1] * v['ud'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])))) * v['tdco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['dfn'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                    # WRF:4997 ufn(i,k)=exp(hm1ez*min(fifty,abcff(n)*ur(i,k))) *tuco2(i,k)
                    v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p167 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(v['n'])-1] * v['ur'][(v['i'])-1, (v['k'])-1])))) * v['tuco2'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==300, 0, v['_go'])
                    # WRF:4999 continue
                    return v
                v = lax.fori_loop(0,_p168,_p169,v)
                v['i'] = jnp.where((_p167 & (v['_go']==0)), jnp.int32((v['myis'])+_p168*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p164,_p165,v)
            v['k'] = jnp.where((_p163 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p164*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5003 do i=myis,myie
            _p171 = (((_p149 & (v['_go']==0))) & (v['_go']==0))
            _p172 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p173(_p174,v):
                v = dict(v)
                v['i'] = jnp.where((_p171 & (v['_go']==0)), jnp.int32((v['myis'])+_p174*(jnp.int32(1))), v['i'])
                # WRF:5004 dfn(i,1)   = one
                v['dfn'] = v['dfn'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p171 & (v['_go']==0))) & (v['_go']==0)), _r(v['one']), v['dfn'][(v['i'])-1, (jnp.int32(1))-1]))
                # WRF:5005 ufn(i,lp1) = dfn(i,lp1)
                v['ufn'] = v['ufn'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p171 & (v['_go']==0))) & (v['_go']==0)), _r(v['dfn'][(v['i'])-1, (v['lp1'])-1]), v['ufn'][(v['i'])-1, (v['lp1'])-1]))
                v['_go'] = jnp.where(v['_go']==310, 0, v['_go'])
                # WRF:5006 continue
                return v
            v = lax.fori_loop(0,_p172,_p173,v)
            v['i'] = jnp.where((_p171 & (v['_go']==0)), jnp.int32((v['myis'])+_p172*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5009 do k=1,lp1
            _p175 = (((_p149 & (v['_go']==0))) & (v['_go']==0))
            _p176 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p177(_p178,v):
                v = dict(v)
                v['k'] = jnp.where((_p175 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p178*(jnp.int32(1))), v['k'])
                # WRF:5010 do i=myis,myie
                _p179 = (((_p175 & (v['_go']==0))) & (v['_go']==0))
                _p180 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
                def _p181(_p182,v):
                    v = dict(v)
                    v['i'] = jnp.where((_p179 & (v['_go']==0)), jnp.int32((v['myis'])+_p182*(jnp.int32(1))), v['i'])
                    # WRF:5011 dfswl(i,k) = dfswl(i,k) +         dfn(i,k)*dfntop(i,n)
                    v['dfswl'] = v['dfswl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p179 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfswl'][(v['i'])-1, (v['k'])-1] + (v['dfn'][(v['i'])-1, (v['k'])-1] * v['dfntop'][(v['i'])-1, (v['n'])-1]))), v['dfswl'][(v['i'])-1, (v['k'])-1]))
                    # WRF:5012 ufswl(i,k) = ufswl(i,k) + alnb(i)*ufn(i,k)*dfntop(i,n)
                    v['ufswl'] = v['ufswl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p179 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufswl'][(v['i'])-1, (v['k'])-1] + ((v['alnb'][(v['i'])-1] * v['ufn'][(v['i'])-1, (v['k'])-1]) * v['dfntop'][(v['i'])-1, (v['n'])-1]))), v['ufswl'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==320, 0, v['_go'])
                    # WRF:5013 continue
                    return v
                v = lax.fori_loop(0,_p180,_p181,v)
                v['i'] = jnp.where((_p179 & (v['_go']==0)), jnp.int32((v['myis'])+_p180*(jnp.int32(1))), v['i'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                return v
            v = lax.fori_loop(0,_p176,_p177,v)
            v['k'] = jnp.where((_p175 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p176*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5014 do i=myis,myie
            _p183 = (((_p149 & (v['_go']==0))) & (v['_go']==0))
            _p184 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p185(_p186,v):
                v = dict(v)
                v['i'] = jnp.where((_p183 & (v['_go']==0)), jnp.int32((v['myis'])+_p186*(jnp.int32(1))), v['i'])
                # WRF:5015 gdfnb(i) = gdfnb(i) + dfn(i,lp1)*dfntop(i,n)
                v['gdfnb'] = v['gdfnb'].at[(v['i'])-1].set(jnp.where((((_p183 & (v['_go']==0))) & (v['_go']==0)), _r((v['gdfnb'][(v['i'])-1] + (v['dfn'][(v['i'])-1, (v['lp1'])-1] * v['dfntop'][(v['i'])-1, (v['n'])-1]))), v['gdfnb'][(v['i'])-1]))
                v['_go'] = jnp.where(v['_go']==330, 0, v['_go'])
                # WRF:5016 continue
                return v
            v = lax.fori_loop(0,_p184,_p185,v)
            v['i'] = jnp.where((_p183 & (v['_go']==0)), jnp.int32((v['myis'])+_p184*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==350, 0, v['_go'])
            # WRF:5017 continue
            return v
        v = lax.fori_loop(0,_p150,_p151,v)
        v['n'] = jnp.where((_p149 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p150*(jnp.int32(1))), v['n'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5018 do k=1,lp1
        _p187 = ((jnp.bool_(True)) & (v['_go']==0))
        _p188 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p189(_p190,v):
            v = dict(v)
            v['k'] = jnp.where((_p187 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p190*(jnp.int32(1))), v['k'])
            # WRF:5019 do i=myis,myie
            _p191 = (((_p187 & (v['_go']==0))) & (v['_go']==0))
            _p192 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p193(_p194,v):
                v = dict(v)
                v['i'] = jnp.where((_p191 & (v['_go']==0)), jnp.int32((v['myis'])+_p194*(jnp.int32(1))), v['i'])
                # WRF:5020 fswl(i,k) = ufswl(i,k)-dfswl(i,k)
                v['fswl'] = v['fswl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p191 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufswl'][(v['i'])-1, (v['k'])-1] - v['dfswl'][(v['i'])-1, (v['k'])-1])), v['fswl'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==360, 0, v['_go'])
                # WRF:5021 continue
                return v
            v = lax.fori_loop(0,_p192,_p193,v)
            v['i'] = jnp.where((_p191 & (v['_go']==0)), jnp.int32((v['myis'])+_p192*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p188,_p189,v)
        v['k'] = jnp.where((_p187 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p188*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5022 do k=1,l
        _p195 = ((jnp.bool_(True)) & (v['_go']==0))
        _p196 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p197(_p198,v):
            v = dict(v)
            v['k'] = jnp.where((_p195 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p198*(jnp.int32(1))), v['k'])
            # WRF:5023 do i=myis,myie
            _p199 = (((_p195 & (v['_go']==0))) & (v['_go']==0))
            _p200 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p201(_p202,v):
                v = dict(v)
                v['i'] = jnp.where((_p199 & (v['_go']==0)), jnp.int32((v['myis'])+_p202*(jnp.int32(1))), v['i'])
                # WRF:5024 hswl(i,k)=radcon*(fswl(i,k+1)-fswl(i,k))/dp(i,k)
                v['hswl'] = v['hswl'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p199 & (v['_go']==0))) & (v['_go']==0)), _r(((v['radcon'] * (v['fswl'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] - v['fswl'][(v['i'])-1, (v['k'])-1])) / v['dp'][(v['i'])-1, (v['k'])-1])), v['hswl'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==370, 0, v['_go'])
                # WRF:5025 continue
                return v
            v = lax.fori_loop(0,_p200,_p201,v)
            v['i'] = jnp.where((_p199 & (v['_go']==0)), jnp.int32((v['myis'])+_p200*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p196,_p197,v)
        v['k'] = jnp.where((_p195 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p196*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5031 kclds=nclds(myis)
        v['kclds'] = jnp.where(((jnp.bool_(True)) & (v['_go']==0)), jnp.int32(v['nclds'][(v['myis'])-1]), v['kclds'])
        # WRF:5032 do i=myis1,myie
        _p203 = ((jnp.bool_(True)) & (v['_go']==0))
        _p204 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis1']))//(jnp.int32(1))+jnp.int32(1))
        def _p205(_p206,v):
            v = dict(v)
            v['i'] = jnp.where((_p203 & (v['_go']==0)), jnp.int32((v['myis1'])+_p206*(jnp.int32(1))), v['i'])
            # WRF:5033 kclds=max(nclds(i),kclds)
            v['kclds'] = jnp.where((((_p203 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(_max(v['nclds'][(v['i'])-1], v['kclds'])), v['kclds'])
            v['_go'] = jnp.where(v['_go']==400, 0, v['_go'])
            # WRF:5034 continue
            return v
        v = lax.fori_loop(0,_p204,_p205,v)
        v['i'] = jnp.where((_p203 & (v['_go']==0)), jnp.int32((v['myis1'])+_p204*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5035 do k=1,lp1
        _p207 = ((jnp.bool_(True)) & (v['_go']==0))
        _p208 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p209(_p210,v):
            v = dict(v)
            v['k'] = jnp.where((_p207 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p210*(jnp.int32(1))), v['k'])
            # WRF:5036 do i=myis,myie
            _p211 = (((_p207 & (v['_go']==0))) & (v['_go']==0))
            _p212 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p213(_p214,v):
                v = dict(v)
                v['i'] = jnp.where((_p211 & (v['_go']==0)), jnp.int32((v['myis'])+_p214*(jnp.int32(1))), v['i'])
                # WRF:5037 dfswc(i,k) = dfswl(i,k)
                v['dfswc'] = v['dfswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p211 & (v['_go']==0))) & (v['_go']==0)), _r(v['dfswl'][(v['i'])-1, (v['k'])-1]), v['dfswc'][(v['i'])-1, (v['k'])-1]))
                # WRF:5038 ufswc(i,k) = ufswl(i,k)
                v['ufswc'] = v['ufswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p211 & (v['_go']==0))) & (v['_go']==0)), _r(v['ufswl'][(v['i'])-1, (v['k'])-1]), v['ufswc'][(v['i'])-1, (v['k'])-1]))
                # WRF:5039 fswc (i,k) = fswl (i,k)
                v['fswc'] = v['fswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p211 & (v['_go']==0))) & (v['_go']==0)), _r(v['fswl'][(v['i'])-1, (v['k'])-1]), v['fswc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==410, 0, v['_go'])
                # WRF:5040 continue
                return v
            v = lax.fori_loop(0,_p212,_p213,v)
            v['i'] = jnp.where((_p211 & (v['_go']==0)), jnp.int32((v['myis'])+_p212*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p208,_p209,v)
        v['k'] = jnp.where((_p207 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p208*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5041 do k=1,l
        _p215 = ((jnp.bool_(True)) & (v['_go']==0))
        _p216 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p217(_p218,v):
            v = dict(v)
            v['k'] = jnp.where((_p215 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p218*(jnp.int32(1))), v['k'])
            # WRF:5042 do i=myis,myie
            _p219 = (((_p215 & (v['_go']==0))) & (v['_go']==0))
            _p220 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p221(_p222,v):
                v = dict(v)
                v['i'] = jnp.where((_p219 & (v['_go']==0)), jnp.int32((v['myis'])+_p222*(jnp.int32(1))), v['i'])
                # WRF:5043 hswc(i,k) = hswl(i,k)
                v['hswc'] = v['hswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p219 & (v['_go']==0))) & (v['_go']==0)), _r(v['hswl'][(v['i'])-1, (v['k'])-1]), v['hswc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==420, 0, v['_go'])
                # WRF:5044 continue
                return v
            v = lax.fori_loop(0,_p220,_p221,v)
            v['i'] = jnp.where((_p219 & (v['_go']==0)), jnp.int32((v['myis'])+_p220*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p216,_p217,v)
        v['k'] = jnp.where((_p215 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p216*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5046 if (kclds .eq. 0)  return
        _p223 = ((jnp.bool_(True)) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
        # WRF:5046 return
        v['_go'] = jnp.where(((_p223) & (v['_go']==0)), -1, v['_go'])
        # WRF:5048 do k=1,lp1
        _p224 = ((jnp.bool_(True)) & (v['_go']==0))
        _p225 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p226(_p227,v):
            v = dict(v)
            v['k'] = jnp.where((_p224 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p227*(jnp.int32(1))), v['k'])
            # WRF:5049 do i=myis,myie
            _p228 = (((_p224 & (v['_go']==0))) & (v['_go']==0))
            _p229 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p230(_p231,v):
                v = dict(v)
                v['i'] = jnp.where((_p228 & (v['_go']==0)), jnp.int32((v['myis'])+_p231*(jnp.int32(1))), v['i'])
                # WRF:5050 xamt(i,k) = camt(i,k)
                v['xamt'] = v['xamt'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p228 & (v['_go']==0))) & (v['_go']==0)), _r(v['camt'][(v['i'])-1, (v['k'])-1]), v['xamt'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==430, 0, v['_go'])
                # WRF:5051 continue
                return v
            v = lax.fori_loop(0,_p229,_p230,v)
            v['i'] = jnp.where((_p228 & (v['_go']==0)), jnp.int32((v['myis'])+_p229*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p225,_p226,v)
        v['k'] = jnp.where((_p224 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p225*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5052 do i=myis,myie
        _p232 = ((jnp.bool_(True)) & (v['_go']==0))
        _p233 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p234(_p235,v):
            v = dict(v)
            v['i'] = jnp.where((_p232 & (v['_go']==0)), jnp.int32((v['myis'])+_p235*(jnp.int32(1))), v['i'])
            # WRF:5053 nnclds   = nclds(i)
            v['nnclds'] = jnp.where((((_p232 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['nnclds'])
            # WRF:5054 ccmax(i) = zero
            v['ccmax'] = v['ccmax'].at[(v['i'])-1].set(jnp.where((((_p232 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['ccmax'][(v['i'])-1]))
            # WRF:5055 if (nnclds .le. 0) go to 470
            _p236 = (((_p232 & (v['_go']==0))) & (v['_go']==0)) & ((v['nnclds'] <= jnp.int32(0)))
            # WRF:5055 go to 470
            v['_go'] = jnp.where(((_p236) & (v['_go']==0)), 470, v['_go'])
            # WRF:5056 ccmax(i) = one
            v['ccmax'] = v['ccmax'].at[(v['i'])-1].set(jnp.where((((_p232 & (v['_go']==0))) & (v['_go']==0)), _r(v['one']), v['ccmax'][(v['i'])-1]))
            # WRF:5057 do k=1,nnclds
            _p237 = (((_p232 & (v['_go']==0))) & (v['_go']==0))
            _p238 = jnp.maximum(jnp.int32(0), ((v['nnclds'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p239(_p240,v):
                v = dict(v)
                v['k'] = jnp.where((_p237 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p240*(jnp.int32(1))), v['k'])
                # WRF:5058 ccmax(i) = ccmax(i) * (one - camt(i,k+1))
                v['ccmax'] = v['ccmax'].at[(v['i'])-1].set(jnp.where((((_p237 & (v['_go']==0))) & (v['_go']==0)), _r((v['ccmax'][(v['i'])-1] * (v['one'] - v['camt'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))), v['ccmax'][(v['i'])-1]))
                v['_go'] = jnp.where(v['_go']==450, 0, v['_go'])
                # WRF:5059 continue
                return v
            v = lax.fori_loop(0,_p238,_p239,v)
            v['k'] = jnp.where((_p237 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p238*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5060 ccmax(i) = one - ccmax(i)
            v['ccmax'] = v['ccmax'].at[(v['i'])-1].set(jnp.where((((_p232 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - v['ccmax'][(v['i'])-1])), v['ccmax'][(v['i'])-1]))
            # WRF:5061 __if__ ccmax(i) .gt. zero
            _p241 = (((_p232 & (v['_go']==0))) & (v['_go']==0)) & ((v['ccmax'][(v['i'])-1] > v['zero']))
            _p242 = (((_p232 & (v['_go']==0))) & (v['_go']==0)) & ~((v['ccmax'][(v['i'])-1] > v['zero']))
            # WRF:5062 do k=1,nnclds
            _p243 = ((_p241) & (v['_go']==0))
            _p244 = jnp.maximum(jnp.int32(0), ((v['nnclds'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p245(_p246,v):
                v = dict(v)
                v['k'] = jnp.where((_p243 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p246*(jnp.int32(1))), v['k'])
                # WRF:5063 xamt(i,k+1) = camt(i,k+1)/ccmax(i)
                v['xamt'] = v['xamt'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p243 & (v['_go']==0))) & (v['_go']==0)), _r((v['camt'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] / v['ccmax'][(v['i'])-1])), v['xamt'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==460, 0, v['_go'])
                # WRF:5064 continue
                return v
            v = lax.fori_loop(0,_p244,_p245,v)
            v['k'] = jnp.where((_p243 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p244*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==470, 0, v['_go'])
            # WRF:5066 continue
            return v
        v = lax.fori_loop(0,_p233,_p234,v)
        v['i'] = jnp.where((_p232 & (v['_go']==0)), jnp.int32((v['myis'])+_p233*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5067 do k=1,lp1
        _p247 = ((jnp.bool_(True)) & (v['_go']==0))
        _p248 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p249(_p250,v):
            v = dict(v)
            v['k'] = jnp.where((_p247 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p250*(jnp.int32(1))), v['k'])
            # WRF:5068 do i=myis,myie
            _p251 = (((_p247 & (v['_go']==0))) & (v['_go']==0))
            _p252 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p253(_p254,v):
                v = dict(v)
                v['i'] = jnp.where((_p251 & (v['_go']==0)), jnp.int32((v['myis'])+_p254*(jnp.int32(1))), v['i'])
                # WRF:5069 ff   (i,k) = diffctr
                v['ff'] = v['ff'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p251 & (v['_go']==0))) & (v['_go']==0)), _r(v['diffctr']), v['ff'][(v['i'])-1, (v['k'])-1]))
                # WRF:5070 ffco2(i,k) = diffctr
                v['ffco2'] = v['ffco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p251 & (v['_go']==0))) & (v['_go']==0)), _r(v['diffctr']), v['ffco2'][(v['i'])-1, (v['k'])-1]))
                # WRF:5071 ffo3 (i,k) = o3difctr
                v['ffo3'] = v['ffo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p251 & (v['_go']==0))) & (v['_go']==0)), _r(v['o3difctr']), v['ffo3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==480, 0, v['_go'])
                # WRF:5072 continue
                return v
            v = lax.fori_loop(0,_p252,_p253,v)
            v['i'] = jnp.where((_p251 & (v['_go']==0)), jnp.int32((v['myis'])+_p252*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p248,_p249,v)
        v['k'] = jnp.where((_p247 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p248*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5073 do ip=myis,myie
        _p255 = ((jnp.bool_(True)) & (v['_go']==0))
        _p256 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p257(_p258,v):
            v = dict(v)
            v['ip'] = jnp.where((_p255 & (v['_go']==0)), jnp.int32((v['myis'])+_p258*(jnp.int32(1))), v['ip'])
            # WRF:5074 jtop = ktopsw(ip,nclds(ip)+1)
            v['jtop'] = jnp.where((((_p255 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['ktopsw'][(v['ip'])-1, ((v['nclds'][(v['ip'])-1] + jnp.int32(1)))-1]), v['jtop'])
            # WRF:5075 do k=1,jtop
            _p259 = (((_p255 & (v['_go']==0))) & (v['_go']==0))
            _p260 = jnp.maximum(jnp.int32(0), ((v['jtop'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p261(_p262,v):
                v = dict(v)
                v['k'] = jnp.where((_p259 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p262*(jnp.int32(1))), v['k'])
                # WRF:5076 ff   (ip,k) = secz(ip)
                v['ff'] = v['ff'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p259 & (v['_go']==0))) & (v['_go']==0)), _r(v['secz'][(v['ip'])-1]), v['ff'][(v['ip'])-1, (v['k'])-1]))
                # WRF:5077 ffco2(ip,k) = secz(ip)
                v['ffco2'] = v['ffco2'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p259 & (v['_go']==0))) & (v['_go']==0)), _r(v['secz'][(v['ip'])-1]), v['ffco2'][(v['ip'])-1, (v['k'])-1]))
                # WRF:5078 ffo3 (ip,k) = secz(ip)
                v['ffo3'] = v['ffo3'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p259 & (v['_go']==0))) & (v['_go']==0)), _r(v['secz'][(v['ip'])-1]), v['ffo3'][(v['ip'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==490, 0, v['_go'])
                # WRF:5079 continue
                return v
            v = lax.fori_loop(0,_p260,_p261,v)
            v['k'] = jnp.where((_p259 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p260*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p256,_p257,v)
        v['ip'] = jnp.where((_p255 & (v['_go']==0)), jnp.int32((v['myis'])+_p256*(jnp.int32(1))), v['ip'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5080 do i=myis,myie
        _p263 = ((jnp.bool_(True)) & (v['_go']==0))
        _p264 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p265(_p266,v):
            v = dict(v)
            v['i'] = jnp.where((_p263 & (v['_go']==0)), jnp.int32((v['myis'])+_p266*(jnp.int32(1))), v['i'])
            # WRF:5081 rray(i) = hp219/(one+hp816*coszro(i))
            v['rray'] = v['rray'].at[(v['i'])-1].set(jnp.where((((_p263 & (v['_go']==0))) & (v['_go']==0)), _r((v['hp219'] / (v['one'] + (v['hp816'] * v['coszro'][(v['i'])-1])))), v['rray'][(v['i'])-1]))
            # WRF:5082 refl(i) = rray(i) + (one-rray(i))*(one-rrayav)*alvd(i)/ (one-alvd(i)*rrayav)
            v['refl'] = v['refl'].at[(v['i'])-1].set(jnp.where((((_p263 & (v['_go']==0))) & (v['_go']==0)), _r((v['rray'][(v['i'])-1] + ((((v['one'] - v['rray'][(v['i'])-1]) * (v['one'] - v['rrayav'])) * v['alvd'][(v['i'])-1]) / (v['one'] - (v['alvd'][(v['i'])-1] * v['rrayav']))))), v['refl'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==500, 0, v['_go'])
            # WRF:5084 continue
            return v
        v = lax.fori_loop(0,_p264,_p265,v)
        v['i'] = jnp.where((_p263 & (v['_go']==0)), jnp.int32((v['myis'])+_p264*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5085 do ip=myis,myie
        _p267 = ((jnp.bool_(True)) & (v['_go']==0))
        _p268 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p269(_p270,v):
            v = dict(v)
            v['ip'] = jnp.where((_p267 & (v['_go']==0)), jnp.int32((v['myis'])+_p270*(jnp.int32(1))), v['ip'])
            # WRF:5086 ud   (ip,1) = zero
            v['ud'] = v['ud'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p267 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['ud'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:5087 udco2(ip,1) = zero
            v['udco2'] = v['udco2'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p267 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['udco2'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:5088 udo3 (ip,1) = zero
            v['udo3'] = v['udo3'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p267 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['udo3'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:5090 uo3  (ip,1) = udo3 (ip,1)
            v['uo3'] = v['uo3'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p267 & (v['_go']==0))) & (v['_go']==0)), _r(v['udo3'][(v['ip'])-1, (jnp.int32(1))-1]), v['uo3'][(v['ip'])-1, (jnp.int32(1))-1]))
            # WRF:5091 uco2 (ip,1) = udco2(ip,1)
            v['uco2'] = v['uco2'].at[(v['ip'])-1, (jnp.int32(1))-1].set(jnp.where((((_p267 & (v['_go']==0))) & (v['_go']==0)), _r(v['udco2'][(v['ip'])-1, (jnp.int32(1))-1]), v['uco2'][(v['ip'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==510, 0, v['_go'])
            # WRF:5093 continue
            return v
        v = lax.fori_loop(0,_p268,_p269,v)
        v['ip'] = jnp.where((_p267 & (v['_go']==0)), jnp.int32((v['myis'])+_p268*(jnp.int32(1))), v['ip'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5094 do k=2,lp1
        _p271 = ((jnp.bool_(True)) & (v['_go']==0))
        _p272 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p273(_p274,v):
            v = dict(v)
            v['k'] = jnp.where((_p271 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p274*(jnp.int32(1))), v['k'])
            # WRF:5095 do i=myis,myie
            _p275 = (((_p271 & (v['_go']==0))) & (v['_go']==0))
            _p276 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p277(_p278,v):
                v = dict(v)
                v['i'] = jnp.where((_p275 & (v['_go']==0)), jnp.int32((v['myis'])+_p278*(jnp.int32(1))), v['i'])
                # WRF:5096 ud   (i,k) = ud   (i,k-1)+du   (i,k-1)*ff   (i,k)
                v['ud'] = v['ud'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p275 & (v['_go']==0))) & (v['_go']==0)), _r((v['ud'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + (v['du'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] * v['ff'][(v['i'])-1, (v['k'])-1]))), v['ud'][(v['i'])-1, (v['k'])-1]))
                # WRF:5097 udco2(i,k) = udco2(i,k-1)+duco2(i,k-1)*ffco2(i,k)
                v['udco2'] = v['udco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p275 & (v['_go']==0))) & (v['_go']==0)), _r((v['udco2'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + (v['duco2'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] * v['ffco2'][(v['i'])-1, (v['k'])-1]))), v['udco2'][(v['i'])-1, (v['k'])-1]))
                # WRF:5098 udo3 (i,k) = udo3 (i,k-1)+duo3 (i,k-1)*ffo3 (i,k)
                v['udo3'] = v['udo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p275 & (v['_go']==0))) & (v['_go']==0)), _r((v['udo3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] + (v['duo3'][(v['i'])-1, ((v['k'] - jnp.int32(1)))-1] * v['ffo3'][(v['i'])-1, (v['k'])-1]))), v['udo3'][(v['i'])-1, (v['k'])-1]))
                # WRF:5100 uo3 (i,k)  = udo3 (i,k)
                v['uo3'] = v['uo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p275 & (v['_go']==0))) & (v['_go']==0)), _r(v['udo3'][(v['i'])-1, (v['k'])-1]), v['uo3'][(v['i'])-1, (v['k'])-1]))
                # WRF:5101 uco2(i,k)  = udco2(i,k)
                v['uco2'] = v['uco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p275 & (v['_go']==0))) & (v['_go']==0)), _r(v['udco2'][(v['i'])-1, (v['k'])-1]), v['uco2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==520, 0, v['_go'])
                # WRF:5103 continue
                return v
            v = lax.fori_loop(0,_p276,_p277,v)
            v['i'] = jnp.where((_p275 & (v['_go']==0)), jnp.int32((v['myis'])+_p276*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p272,_p273,v)
        v['k'] = jnp.where((_p271 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p272*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5104 do ip=myis,myie
        _p279 = ((jnp.bool_(True)) & (v['_go']==0))
        _p280 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p281(_p282,v):
            v = dict(v)
            v['ip'] = jnp.where((_p279 & (v['_go']==0)), jnp.int32((v['myis'])+_p282*(jnp.int32(1))), v['ip'])
            # WRF:5105 ur   (ip,lp1) = ud   (ip,lp1)
            v['ur'] = v['ur'].at[(v['ip'])-1, (v['lp1'])-1].set(jnp.where((((_p279 & (v['_go']==0))) & (v['_go']==0)), _r(v['ud'][(v['ip'])-1, (v['lp1'])-1]), v['ur'][(v['ip'])-1, (v['lp1'])-1]))
            # WRF:5106 urco2(ip,lp1) = udco2(ip,lp1)
            v['urco2'] = v['urco2'].at[(v['ip'])-1, (v['lp1'])-1].set(jnp.where((((_p279 & (v['_go']==0))) & (v['_go']==0)), _r(v['udco2'][(v['ip'])-1, (v['lp1'])-1]), v['urco2'][(v['ip'])-1, (v['lp1'])-1]))
            # WRF:5107 uro3 (ip,lp1) = udo3 (ip,lp1)
            v['uro3'] = v['uro3'].at[(v['ip'])-1, (v['lp1'])-1].set(jnp.where((((_p279 & (v['_go']==0))) & (v['_go']==0)), _r(v['udo3'][(v['ip'])-1, (v['lp1'])-1]), v['uro3'][(v['ip'])-1, (v['lp1'])-1]))
            # WRF:5109 uo3  (ip,lp1+lp1) = uro3 (ip,lp1)
            v['uo3'] = v['uo3'].at[(v['ip'])-1, ((v['lp1'] + v['lp1']))-1].set(jnp.where((((_p279 & (v['_go']==0))) & (v['_go']==0)), _r(v['uro3'][(v['ip'])-1, (v['lp1'])-1]), v['uo3'][(v['ip'])-1, ((v['lp1'] + v['lp1']))-1]))
            # WRF:5110 uco2 (ip,lp1+lp1) = urco2(ip,lp1)
            v['uco2'] = v['uco2'].at[(v['ip'])-1, ((v['lp1'] + v['lp1']))-1].set(jnp.where((((_p279 & (v['_go']==0))) & (v['_go']==0)), _r(v['urco2'][(v['ip'])-1, (v['lp1'])-1]), v['uco2'][(v['ip'])-1, ((v['lp1'] + v['lp1']))-1]))
            v['_go'] = jnp.where(v['_go']==530, 0, v['_go'])
            # WRF:5112 continue
            return v
        v = lax.fori_loop(0,_p280,_p281,v)
        v['ip'] = jnp.where((_p279 & (v['_go']==0)), jnp.int32((v['myis'])+_p280*(jnp.int32(1))), v['ip'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5113 do k=l,1,-1
        _p283 = ((jnp.bool_(True)) & (v['_go']==0))
        _p284 = jnp.maximum(jnp.int32(0), ((jnp.int32(1))-(v['l']))//(jnp.int32(-1))+jnp.int32(1))
        def _p285(_p286,v):
            v = dict(v)
            v['k'] = jnp.where((_p283 & (v['_go']==0)), jnp.int32((v['l'])+_p286*(jnp.int32(-1))), v['k'])
            # WRF:5114 do ip=myis,myie
            _p287 = (((_p283 & (v['_go']==0))) & (v['_go']==0))
            _p288 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p289(_p290,v):
                v = dict(v)
                v['ip'] = jnp.where((_p287 & (v['_go']==0)), jnp.int32((v['myis'])+_p290*(jnp.int32(1))), v['ip'])
                # WRF:5115 ur   (ip,k) = ur   (ip,k+1)+du   (ip,k)*diffctr
                v['ur'] = v['ur'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p287 & (v['_go']==0))) & (v['_go']==0)), _r((v['ur'][(v['ip'])-1, ((v['k'] + jnp.int32(1)))-1] + (v['du'][(v['ip'])-1, (v['k'])-1] * v['diffctr']))), v['ur'][(v['ip'])-1, (v['k'])-1]))
                # WRF:5116 urco2(ip,k) = urco2(ip,k+1)+duco2(ip,k)*diffctr
                v['urco2'] = v['urco2'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p287 & (v['_go']==0))) & (v['_go']==0)), _r((v['urco2'][(v['ip'])-1, ((v['k'] + jnp.int32(1)))-1] + (v['duco2'][(v['ip'])-1, (v['k'])-1] * v['diffctr']))), v['urco2'][(v['ip'])-1, (v['k'])-1]))
                # WRF:5117 uro3 (ip,k) = uro3 (ip,k+1)+duo3 (ip,k)*o3difctr
                v['uro3'] = v['uro3'].at[(v['ip'])-1, (v['k'])-1].set(jnp.where((((_p287 & (v['_go']==0))) & (v['_go']==0)), _r((v['uro3'][(v['ip'])-1, ((v['k'] + jnp.int32(1)))-1] + (v['duo3'][(v['ip'])-1, (v['k'])-1] * v['o3difctr']))), v['uro3'][(v['ip'])-1, (v['k'])-1]))
                # WRF:5119 uo3 (ip,lp1+k) = uro3 (ip,k)
                v['uo3'] = v['uo3'].at[(v['ip'])-1, ((v['lp1'] + v['k']))-1].set(jnp.where((((_p287 & (v['_go']==0))) & (v['_go']==0)), _r(v['uro3'][(v['ip'])-1, (v['k'])-1]), v['uo3'][(v['ip'])-1, ((v['lp1'] + v['k']))-1]))
                # WRF:5120 uco2(ip,lp1+k) = urco2(ip,k)
                v['uco2'] = v['uco2'].at[(v['ip'])-1, ((v['lp1'] + v['k']))-1].set(jnp.where((((_p287 & (v['_go']==0))) & (v['_go']==0)), _r(v['urco2'][(v['ip'])-1, (v['k'])-1]), v['uco2'][(v['ip'])-1, ((v['lp1'] + v['k']))-1]))
                v['_go'] = jnp.where(v['_go']==540, 0, v['_go'])
                # WRF:5122 continue
                return v
            v = lax.fori_loop(0,_p288,_p289,v)
            v['ip'] = jnp.where((_p287 & (v['_go']==0)), jnp.int32((v['myis'])+_p288*(jnp.int32(1))), v['ip'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p284,_p285,v)
        v['k'] = jnp.where((_p283 & (v['_go']==0)), jnp.int32((v['l'])+_p284*(jnp.int32(-1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5123 do k=1,ll
        _p291 = ((jnp.bool_(True)) & (v['_go']==0))
        _p292 = jnp.maximum(jnp.int32(0), ((v['ll'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p293(_p294,v):
            v = dict(v)
            v['k'] = jnp.where((_p291 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p294*(jnp.int32(1))), v['k'])
            # WRF:5124 do i=myis,myie
            _p295 = (((_p291 & (v['_go']==0))) & (v['_go']==0))
            _p296 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p297(_p298,v):
                v = dict(v)
                v['i'] = jnp.where((_p295 & (v['_go']==0)), jnp.int32((v['myis'])+_p298*(jnp.int32(1))), v['i'])
                # WRF:5125 tco2(i,k+1)=one-two*(h235m3*exp(hp26*log(uco2(i,k+1)+h129m2)) -h75826m4)
                v['tco2'] = v['tco2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p295 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - (v['two'] * ((v['h235m3'] * jnp.exp((v['hp26'] * jnp.log((v['uco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] + v['h129m2']))))) - v['h75826m4'])))), v['tco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==550, 0, v['_go'])
                # WRF:5127 continue
                return v
            v = lax.fori_loop(0,_p296,_p297,v)
            v['i'] = jnp.where((_p295 & (v['_go']==0)), jnp.int32((v['myis'])+_p296*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p292,_p293,v)
        v['k'] = jnp.where((_p291 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p292*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5129 do k=1,l
        _p299 = ((jnp.bool_(True)) & (v['_go']==0))
        _p300 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p301(_p302,v):
            v = dict(v)
            v['k'] = jnp.where((_p299 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p302*(jnp.int32(1))), v['k'])
            # WRF:5130 do i=myis,myie
            _p303 = (((_p299 & (v['_go']==0))) & (v['_go']==0))
            _p304 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p305(_p306,v):
                v = dict(v)
                v['i'] = jnp.where((_p303 & (v['_go']==0)), jnp.int32((v['myis'])+_p306*(jnp.int32(1))), v['i'])
                # WRF:5131 tdco2(i,k+1)=tco2(i,k+1)
                v['tdco2'] = v['tdco2'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p303 & (v['_go']==0))) & (v['_go']==0)), _r(v['tco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]), v['tdco2'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==551, 0, v['_go'])
                # WRF:5132 continue
                return v
            v = lax.fori_loop(0,_p304,_p305,v)
            v['i'] = jnp.where((_p303 & (v['_go']==0)), jnp.int32((v['myis'])+_p304*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p300,_p301,v)
        v['k'] = jnp.where((_p299 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p300*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5133 do k=1,l
        _p307 = ((jnp.bool_(True)) & (v['_go']==0))
        _p308 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p309(_p310,v):
            v = dict(v)
            v['k'] = jnp.where((_p307 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p310*(jnp.int32(1))), v['k'])
            # WRF:5134 do i=myis,myie
            _p311 = (((_p307 & (v['_go']==0))) & (v['_go']==0))
            _p312 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p313(_p314,v):
                v = dict(v)
                v['i'] = jnp.where((_p311 & (v['_go']==0)), jnp.int32((v['myis'])+_p314*(jnp.int32(1))), v['i'])
                # WRF:5135 tuco2(i,k)=tco2(i,lp1+k)
                v['tuco2'] = v['tuco2'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p311 & (v['_go']==0))) & (v['_go']==0)), _r(v['tco2'][(v['i'])-1, ((v['lp1'] + v['k']))-1]), v['tuco2'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==552, 0, v['_go'])
                # WRF:5136 continue
                return v
            v = lax.fori_loop(0,_p312,_p313,v)
            v['i'] = jnp.where((_p311 & (v['_go']==0)), jnp.int32((v['myis'])+_p312*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p308,_p309,v)
        v['k'] = jnp.where((_p307 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p308*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5138 do k=1,ll
        _p315 = ((jnp.bool_(True)) & (v['_go']==0))
        _p316 = jnp.maximum(jnp.int32(0), ((v['ll'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p317(_p318,v):
            v = dict(v)
            v['k'] = jnp.where((_p315 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p318*(jnp.int32(1))), v['k'])
            # WRF:5139 do i=myis,myie
            _p319 = (((_p315 & (v['_go']==0))) & (v['_go']==0))
            _p320 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p321(_p322,v):
                v = dict(v)
                v['i'] = jnp.where((_p319 & (v['_go']==0)), jnp.int32((v['myis'])+_p322*(jnp.int32(1))), v['i'])
                # WRF:5140 to3(i,k+1)=one-two*uo3(i,k+1)* (h1p082*exp(hmp805*log(one+h1386e2*uo3(i,k+1)))+ h658m2/(one+htemp*uo3(i,k+1)*uo3(i,k+1)*uo3(i,k+1))+ h2118m2/(one+uo3(i,k+1)*(h42m2+h323m4*uo3(i,k+1))))
                v['to3'] = v['to3'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p319 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - ((v['two'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * (((v['h1p082'] * jnp.exp((v['hmp805'] * jnp.log((v['one'] + (v['h1386e2'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])))))) + (v['h658m2'] / (v['one'] + (((v['htemp'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]) * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])))) + (v['h2118m2'] / (v['one'] + (v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * (v['h42m2'] + (v['h323m4'] * v['uo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))))))))), v['to3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==560, 0, v['_go'])
                # WRF:5144 continue
                return v
            v = lax.fori_loop(0,_p320,_p321,v)
            v['i'] = jnp.where((_p319 & (v['_go']==0)), jnp.int32((v['myis'])+_p320*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p316,_p317,v)
        v['k'] = jnp.where((_p315 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p316*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5146 do k=1,l
        _p323 = ((jnp.bool_(True)) & (v['_go']==0))
        _p324 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p325(_p326,v):
            v = dict(v)
            v['k'] = jnp.where((_p323 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p326*(jnp.int32(1))), v['k'])
            # WRF:5147 do i=myis,myie
            _p327 = (((_p323 & (v['_go']==0))) & (v['_go']==0))
            _p328 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p329(_p330,v):
                v = dict(v)
                v['i'] = jnp.where((_p327 & (v['_go']==0)), jnp.int32((v['myis'])+_p330*(jnp.int32(1))), v['i'])
                # WRF:5148 tdo3(i,k+1)=to3(i,k+1)
                v['tdo3'] = v['tdo3'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p327 & (v['_go']==0))) & (v['_go']==0)), _r(v['to3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]), v['tdo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==561, 0, v['_go'])
                # WRF:5149 continue
                return v
            v = lax.fori_loop(0,_p328,_p329,v)
            v['i'] = jnp.where((_p327 & (v['_go']==0)), jnp.int32((v['myis'])+_p328*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p324,_p325,v)
        v['k'] = jnp.where((_p323 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p324*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5150 do k=1,l
        _p331 = ((jnp.bool_(True)) & (v['_go']==0))
        _p332 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p333(_p334,v):
            v = dict(v)
            v['k'] = jnp.where((_p331 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p334*(jnp.int32(1))), v['k'])
            # WRF:5151 do i=myis,myie
            _p335 = (((_p331 & (v['_go']==0))) & (v['_go']==0))
            _p336 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p337(_p338,v):
                v = dict(v)
                v['i'] = jnp.where((_p335 & (v['_go']==0)), jnp.int32((v['myis'])+_p338*(jnp.int32(1))), v['i'])
                # WRF:5152 tuo3(i,k)=to3(i,lp1+k)
                v['tuo3'] = v['tuo3'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p335 & (v['_go']==0))) & (v['_go']==0)), _r(v['to3'][(v['i'])-1, ((v['lp1'] + v['k']))-1]), v['tuo3'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==562, 0, v['_go'])
                # WRF:5153 continue
                return v
            v = lax.fori_loop(0,_p336,_p337,v)
            v['i'] = jnp.where((_p335 & (v['_go']==0)), jnp.int32((v['myis'])+_p336*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p332,_p333,v)
        v['k'] = jnp.where((_p331 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p332*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5159 do i=myis,myie
        _p339 = ((jnp.bool_(True)) & (v['_go']==0))
        _p340 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p341(_p342,v):
            v = dict(v)
            v['i'] = jnp.where((_p339 & (v['_go']==0)), jnp.int32((v['myis'])+_p342*(jnp.int32(1))), v['i'])
            # WRF:5160 cr(i,1) = refl(i)
            v['cr'] = v['cr'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p339 & (v['_go']==0))) & (v['_go']==0)), _r(v['refl'][(v['i'])-1]), v['cr'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==570, 0, v['_go'])
            # WRF:5161 continue
            return v
        v = lax.fori_loop(0,_p340,_p341,v)
        v['i'] = jnp.where((_p339 & (v['_go']==0)), jnp.int32((v['myis'])+_p340*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5166 do i=myis,myie
        _p343 = ((jnp.bool_(True)) & (v['_go']==0))
        _p344 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p345(_p346,v):
            v = dict(v)
            v['i'] = jnp.where((_p343 & (v['_go']==0)), jnp.int32((v['myis'])+_p346*(jnp.int32(1))), v['i'])
            # WRF:5167 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p343 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5168 if(kclds.eq.0) go to 581
            _p347 = (((_p343 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5168 go to 581
            v['_go'] = jnp.where(((_p347) & (v['_go']==0)), 581, v['_go'])
            # WRF:5169 do kk=2,kclds+1
            _p348 = (((_p343 & (v['_go']==0))) & (v['_go']==0))
            _p349 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
            def _p350(_p351,v):
                v = dict(v)
                v['kk'] = jnp.where((_p348 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p351*(jnp.int32(1))), v['kk'])
                # WRF:5170 cr(i,kk) = crr(i,1,kk)*xamt(i,kk)
                v['cr'] = v['cr'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p348 & (v['_go']==0))) & (v['_go']==0)), _r((v['crr'][(v['i'])-1, (jnp.int32(1))-1, (v['kk'])-1] * v['xamt'][(v['i'])-1, (v['kk'])-1])), v['cr'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5171 ct(i,kk) = one - (one-ctt(i,1,kk))*xamt(i,kk)
                v['ct'] = v['ct'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p348 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - ((v['one'] - v['ctt'][(v['i'])-1, (jnp.int32(1))-1, (v['kk'])-1]) * v['xamt'][(v['i'])-1, (v['kk'])-1]))), v['ct'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==580, 0, v['_go'])
                # WRF:5172 continue
                return v
            v = lax.fori_loop(0,_p349,_p350,v)
            v['kk'] = jnp.where((_p348 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p349*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==581, 0, v['_go'])
            # WRF:5173 continue
            return v
        v = lax.fori_loop(0,_p344,_p345,v)
        v['i'] = jnp.where((_p343 & (v['_go']==0)), jnp.int32((v['myis'])+_p344*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5178 do i=myis,myie
        _p352 = ((jnp.bool_(True)) & (v['_go']==0))
        _p353 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p354(_p355,v):
            v = dict(v)
            v['i'] = jnp.where((_p352 & (v['_go']==0)), jnp.int32((v['myis'])+_p355*(jnp.int32(1))), v['i'])
            # WRF:5179 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p352 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5180 if(kclds.eq.0) go to 591
            _p356 = (((_p352 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5180 go to 591
            v['_go'] = jnp.where(((_p356) & (v['_go']==0)), 591, v['_go'])
            # WRF:5181 do kk=1,kclds
            _p357 = (((_p352 & (v['_go']==0))) & (v['_go']==0))
            _p358 = jnp.maximum(jnp.int32(0), ((v['kclds'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p359(_p360,v):
                v = dict(v)
                v['kk'] = jnp.where((_p357 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p360*(jnp.int32(1))), v['kk'])
                # WRF:5182 __if__ (kbtmsw(i,kk+1)-1).gt.ktopsw(i,kk+1)
                _p361 = (((_p357 & (v['_go']==0))) & (v['_go']==0)) & (((v['kbtmsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] - jnp.int32(1)) > v['ktopsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]))
                _p362 = (((_p357 & (v['_go']==0))) & (v['_go']==0)) & ~(((v['kbtmsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] - jnp.int32(1)) > v['ktopsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]))
                # WRF:5183 pptop(i,kk)=pp(i,ktopsw(i,kk+1))
                v['pptop'] = v['pptop'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where(((_p361) & (v['_go']==0)), _r(v['pp'][(v['i'])-1, (v['ktopsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])-1]), v['pptop'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5184 dpcld(i,kk)=one/(pptop(i,kk)-pp(i,kbtmsw(i,kk+1)))
                v['dpcld'] = v['dpcld'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where(((_p361) & (v['_go']==0)), _r((v['one'] / (v['pptop'][(v['i'])-1, (v['kk'])-1] - v['pp'][(v['i'])-1, (v['kbtmsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])-1]))), v['dpcld'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==590, 0, v['_go'])
                # WRF:5186 continue
                return v
            v = lax.fori_loop(0,_p358,_p359,v)
            v['kk'] = jnp.where((_p357 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p358*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==591, 0, v['_go'])
            # WRF:5187 continue
            return v
        v = lax.fori_loop(0,_p353,_p354,v)
        v['i'] = jnp.where((_p352 & (v['_go']==0)), jnp.int32((v['myis'])+_p353*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5188 do k=1,l
        _p363 = ((jnp.bool_(True)) & (v['_go']==0))
        _p364 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
        def _p365(_p366,v):
            v = dict(v)
            v['k'] = jnp.where((_p363 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p366*(jnp.int32(1))), v['k'])
            # WRF:5189 do i=myis,myie
            _p367 = (((_p363 & (v['_go']==0))) & (v['_go']==0))
            _p368 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p369(_p370,v):
                v = dict(v)
                v['i'] = jnp.where((_p367 & (v['_go']==0)), jnp.int32((v['myis'])+_p370*(jnp.int32(1))), v['i'])
                # WRF:5190 ttdb1(i,k+1) = exp(hm1ez*min(fifty,abcff(1)*ud(i,k+1)))
                v['ttdb1'] = v['ttdb1'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p367 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(jnp.int32(1))-1] * v['ud'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))))), v['ttdb1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:5191 ttub1(i,k) = exp(hm1ez*min(fifty,abcff(1)*ur(i,k)))
                v['ttub1'] = v['ttub1'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p367 & (v['_go']==0))) & (v['_go']==0)), _r(jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(jnp.int32(1))-1] * v['ur'][(v['i'])-1, (v['k'])-1]))))), v['ttub1'][(v['i'])-1, (v['k'])-1]))
                # WRF:5192 ttd  (i,k+1) = ttdb1(i,k+1)*tdo3(i,k+1)
                v['ttd'] = v['ttd'].at[(v['i'])-1, ((v['k'] + jnp.int32(1)))-1].set(jnp.where((((_p367 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttdb1'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1] * v['tdo3'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1])), v['ttd'][(v['i'])-1, ((v['k'] + jnp.int32(1)))-1]))
                # WRF:5193 ttu  (i,k) = ttub1(i,k)*tuo3(i,k)
                v['ttu'] = v['ttu'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p367 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttub1'][(v['i'])-1, (v['k'])-1] * v['tuo3'][(v['i'])-1, (v['k'])-1])), v['ttu'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==600, 0, v['_go'])
                # WRF:5194 continue
                return v
            v = lax.fori_loop(0,_p368,_p369,v)
            v['i'] = jnp.where((_p367 & (v['_go']==0)), jnp.int32((v['myis'])+_p368*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            return v
        v = lax.fori_loop(0,_p364,_p365,v)
        v['k'] = jnp.where((_p363 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p364*(jnp.int32(1))), v['k'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5195 do i=myis,myie
        _p371 = ((jnp.bool_(True)) & (v['_go']==0))
        _p372 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p373(_p374,v):
            v = dict(v)
            v['i'] = jnp.where((_p371 & (v['_go']==0)), jnp.int32((v['myis'])+_p374*(jnp.int32(1))), v['i'])
            # WRF:5196 ttd(i,1)   = one
            v['ttd'] = v['ttd'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p371 & (v['_go']==0))) & (v['_go']==0)), _r(v['one']), v['ttd'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:5197 ttu(i,lp1) = ttd(i,lp1)
            v['ttu'] = v['ttu'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p371 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['lp1'])-1]), v['ttu'][(v['i'])-1, (v['lp1'])-1]))
            v['_go'] = jnp.where(v['_go']==610, 0, v['_go'])
            # WRF:5198 continue
            return v
        v = lax.fori_loop(0,_p372,_p373,v)
        v['i'] = jnp.where((_p371 & (v['_go']==0)), jnp.int32((v['myis'])+_p372*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5209 do i=myis,myie
        _p375 = ((jnp.bool_(True)) & (v['_go']==0))
        _p376 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p377(_p378,v):
            v = dict(v)
            v['i'] = jnp.where((_p375 & (v['_go']==0)), jnp.int32((v['myis'])+_p378*(jnp.int32(1))), v['i'])
            # WRF:5210 tdcl1 (i,1) = ttd(i,lp1)
            v['tdcl1'] = v['tdcl1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p375 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['lp1'])-1]), v['tdcl1'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:5211 tucl1 (i,1) = ttu(i,lp1)
            v['tucl1'] = v['tucl1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p375 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttu'][(v['i'])-1, (v['lp1'])-1]), v['tucl1'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:5212 tdcl2 (i,1) = tdcl1(i,1)
            v['tdcl2'] = v['tdcl2'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p375 & (v['_go']==0))) & (v['_go']==0)), _r(v['tdcl1'][(v['i'])-1, (jnp.int32(1))-1]), v['tdcl2'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:5213 dfntrn(i,1) = one/tdcl1(i,1)
            v['dfntrn'] = v['dfntrn'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p375 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['tdcl1'][(v['i'])-1, (jnp.int32(1))-1])), v['dfntrn'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:5214 ufntrn(i,1) = dfntrn(i,1)
            v['ufntrn'] = v['ufntrn'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p375 & (v['_go']==0))) & (v['_go']==0)), _r(v['dfntrn'][(v['i'])-1, (jnp.int32(1))-1]), v['ufntrn'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==620, 0, v['_go'])
            # WRF:5215 continue
            return v
        v = lax.fori_loop(0,_p376,_p377,v)
        v['i'] = jnp.where((_p375 & (v['_go']==0)), jnp.int32((v['myis'])+_p376*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5216 do i=myis,myie
        _p379 = ((jnp.bool_(True)) & (v['_go']==0))
        _p380 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p381(_p382,v):
            v = dict(v)
            v['i'] = jnp.where((_p379 & (v['_go']==0)), jnp.int32((v['myis'])+_p382*(jnp.int32(1))), v['i'])
            # WRF:5217 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p379 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5218 if(kclds.eq.0) go to 631
            _p383 = (((_p379 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5218 go to 631
            v['_go'] = jnp.where(((_p383) & (v['_go']==0)), 631, v['_go'])
            # WRF:5219 do kk=2,kclds+1
            _p384 = (((_p379 & (v['_go']==0))) & (v['_go']==0))
            _p385 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
            def _p386(_p387,v):
                v = dict(v)
                v['kk'] = jnp.where((_p384 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p387*(jnp.int32(1))), v['kk'])
                # WRF:5220 tdcl1(i,kk) = ttd(i,ktopsw(i,kk))
                v['tdcl1'] = v['tdcl1'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p384 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['ktopsw'][(v['i'])-1, (v['kk'])-1])-1]), v['tdcl1'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5221 tucl1(i,kk) = ttu(i,ktopsw(i,kk))
                v['tucl1'] = v['tucl1'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p384 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttu'][(v['i'])-1, (v['ktopsw'][(v['i'])-1, (v['kk'])-1])-1]), v['tucl1'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5222 tdcl2(i,kk) = ttd(i,kbtmsw(i,kk))
                v['tdcl2'] = v['tdcl2'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p384 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['kbtmsw'][(v['i'])-1, (v['kk'])-1])-1]), v['tdcl2'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==630, 0, v['_go'])
                # WRF:5223 continue
                return v
            v = lax.fori_loop(0,_p385,_p386,v)
            v['kk'] = jnp.where((_p384 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p385*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==631, 0, v['_go'])
            # WRF:5224 continue
            return v
        v = lax.fori_loop(0,_p380,_p381,v)
        v['i'] = jnp.where((_p379 & (v['_go']==0)), jnp.int32((v['myis'])+_p380*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5226 do i=myis,myie
        _p388 = ((jnp.bool_(True)) & (v['_go']==0))
        _p389 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p390(_p391,v):
            v = dict(v)
            v['i'] = jnp.where((_p388 & (v['_go']==0)), jnp.int32((v['myis'])+_p391*(jnp.int32(1))), v['i'])
            # WRF:5227 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p388 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5228 if(kclds.eq.0) go to 641
            _p392 = (((_p388 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5228 go to 641
            v['_go'] = jnp.where(((_p392) & (v['_go']==0)), 641, v['_go'])
            # WRF:5230 do kk=2,kclds+1
            _p393 = (((_p388 & (v['_go']==0))) & (v['_go']==0))
            _p394 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
            def _p395(_p396,v):
                v = dict(v)
                v['kk'] = jnp.where((_p393 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p396*(jnp.int32(1))), v['kk'])
                # WRF:5231 dfntrn(i,kk) = one/tdcl1(i,kk)
                v['dfntrn'] = v['dfntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p393 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['tdcl1'][(v['i'])-1, (v['kk'])-1])), v['dfntrn'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5232 ufntrn(i,kk) = one/tucl1(i,kk)
                v['ufntrn'] = v['ufntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p393 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['tucl1'][(v['i'])-1, (v['kk'])-1])), v['ufntrn'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==640, 0, v['_go'])
                # WRF:5233 continue
                return v
            v = lax.fori_loop(0,_p394,_p395,v)
            v['kk'] = jnp.where((_p393 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p394*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==641, 0, v['_go'])
            # WRF:5234 continue
            return v
        v = lax.fori_loop(0,_p389,_p390,v)
        v['i'] = jnp.where((_p388 & (v['_go']==0)), jnp.int32((v['myis'])+_p389*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5240 do i=myis,myie
        _p397 = ((jnp.bool_(True)) & (v['_go']==0))
        _p398 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p399(_p400,v):
            v = dict(v)
            v['i'] = jnp.where((_p397 & (v['_go']==0)), jnp.int32((v['myis'])+_p400*(jnp.int32(1))), v['i'])
            # WRF:5241 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p397 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5242 if(kclds.eq.0) go to 651
            _p401 = (((_p397 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5242 go to 651
            v['_go'] = jnp.where(((_p401) & (v['_go']==0)), 651, v['_go'])
            # WRF:5243 do kk=1,kclds
            _p402 = (((_p397 & (v['_go']==0))) & (v['_go']==0))
            _p403 = jnp.maximum(jnp.int32(0), ((v['kclds'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p404(_p405,v):
                v = dict(v)
                v['kk'] = jnp.where((_p402 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p405*(jnp.int32(1))), v['kk'])
                # WRF:5244 tclu(i,kk) = tdcl1(i,kk)*dfntrn(i,kk+1)*ct(i,kk+1)
                v['tclu'] = v['tclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p402 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tdcl1'][(v['i'])-1, (v['kk'])-1] * v['dfntrn'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]) * v['ct'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])), v['tclu'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5245 tcld(i,kk) = tdcl1(i,kk)/tdcl2(i,kk+1)
                v['tcld'] = v['tcld'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p402 & (v['_go']==0))) & (v['_go']==0)), _r((v['tdcl1'][(v['i'])-1, (v['kk'])-1] / v['tdcl2'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])), v['tcld'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==650, 0, v['_go'])
                # WRF:5246 continue
                return v
            v = lax.fori_loop(0,_p403,_p404,v)
            v['kk'] = jnp.where((_p402 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p403*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==651, 0, v['_go'])
            # WRF:5247 continue
            return v
        v = lax.fori_loop(0,_p398,_p399,v)
        v['i'] = jnp.where((_p397 & (v['_go']==0)), jnp.int32((v['myis'])+_p398*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5253 do i=myis,myie
        _p406 = ((jnp.bool_(True)) & (v['_go']==0))
        _p407 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p408(_p409,v):
            v = dict(v)
            v['i'] = jnp.where((_p406 & (v['_go']==0)), jnp.int32((v['myis'])+_p409*(jnp.int32(1))), v['i'])
            # WRF:5254 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p406 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5255 if(kclds.eq.0) go to 660
            _p410 = (((_p406 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5255 go to 660
            v['_go'] = jnp.where(((_p410) & (v['_go']==0)), 660, v['_go'])
            # WRF:5256 alfa (i,1)=cr(i,1)
            v['alfa'] = v['alfa'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p406 & (v['_go']==0))) & (v['_go']==0)), _r(v['cr'][(v['i'])-1, (jnp.int32(1))-1]), v['alfa'][(v['i'])-1, (jnp.int32(1))-1]))
            # WRF:5257 alfau(i,1)=zero
            v['alfau'] = v['alfau'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p406 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['alfau'][(v['i'])-1, (jnp.int32(1))-1]))
            v['_go'] = jnp.where(v['_go']==660, 0, v['_go'])
            # WRF:5258 continue
            return v
        v = lax.fori_loop(0,_p407,_p408,v)
        v['i'] = jnp.where((_p406 & (v['_go']==0)), jnp.int32((v['myis'])+_p407*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5260 do i=myis,myie
        _p411 = ((jnp.bool_(True)) & (v['_go']==0))
        _p412 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p413(_p414,v):
            v = dict(v)
            v['i'] = jnp.where((_p411 & (v['_go']==0)), jnp.int32((v['myis'])+_p414*(jnp.int32(1))), v['i'])
            # WRF:5261 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p411 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5262 if(kclds.eq.0) go to 671
            _p415 = (((_p411 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5262 go to 671
            v['_go'] = jnp.where(((_p415) & (v['_go']==0)), 671, v['_go'])
            # WRF:5263 do kk=2,kclds+1
            _p416 = (((_p411 & (v['_go']==0))) & (v['_go']==0))
            _p417 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
            def _p418(_p419,v):
                v = dict(v)
                v['kk'] = jnp.where((_p416 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p419*(jnp.int32(1))), v['kk'])
                # WRF:5264 alfau(i,kk)= tclu(i,kk-1)*tclu(i,kk-1)*alfa(i,kk-1)/ (one - tcld(i,kk-1)*tcld(i,kk-1)*alfa(i,kk-1)*cr(i,kk))
                v['alfau'] = v['alfau'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p416 & (v['_go']==0))) & (v['_go']==0)), _r((((v['tclu'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1] * v['tclu'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) * v['alfa'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) / (v['one'] - (((v['tcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1] * v['tcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) * v['alfa'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) * v['cr'][(v['i'])-1, (v['kk'])-1])))), v['alfau'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5266 alfa (i,kk)= alfau(i,kk)+cr(i,kk)
                v['alfa'] = v['alfa'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p416 & (v['_go']==0))) & (v['_go']==0)), _r((v['alfau'][(v['i'])-1, (v['kk'])-1] + v['cr'][(v['i'])-1, (v['kk'])-1])), v['alfa'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==670, 0, v['_go'])
                # WRF:5267 continue
                return v
            v = lax.fori_loop(0,_p417,_p418,v)
            v['kk'] = jnp.where((_p416 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p417*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==671, 0, v['_go'])
            # WRF:5268 continue
            return v
        v = lax.fori_loop(0,_p412,_p413,v)
        v['i'] = jnp.where((_p411 & (v['_go']==0)), jnp.int32((v['myis'])+_p412*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5276 do i=myis,myie
        _p420 = ((jnp.bool_(True)) & (v['_go']==0))
        _p421 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p422(_p423,v):
            v = dict(v)
            v['i'] = jnp.where((_p420 & (v['_go']==0)), jnp.int32((v['myis'])+_p423*(jnp.int32(1))), v['i'])
            # WRF:5277 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p420 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5278 if(kclds.eq.0) go to 680
            _p424 = (((_p420 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5278 go to 680
            v['_go'] = jnp.where(((_p424) & (v['_go']==0)), 680, v['_go'])
            # WRF:5279 ufnclu(i,kclds+1) = alfa(i,kclds+1)*tdcl1(i,kclds+1)
            v['ufnclu'] = v['ufnclu'].at[(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1].set(jnp.where((((_p420 & (v['_go']==0))) & (v['_go']==0)), _r((v['alfa'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1] * v['tdcl1'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1])), v['ufnclu'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1]))
            # WRF:5280 dfnclu(i,kclds+1) = tdcl1(i,kclds+1)
            v['dfnclu'] = v['dfnclu'].at[(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1].set(jnp.where((((_p420 & (v['_go']==0))) & (v['_go']==0)), _r(v['tdcl1'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1]), v['dfnclu'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1]))
            v['_go'] = jnp.where(v['_go']==680, 0, v['_go'])
            # WRF:5281 continue
            return v
        v = lax.fori_loop(0,_p421,_p422,v)
        v['i'] = jnp.where((_p420 & (v['_go']==0)), jnp.int32((v['myis'])+_p421*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5284 do i=myis,myie
        _p425 = ((jnp.bool_(True)) & (v['_go']==0))
        _p426 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p427(_p428,v):
            v = dict(v)
            v['i'] = jnp.where((_p425 & (v['_go']==0)), jnp.int32((v['myis'])+_p428*(jnp.int32(1))), v['i'])
            # WRF:5285 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p425 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5286 if(kclds.eq.0) go to 691
            _p429 = (((_p425 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5286 go to 691
            v['_go'] = jnp.where(((_p429) & (v['_go']==0)), 691, v['_go'])
            # WRF:5287 do kk=kclds,1,-1
            _p430 = (((_p425 & (v['_go']==0))) & (v['_go']==0))
            _p431 = jnp.maximum(jnp.int32(0), ((jnp.int32(1))-(v['kclds']))//(jnp.int32(-1))+jnp.int32(1))
            def _p432(_p433,v):
                v = dict(v)
                v['kk'] = jnp.where((_p430 & (v['_go']==0)), jnp.int32((v['kclds'])+_p433*(jnp.int32(-1))), v['kk'])
                # WRF:5288 ufnclu(i,kk) = ufnclu(i,kk+1)*alfau(i,kk+1)/(alfa(i,kk+1)* tclu(i,kk))
                v['ufnclu'] = v['ufnclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p430 & (v['_go']==0))) & (v['_go']==0)), _r(((v['ufnclu'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] * v['alfau'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]) / (v['alfa'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] * v['tclu'][(v['i'])-1, (v['kk'])-1]))), v['ufnclu'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5290 dfnclu(i,kk) = ufnclu(i,kk)/alfa(i,kk)
                v['dfnclu'] = v['dfnclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p430 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufnclu'][(v['i'])-1, (v['kk'])-1] / v['alfa'][(v['i'])-1, (v['kk'])-1])), v['dfnclu'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==690, 0, v['_go'])
                # WRF:5291 continue
                return v
            v = lax.fori_loop(0,_p431,_p432,v)
            v['kk'] = jnp.where((_p430 & (v['_go']==0)), jnp.int32((v['kclds'])+_p431*(jnp.int32(-1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==691, 0, v['_go'])
            # WRF:5292 continue
            return v
        v = lax.fori_loop(0,_p426,_p427,v)
        v['i'] = jnp.where((_p425 & (v['_go']==0)), jnp.int32((v['myis'])+_p426*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5293 do i=myis,myie
        _p434 = ((jnp.bool_(True)) & (v['_go']==0))
        _p435 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p436(_p437,v):
            v = dict(v)
            v['i'] = jnp.where((_p434 & (v['_go']==0)), jnp.int32((v['myis'])+_p437*(jnp.int32(1))), v['i'])
            # WRF:5294 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p434 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5295 if(kclds.eq.0) go to 701
            _p438 = (((_p434 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5295 go to 701
            v['_go'] = jnp.where(((_p438) & (v['_go']==0)), 701, v['_go'])
            # WRF:5296 do kk=1,kclds+1
            _p439 = (((_p434 & (v['_go']==0))) & (v['_go']==0))
            _p440 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p441(_p442,v):
                v = dict(v)
                v['kk'] = jnp.where((_p439 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p442*(jnp.int32(1))), v['kk'])
                # WRF:5297 ufntrn(i,kk) = ufnclu(i,kk)*ufntrn(i,kk)
                v['ufntrn'] = v['ufntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p439 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufnclu'][(v['i'])-1, (v['kk'])-1] * v['ufntrn'][(v['i'])-1, (v['kk'])-1])), v['ufntrn'][(v['i'])-1, (v['kk'])-1]))
                # WRF:5298 dfntrn(i,kk) = dfnclu(i,kk)*dfntrn(i,kk)
                v['dfntrn'] = v['dfntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p439 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfnclu'][(v['i'])-1, (v['kk'])-1] * v['dfntrn'][(v['i'])-1, (v['kk'])-1])), v['dfntrn'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==700, 0, v['_go'])
                # WRF:5299 continue
                return v
            v = lax.fori_loop(0,_p440,_p441,v)
            v['kk'] = jnp.where((_p439 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p440*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==701, 0, v['_go'])
            # WRF:5300 continue
            return v
        v = lax.fori_loop(0,_p435,_p436,v)
        v['i'] = jnp.where((_p434 & (v['_go']==0)), jnp.int32((v['myis'])+_p435*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5302 do i=myis,myie
        _p443 = ((jnp.bool_(True)) & (v['_go']==0))
        _p444 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p445(_p446,v):
            v = dict(v)
            v['i'] = jnp.where((_p443 & (v['_go']==0)), jnp.int32((v['myis'])+_p446*(jnp.int32(1))), v['i'])
            # WRF:5303 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p443 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5304 if(kclds.eq.0) go to 720
            _p447 = (((_p443 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5304 go to 720
            v['_go'] = jnp.where(((_p447) & (v['_go']==0)), 720, v['_go'])
            # WRF:5305 j2=kbtmsw(i,2)
            v['j2'] = jnp.where((((_p443 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtmsw'][(v['i'])-1, (jnp.int32(2))-1]), v['j2'])
            # WRF:5306 do k=j2,lp1
            _p448 = (((_p443 & (v['_go']==0))) & (v['_go']==0))
            _p449 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(v['j2']))//(jnp.int32(1))+jnp.int32(1))
            def _p450(_p451,v):
                v = dict(v)
                v['k'] = jnp.where((_p448 & (v['_go']==0)), jnp.int32((v['j2'])+_p451*(jnp.int32(1))), v['k'])
                # WRF:5307 ufn(i,k) = ufntrn(i,1)*ttu(i,k)
                v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p448 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufntrn'][(v['i'])-1, (jnp.int32(1))-1] * v['ttu'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                # WRF:5308 dfn(i,k) = dfntrn(i,1)*ttd(i,k)
                v['dfn'] = v['dfn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p448 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfntrn'][(v['i'])-1, (jnp.int32(1))-1] * v['ttd'][(v['i'])-1, (v['k'])-1])), v['dfn'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==710, 0, v['_go'])
                # WRF:5309 continue
                return v
            v = lax.fori_loop(0,_p449,_p450,v)
            v['k'] = jnp.where((_p448 & (v['_go']==0)), jnp.int32((v['j2'])+_p449*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==720, 0, v['_go'])
            # WRF:5310 continue
            return v
        v = lax.fori_loop(0,_p444,_p445,v)
        v['i'] = jnp.where((_p443 & (v['_go']==0)), jnp.int32((v['myis'])+_p444*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5312 do i=myis,myie
        _p452 = ((jnp.bool_(True)) & (v['_go']==0))
        _p453 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p454(_p455,v):
            v = dict(v)
            v['i'] = jnp.where((_p452 & (v['_go']==0)), jnp.int32((v['myis'])+_p455*(jnp.int32(1))), v['i'])
            # WRF:5313 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p452 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5314 if(kclds.eq.0) go to 760
            _p456 = (((_p452 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5314 go to 760
            v['_go'] = jnp.where(((_p456) & (v['_go']==0)), 760, v['_go'])
            # WRF:5315 do kk=2,kclds+1
            _p457 = (((_p452 & (v['_go']==0))) & (v['_go']==0))
            _p458 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
            def _p459(_p460,v):
                v = dict(v)
                v['kk'] = jnp.where((_p457 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p460*(jnp.int32(1))), v['kk'])
                # WRF:5316 j1=ktopsw(i,kk)
                v['j1'] = jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['ktopsw'][(v['i'])-1, (v['kk'])-1]), v['j1'])
                # WRF:5317 j2=kbtmsw(i,kk+1)
                v['j2'] = jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtmsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]), v['j2'])
                # WRF:5318 if (j1.eq.1) go to 755
                _p461 = (((_p457 & (v['_go']==0))) & (v['_go']==0)) & ((v['j1'] == jnp.int32(1)))
                # WRF:5318 go to 755
                v['_go'] = jnp.where(((_p461) & (v['_go']==0)), 755, v['_go'])
                # WRF:5319 do k=j2,j1
                _p462 = (((_p457 & (v['_go']==0))) & (v['_go']==0))
                _p463 = jnp.maximum(jnp.int32(0), ((v['j1'])-(v['j2']))//(jnp.int32(1))+jnp.int32(1))
                def _p464(_p465,v):
                    v = dict(v)
                    v['k'] = jnp.where((_p462 & (v['_go']==0)), jnp.int32((v['j2'])+_p465*(jnp.int32(1))), v['k'])
                    # WRF:5320 ufn(i,k) = ufntrn(i,kk)*ttu(i,k)
                    v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p462 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufntrn'][(v['i'])-1, (v['kk'])-1] * v['ttu'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                    # WRF:5321 dfn(i,k) = dfntrn(i,kk)*ttd(i,k)
                    v['dfn'] = v['dfn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p462 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfntrn'][(v['i'])-1, (v['kk'])-1] * v['ttd'][(v['i'])-1, (v['k'])-1])), v['dfn'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==730, 0, v['_go'])
                    # WRF:5322 continue
                    return v
                v = lax.fori_loop(0,_p463,_p464,v)
                v['k'] = jnp.where((_p462 & (v['_go']==0)), jnp.int32((v['j2'])+_p463*(jnp.int32(1))), v['k'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                # WRF:5326 j3=kbtmsw(i,kk)
                v['j3'] = jnp.where((((_p457 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtmsw'][(v['i'])-1, (v['kk'])-1]), v['j3'])
                # WRF:5327 __if__ (j3-j1).gt.1
                _p466 = (((_p457 & (v['_go']==0))) & (v['_go']==0)) & (((v['j3'] - v['j1']) > jnp.int32(1)))
                _p467 = (((_p457 & (v['_go']==0))) & (v['_go']==0)) & ~(((v['j3'] - v['j1']) > jnp.int32(1)))
                # WRF:5328 tempf = (ufnclu(i,kk)-ufn(i,j3))*dpcld(i,kk-1)
                v['tempf'] = jnp.where(((_p466) & (v['_go']==0)), _r(((v['ufnclu'][(v['i'])-1, (v['kk'])-1] - v['ufn'][(v['i'])-1, (v['j3'])-1]) * v['dpcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])), v['tempf'])
                # WRF:5329 tempg = (dfnclu(i,kk)-dfn(i,j3))*dpcld(i,kk-1)
                v['tempg'] = jnp.where(((_p466) & (v['_go']==0)), _r(((v['dfnclu'][(v['i'])-1, (v['kk'])-1] - v['dfn'][(v['i'])-1, (v['j3'])-1]) * v['dpcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])), v['tempg'])
                # WRF:5330 do k=j1+1,j3-1
                _p468 = ((_p466) & (v['_go']==0))
                _p469 = jnp.maximum(jnp.int32(0), (((v['j3'] - jnp.int32(1)))-((v['j1'] + jnp.int32(1))))//(jnp.int32(1))+jnp.int32(1))
                def _p470(_p471,v):
                    v = dict(v)
                    v['k'] = jnp.where((_p468 & (v['_go']==0)), jnp.int32(((v['j1'] + jnp.int32(1)))+_p471*(jnp.int32(1))), v['k'])
                    # WRF:5331 ufn(i,k) = ufnclu(i,kk)+tempf*(pp(i,k)-pptop(i,kk-1))
                    v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p468 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufnclu'][(v['i'])-1, (v['kk'])-1] + (v['tempf'] * (v['pp'][(v['i'])-1, (v['k'])-1] - v['pptop'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])))), v['ufn'][(v['i'])-1, (v['k'])-1]))
                    # WRF:5332 dfn(i,k) = dfnclu(i,kk)+tempg*(pp(i,k)-pptop(i,kk-1))
                    v['dfn'] = v['dfn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p468 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfnclu'][(v['i'])-1, (v['kk'])-1] + (v['tempg'] * (v['pp'][(v['i'])-1, (v['k'])-1] - v['pptop'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])))), v['dfn'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==740, 0, v['_go'])
                    # WRF:5333 continue
                    return v
                v = lax.fori_loop(0,_p469,_p470,v)
                v['k'] = jnp.where((_p468 & (v['_go']==0)), jnp.int32(((v['j1'] + jnp.int32(1)))+_p469*(jnp.int32(1))), v['k'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==755, 0, v['_go'])
                # WRF:5335 continue
                return v
            v = lax.fori_loop(0,_p458,_p459,v)
            v['kk'] = jnp.where((_p457 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p458*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==760, 0, v['_go'])
            # WRF:5336 continue
            return v
        v = lax.fori_loop(0,_p453,_p454,v)
        v['i'] = jnp.where((_p452 & (v['_go']==0)), jnp.int32((v['myis'])+_p453*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5337 do i=myis,myie
        _p472 = ((jnp.bool_(True)) & (v['_go']==0))
        _p473 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p474(_p475,v):
            v = dict(v)
            v['i'] = jnp.where((_p472 & (v['_go']==0)), jnp.int32((v['myis'])+_p475*(jnp.int32(1))), v['i'])
            # WRF:5338 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p472 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5339 if(kclds.eq.0) go to 770
            _p476 = (((_p472 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5339 go to 770
            v['_go'] = jnp.where(((_p476) & (v['_go']==0)), 770, v['_go'])
            # WRF:5340 do k=1,lp1
            _p477 = (((_p472 & (v['_go']==0))) & (v['_go']==0))
            _p478 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p479(_p480,v):
                v = dict(v)
                v['k'] = jnp.where((_p477 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p480*(jnp.int32(1))), v['k'])
                # WRF:5341 dfswc(i,k) = dfn(i,k)*dfntop(i,1)
                v['dfswc'] = v['dfswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p477 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfn'][(v['i'])-1, (v['k'])-1] * v['dfntop'][(v['i'])-1, (jnp.int32(1))-1])), v['dfswc'][(v['i'])-1, (v['k'])-1]))
                # WRF:5342 ufswc(i,k) = ufn(i,k)*dfntop(i,1)
                v['ufswc'] = v['ufswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p477 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufn'][(v['i'])-1, (v['k'])-1] * v['dfntop'][(v['i'])-1, (jnp.int32(1))-1])), v['ufswc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==771, 0, v['_go'])
                # WRF:5343 continue
                return v
            v = lax.fori_loop(0,_p478,_p479,v)
            v['k'] = jnp.where((_p477 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p478*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==770, 0, v['_go'])
            # WRF:5344 continue
            return v
        v = lax.fori_loop(0,_p473,_p474,v)
        v['i'] = jnp.where((_p472 & (v['_go']==0)), jnp.int32((v['myis'])+_p473*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5345 do i=myis,myie
        _p481 = ((jnp.bool_(True)) & (v['_go']==0))
        _p482 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p483(_p484,v):
            v = dict(v)
            v['i'] = jnp.where((_p481 & (v['_go']==0)), jnp.int32((v['myis'])+_p484*(jnp.int32(1))), v['i'])
            # WRF:5346 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p481 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5347 if(kclds.eq.0) go to 780
            _p485 = (((_p481 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5347 go to 780
            v['_go'] = jnp.where(((_p485) & (v['_go']==0)), 780, v['_go'])
            # WRF:5348 tmp1(i) = one - ccmax(i)
            v['tmp1'] = v['tmp1'].at[(v['i'])-1].set(jnp.where((((_p481 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - v['ccmax'][(v['i'])-1])), v['tmp1'][(v['i'])-1]))
            # WRF:5349 gdfvb(i) = tmp1(i)*gdfvb(i)
            v['gdfvb'] = v['gdfvb'].at[(v['i'])-1].set(jnp.where((((_p481 & (v['_go']==0))) & (v['_go']==0)), _r((v['tmp1'][(v['i'])-1] * v['gdfvb'][(v['i'])-1])), v['gdfvb'][(v['i'])-1]))
            # WRF:5350 gdfnb(i) = tmp1(i)*gdfnb(i)
            v['gdfnb'] = v['gdfnb'].at[(v['i'])-1].set(jnp.where((((_p481 & (v['_go']==0))) & (v['_go']==0)), _r((v['tmp1'][(v['i'])-1] * v['gdfnb'][(v['i'])-1])), v['gdfnb'][(v['i'])-1]))
            # WRF:5351 gdfvd(i) = tmp1(i)*gdfvd(i) + ccmax(i)*dfswc(i,lp1)
            v['gdfvd'] = v['gdfvd'].at[(v['i'])-1].set(jnp.where((((_p481 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tmp1'][(v['i'])-1] * v['gdfvd'][(v['i'])-1]) + (v['ccmax'][(v['i'])-1] * v['dfswc'][(v['i'])-1, (v['lp1'])-1]))), v['gdfvd'][(v['i'])-1]))
            v['_go'] = jnp.where(v['_go']==780, 0, v['_go'])
            # WRF:5352 continue
            return v
        v = lax.fori_loop(0,_p482,_p483,v)
        v['i'] = jnp.where((_p481 & (v['_go']==0)), jnp.int32((v['myis'])+_p482*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5358 do n=2,nb
        _p486 = ((jnp.bool_(True)) & (v['_go']==0))
        _p487 = jnp.maximum(jnp.int32(0), ((jnp.int32(jnp.int32(12)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
        def _p488(_p489,v):
            v = dict(v)
            v['n'] = jnp.where((_p486 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p489*(jnp.int32(1))), v['n'])
            # WRF:5360 do i=myis,myie
            _p490 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p491 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p492(_p493,v):
                v = dict(v)
                v['i'] = jnp.where((_p490 & (v['_go']==0)), jnp.int32((v['myis'])+_p493*(jnp.int32(1))), v['i'])
                # WRF:5361 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p490 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5362 if(kclds.eq.0) go to 791
                _p494 = (((_p490 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5362 go to 791
                v['_go'] = jnp.where(((_p494) & (v['_go']==0)), 791, v['_go'])
                # WRF:5363 do k=1,kclds+1
                _p495 = (((_p490 & (v['_go']==0))) & (v['_go']==0))
                _p496 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
                def _p497(_p498,v):
                    v = dict(v)
                    v['k'] = jnp.where((_p495 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p498*(jnp.int32(1))), v['k'])
                    # WRF:5364 cr(i,k) = crr(i,n,k)*xamt(i,k)
                    v['cr'] = v['cr'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p495 & (v['_go']==0))) & (v['_go']==0)), _r((v['crr'][(v['i'])-1, (v['n'])-1, (v['k'])-1] * v['xamt'][(v['i'])-1, (v['k'])-1])), v['cr'][(v['i'])-1, (v['k'])-1]))
                    # WRF:5365 ct(i,k) = one - (one-ctt(i,n,k))*xamt(i,k)
                    v['ct'] = v['ct'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p495 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] - ((v['one'] - v['ctt'][(v['i'])-1, (v['n'])-1, (v['k'])-1]) * v['xamt'][(v['i'])-1, (v['k'])-1]))), v['ct'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==790, 0, v['_go'])
                    # WRF:5366 continue
                    return v
                v = lax.fori_loop(0,_p496,_p497,v)
                v['k'] = jnp.where((_p495 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p496*(jnp.int32(1))), v['k'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==791, 0, v['_go'])
                # WRF:5367 continue
                return v
            v = lax.fori_loop(0,_p491,_p492,v)
            v['i'] = jnp.where((_p490 & (v['_go']==0)), jnp.int32((v['myis'])+_p491*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5369 __if__ n.eq.2
            _p499 = (((_p486 & (v['_go']==0))) & (v['_go']==0)) & ((v['n'] == jnp.int32(2)))
            _p500 = (((_p486 & (v['_go']==0))) & (v['_go']==0)) & ~((v['n'] == jnp.int32(2)))
            # WRF:5372 do i=myis,myie
            _p501 = ((_p499) & (v['_go']==0))
            _p502 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p503(_p504,v):
                v = dict(v)
                v['i'] = jnp.where((_p501 & (v['_go']==0)), jnp.int32((v['myis'])+_p504*(jnp.int32(1))), v['i'])
                # WRF:5373 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p501 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5374 if(kclds.eq.0) go to 800
                _p505 = (((_p501 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5374 go to 800
                v['_go'] = jnp.where(((_p505) & (v['_go']==0)), 800, v['_go'])
                # WRF:5375 do kk=2,lp1
                _p506 = (((_p501 & (v['_go']==0))) & (v['_go']==0))
                _p507 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
                def _p508(_p509,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p506 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p509*(jnp.int32(1))), v['kk'])
                    # WRF:5376 ttd(i,kk) = ttdb1(i,kk)*tdco2(i,kk)
                    v['ttd'] = v['ttd'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p506 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttdb1'][(v['i'])-1, (v['kk'])-1] * v['tdco2'][(v['i'])-1, (v['kk'])-1])), v['ttd'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==801, 0, v['_go'])
                    # WRF:5377 continue
                    return v
                v = lax.fori_loop(0,_p507,_p508,v)
                v['kk'] = jnp.where((_p506 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p507*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                # WRF:5378 do kk=1,l
                _p510 = (((_p501 & (v['_go']==0))) & (v['_go']==0))
                _p511 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
                def _p512(_p513,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p510 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p513*(jnp.int32(1))), v['kk'])
                    # WRF:5379 ttu(i,kk) = ttub1(i,kk)*tuco2(i,kk)
                    v['ttu'] = v['ttu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p510 & (v['_go']==0))) & (v['_go']==0)), _r((v['ttub1'][(v['i'])-1, (v['kk'])-1] * v['tuco2'][(v['i'])-1, (v['kk'])-1])), v['ttu'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==802, 0, v['_go'])
                    # WRF:5380 continue
                    return v
                v = lax.fori_loop(0,_p511,_p512,v)
                v['kk'] = jnp.where((_p510 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p511*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==800, 0, v['_go'])
                # WRF:5381 continue
                return v
            v = lax.fori_loop(0,_p502,_p503,v)
            v['i'] = jnp.where((_p501 & (v['_go']==0)), jnp.int32((v['myis'])+_p502*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5383 do i=myis,myie
            _p514 = ((_p500) & (v['_go']==0))
            _p515 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p516(_p517,v):
                v = dict(v)
                v['i'] = jnp.where((_p514 & (v['_go']==0)), jnp.int32((v['myis'])+_p517*(jnp.int32(1))), v['i'])
                # WRF:5384 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p514 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5385 if(kclds.eq.0) go to 810
                _p518 = (((_p514 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5385 go to 810
                v['_go'] = jnp.where(((_p518) & (v['_go']==0)), 810, v['_go'])
                # WRF:5386 do kk=2,lp1
                _p519 = (((_p514 & (v['_go']==0))) & (v['_go']==0))
                _p520 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
                def _p521(_p522,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p519 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p522*(jnp.int32(1))), v['kk'])
                    # WRF:5387 ttd(i,kk) = exp(hm1ez*min(fifty,abcff(n)*ud(i,kk))) * tdco2(i,kk)
                    v['ttd'] = v['ttd'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p519 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(v['n'])-1] * v['ud'][(v['i'])-1, (v['kk'])-1])))) * v['tdco2'][(v['i'])-1, (v['kk'])-1])), v['ttd'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==811, 0, v['_go'])
                    # WRF:5389 continue
                    return v
                v = lax.fori_loop(0,_p520,_p521,v)
                v['kk'] = jnp.where((_p519 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p520*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                # WRF:5390 do kk=1,l
                _p523 = (((_p514 & (v['_go']==0))) & (v['_go']==0))
                _p524 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
                def _p525(_p526,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p523 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p526*(jnp.int32(1))), v['kk'])
                    # WRF:5391 ttu(i,kk) = exp(hm1ez*min(fifty,abcff(n)*ur(i,kk))) * tuco2(i,kk)
                    v['ttu'] = v['ttu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p523 & (v['_go']==0))) & (v['_go']==0)), _r((jnp.exp((v['hm1ez'] * _min(v['fifty'], (v['abcff'][(v['n'])-1] * v['ur'][(v['i'])-1, (v['kk'])-1])))) * v['tuco2'][(v['i'])-1, (v['kk'])-1])), v['ttu'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==812, 0, v['_go'])
                    # WRF:5393 continue
                    return v
                v = lax.fori_loop(0,_p524,_p525,v)
                v['kk'] = jnp.where((_p523 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p524*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==810, 0, v['_go'])
                # WRF:5394 continue
                return v
            v = lax.fori_loop(0,_p515,_p516,v)
            v['i'] = jnp.where((_p514 & (v['_go']==0)), jnp.int32((v['myis'])+_p515*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5398 do i=myis,myie
            _p527 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p528 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p529(_p530,v):
                v = dict(v)
                v['i'] = jnp.where((_p527 & (v['_go']==0)), jnp.int32((v['myis'])+_p530*(jnp.int32(1))), v['i'])
                # WRF:5399 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p527 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5400 if(kclds.eq.0) go to 820
                _p531 = (((_p527 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5400 go to 820
                v['_go'] = jnp.where(((_p531) & (v['_go']==0)), 820, v['_go'])
                # WRF:5401 ttu(i,lp1) = ttd(i,lp1)
                v['ttu'] = v['ttu'].at[(v['i'])-1, (v['lp1'])-1].set(jnp.where((((_p527 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['lp1'])-1]), v['ttu'][(v['i'])-1, (v['lp1'])-1]))
                # WRF:5402 ttd(i,1)   = one
                v['ttd'] = v['ttd'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p527 & (v['_go']==0))) & (v['_go']==0)), _r(v['one']), v['ttd'][(v['i'])-1, (jnp.int32(1))-1]))
                v['_go'] = jnp.where(v['_go']==820, 0, v['_go'])
                # WRF:5403 continue
                return v
            v = lax.fori_loop(0,_p528,_p529,v)
            v['i'] = jnp.where((_p527 & (v['_go']==0)), jnp.int32((v['myis'])+_p528*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5414 do i=myis,myie
            _p532 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p533 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p534(_p535,v):
                v = dict(v)
                v['i'] = jnp.where((_p532 & (v['_go']==0)), jnp.int32((v['myis'])+_p535*(jnp.int32(1))), v['i'])
                # WRF:5415 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p532 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5416 if(kclds.eq.0) go to 830
                _p536 = (((_p532 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5416 go to 830
                v['_go'] = jnp.where(((_p536) & (v['_go']==0)), 830, v['_go'])
                # WRF:5417 tdcl1 (i,1) = ttd(i,lp1)
                v['tdcl1'] = v['tdcl1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p532 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['lp1'])-1]), v['tdcl1'][(v['i'])-1, (jnp.int32(1))-1]))
                # WRF:5418 tucl1 (i,1) = ttu(i,lp1)
                v['tucl1'] = v['tucl1'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p532 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttu'][(v['i'])-1, (v['lp1'])-1]), v['tucl1'][(v['i'])-1, (jnp.int32(1))-1]))
                # WRF:5419 tdcl2 (i,1) = tdcl1(i,1)
                v['tdcl2'] = v['tdcl2'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p532 & (v['_go']==0))) & (v['_go']==0)), _r(v['tdcl1'][(v['i'])-1, (jnp.int32(1))-1]), v['tdcl2'][(v['i'])-1, (jnp.int32(1))-1]))
                # WRF:5420 dfntrn(i,1) = one/tdcl1(i,1)
                v['dfntrn'] = v['dfntrn'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p532 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['tdcl1'][(v['i'])-1, (jnp.int32(1))-1])), v['dfntrn'][(v['i'])-1, (jnp.int32(1))-1]))
                # WRF:5421 ufntrn(i,1) = dfntrn(i,1)
                v['ufntrn'] = v['ufntrn'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p532 & (v['_go']==0))) & (v['_go']==0)), _r(v['dfntrn'][(v['i'])-1, (jnp.int32(1))-1]), v['ufntrn'][(v['i'])-1, (jnp.int32(1))-1]))
                v['_go'] = jnp.where(v['_go']==830, 0, v['_go'])
                # WRF:5422 continue
                return v
            v = lax.fori_loop(0,_p533,_p534,v)
            v['i'] = jnp.where((_p532 & (v['_go']==0)), jnp.int32((v['myis'])+_p533*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5423 do i=myis,myie
            _p537 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p538 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p539(_p540,v):
                v = dict(v)
                v['i'] = jnp.where((_p537 & (v['_go']==0)), jnp.int32((v['myis'])+_p540*(jnp.int32(1))), v['i'])
                # WRF:5424 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p537 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5425 if(kclds.eq.0) go to 841
                _p541 = (((_p537 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5425 go to 841
                v['_go'] = jnp.where(((_p541) & (v['_go']==0)), 841, v['_go'])
                # WRF:5426 do kk=2,kclds+1
                _p542 = (((_p537 & (v['_go']==0))) & (v['_go']==0))
                _p543 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
                def _p544(_p545,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p542 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p545*(jnp.int32(1))), v['kk'])
                    # WRF:5427 tdcl1(i,kk) = ttd(i,ktopsw(i,kk))
                    v['tdcl1'] = v['tdcl1'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p542 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['ktopsw'][(v['i'])-1, (v['kk'])-1])-1]), v['tdcl1'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5428 tucl1(i,kk) = ttu(i,ktopsw(i,kk))
                    v['tucl1'] = v['tucl1'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p542 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttu'][(v['i'])-1, (v['ktopsw'][(v['i'])-1, (v['kk'])-1])-1]), v['tucl1'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5429 tdcl2(i,kk) = ttd(i,kbtmsw(i,kk))
                    v['tdcl2'] = v['tdcl2'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p542 & (v['_go']==0))) & (v['_go']==0)), _r(v['ttd'][(v['i'])-1, (v['kbtmsw'][(v['i'])-1, (v['kk'])-1])-1]), v['tdcl2'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==840, 0, v['_go'])
                    # WRF:5430 continue
                    return v
                v = lax.fori_loop(0,_p543,_p544,v)
                v['kk'] = jnp.where((_p542 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p543*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==841, 0, v['_go'])
                # WRF:5431 continue
                return v
            v = lax.fori_loop(0,_p538,_p539,v)
            v['i'] = jnp.where((_p537 & (v['_go']==0)), jnp.int32((v['myis'])+_p538*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5432 do i=myis,myie
            _p546 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p547 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p548(_p549,v):
                v = dict(v)
                v['i'] = jnp.where((_p546 & (v['_go']==0)), jnp.int32((v['myis'])+_p549*(jnp.int32(1))), v['i'])
                # WRF:5433 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p546 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5434 if(kclds.eq.0) go to 851
                _p550 = (((_p546 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5434 go to 851
                v['_go'] = jnp.where(((_p550) & (v['_go']==0)), 851, v['_go'])
                # WRF:5435 do kk=2,kclds+1
                _p551 = (((_p546 & (v['_go']==0))) & (v['_go']==0))
                _p552 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
                def _p553(_p554,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p551 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p554*(jnp.int32(1))), v['kk'])
                    # WRF:5436 dfntrn(i,kk) = one/tdcl1(i,kk)
                    v['dfntrn'] = v['dfntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p551 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['tdcl1'][(v['i'])-1, (v['kk'])-1])), v['dfntrn'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5437 ufntrn(i,kk) = one/tucl1(i,kk)
                    v['ufntrn'] = v['ufntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p551 & (v['_go']==0))) & (v['_go']==0)), _r((v['one'] / v['tucl1'][(v['i'])-1, (v['kk'])-1])), v['ufntrn'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==850, 0, v['_go'])
                    # WRF:5438 continue
                    return v
                v = lax.fori_loop(0,_p552,_p553,v)
                v['kk'] = jnp.where((_p551 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p552*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==851, 0, v['_go'])
                # WRF:5439 continue
                return v
            v = lax.fori_loop(0,_p547,_p548,v)
            v['i'] = jnp.where((_p546 & (v['_go']==0)), jnp.int32((v['myis'])+_p547*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5440 do i=myis,myie
            _p555 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p556 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p557(_p558,v):
                v = dict(v)
                v['i'] = jnp.where((_p555 & (v['_go']==0)), jnp.int32((v['myis'])+_p558*(jnp.int32(1))), v['i'])
                # WRF:5441 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p555 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5442 if(kclds.eq.0) go to 861
                _p559 = (((_p555 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5442 go to 861
                v['_go'] = jnp.where(((_p559) & (v['_go']==0)), 861, v['_go'])
                # WRF:5443 do kk=1,kclds
                _p560 = (((_p555 & (v['_go']==0))) & (v['_go']==0))
                _p561 = jnp.maximum(jnp.int32(0), ((v['kclds'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
                def _p562(_p563,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p560 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p563*(jnp.int32(1))), v['kk'])
                    # WRF:5444 tclu(i,kk) = tdcl1(i,kk)*dfntrn(i,kk+1)*ct(i,kk+1)
                    v['tclu'] = v['tclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p560 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tdcl1'][(v['i'])-1, (v['kk'])-1] * v['dfntrn'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]) * v['ct'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])), v['tclu'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5445 tcld(i,kk) = tdcl1(i,kk)/tdcl2(i,kk+1)
                    v['tcld'] = v['tcld'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p560 & (v['_go']==0))) & (v['_go']==0)), _r((v['tdcl1'][(v['i'])-1, (v['kk'])-1] / v['tdcl2'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1])), v['tcld'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==860, 0, v['_go'])
                    # WRF:5446 continue
                    return v
                v = lax.fori_loop(0,_p561,_p562,v)
                v['kk'] = jnp.where((_p560 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p561*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==861, 0, v['_go'])
                # WRF:5447 continue
                return v
            v = lax.fori_loop(0,_p556,_p557,v)
            v['i'] = jnp.where((_p555 & (v['_go']==0)), jnp.int32((v['myis'])+_p556*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5452 do i=myis,myie
            _p564 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p565 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p566(_p567,v):
                v = dict(v)
                v['i'] = jnp.where((_p564 & (v['_go']==0)), jnp.int32((v['myis'])+_p567*(jnp.int32(1))), v['i'])
                # WRF:5453 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p564 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5454 if(kclds.eq.0) go to 870
                _p568 = (((_p564 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5454 go to 870
                v['_go'] = jnp.where(((_p568) & (v['_go']==0)), 870, v['_go'])
                # WRF:5455 alfa (i,1) = cr(i,1)
                v['alfa'] = v['alfa'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p564 & (v['_go']==0))) & (v['_go']==0)), _r(v['cr'][(v['i'])-1, (jnp.int32(1))-1]), v['alfa'][(v['i'])-1, (jnp.int32(1))-1]))
                # WRF:5456 alfau(i,1) = zero
                v['alfau'] = v['alfau'].at[(v['i'])-1, (jnp.int32(1))-1].set(jnp.where((((_p564 & (v['_go']==0))) & (v['_go']==0)), _r(v['zero']), v['alfau'][(v['i'])-1, (jnp.int32(1))-1]))
                v['_go'] = jnp.where(v['_go']==870, 0, v['_go'])
                # WRF:5457 continue
                return v
            v = lax.fori_loop(0,_p565,_p566,v)
            v['i'] = jnp.where((_p564 & (v['_go']==0)), jnp.int32((v['myis'])+_p565*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5459 do i=myis,myie
            _p569 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p570 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p571(_p572,v):
                v = dict(v)
                v['i'] = jnp.where((_p569 & (v['_go']==0)), jnp.int32((v['myis'])+_p572*(jnp.int32(1))), v['i'])
                # WRF:5460 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p569 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5461 if(kclds.eq.0) go to 881
                _p573 = (((_p569 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5461 go to 881
                v['_go'] = jnp.where(((_p573) & (v['_go']==0)), 881, v['_go'])
                # WRF:5462 do kk=2,kclds+1
                _p574 = (((_p569 & (v['_go']==0))) & (v['_go']==0))
                _p575 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
                def _p576(_p577,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p574 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p577*(jnp.int32(1))), v['kk'])
                    # WRF:5463 alfau(i,kk) = tclu(i,kk-1)*tclu(i,kk-1)*alfa(i,kk-1)/(one - tcld(i,kk-1)*tcld(i,kk-1)*alfa(i,kk-1)*cr(i,kk))
                    v['alfau'] = v['alfau'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p574 & (v['_go']==0))) & (v['_go']==0)), _r((((v['tclu'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1] * v['tclu'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) * v['alfa'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) / (v['one'] - (((v['tcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1] * v['tcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) * v['alfa'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1]) * v['cr'][(v['i'])-1, (v['kk'])-1])))), v['alfau'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5465 alfa (i,kk) = alfau(i,kk)+cr(i,kk)
                    v['alfa'] = v['alfa'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p574 & (v['_go']==0))) & (v['_go']==0)), _r((v['alfau'][(v['i'])-1, (v['kk'])-1] + v['cr'][(v['i'])-1, (v['kk'])-1])), v['alfa'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==880, 0, v['_go'])
                    # WRF:5466 continue
                    return v
                v = lax.fori_loop(0,_p575,_p576,v)
                v['kk'] = jnp.where((_p574 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p575*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==881, 0, v['_go'])
                # WRF:5467 continue
                return v
            v = lax.fori_loop(0,_p570,_p571,v)
            v['i'] = jnp.where((_p569 & (v['_go']==0)), jnp.int32((v['myis'])+_p570*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5475 do i=myis,myie
            _p578 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p579 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p580(_p581,v):
                v = dict(v)
                v['i'] = jnp.where((_p578 & (v['_go']==0)), jnp.int32((v['myis'])+_p581*(jnp.int32(1))), v['i'])
                # WRF:5476 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p578 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5477 if(kclds.eq.0) go to 890
                _p582 = (((_p578 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5477 go to 890
                v['_go'] = jnp.where(((_p582) & (v['_go']==0)), 890, v['_go'])
                # WRF:5478 ufnclu(i,kclds+1) = alfa(i,kclds+1)*tdcl1(i,kclds+1)
                v['ufnclu'] = v['ufnclu'].at[(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1].set(jnp.where((((_p578 & (v['_go']==0))) & (v['_go']==0)), _r((v['alfa'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1] * v['tdcl1'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1])), v['ufnclu'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1]))
                # WRF:5479 dfnclu(i,kclds+1) = tdcl1(i,kclds+1)
                v['dfnclu'] = v['dfnclu'].at[(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1].set(jnp.where((((_p578 & (v['_go']==0))) & (v['_go']==0)), _r(v['tdcl1'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1]), v['dfnclu'][(v['i'])-1, ((v['kclds'] + jnp.int32(1)))-1]))
                v['_go'] = jnp.where(v['_go']==890, 0, v['_go'])
                # WRF:5480 continue
                return v
            v = lax.fori_loop(0,_p579,_p580,v)
            v['i'] = jnp.where((_p578 & (v['_go']==0)), jnp.int32((v['myis'])+_p579*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5481 do i=myis,myie
            _p583 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p584 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p585(_p586,v):
                v = dict(v)
                v['i'] = jnp.where((_p583 & (v['_go']==0)), jnp.int32((v['myis'])+_p586*(jnp.int32(1))), v['i'])
                # WRF:5482 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p583 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5483 if(kclds.eq.0) go to 901
                _p587 = (((_p583 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5483 go to 901
                v['_go'] = jnp.where(((_p587) & (v['_go']==0)), 901, v['_go'])
                # WRF:5484 do kk=kclds,1,-1
                _p588 = (((_p583 & (v['_go']==0))) & (v['_go']==0))
                _p589 = jnp.maximum(jnp.int32(0), ((jnp.int32(1))-(v['kclds']))//(jnp.int32(-1))+jnp.int32(1))
                def _p590(_p591,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p588 & (v['_go']==0)), jnp.int32((v['kclds'])+_p591*(jnp.int32(-1))), v['kk'])
                    # WRF:5488 denom=alfa(i,kk+1)*tclu(i,kk)
                    v['denom'] = jnp.where((((_p588 & (v['_go']==0))) & (v['_go']==0)), _r((v['alfa'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] * v['tclu'][(v['i'])-1, (v['kk'])-1])), v['denom'])
                    # WRF:5489 __if__ denom.gt.rthresh
                    _p592 = (((_p588 & (v['_go']==0))) & (v['_go']==0)) & ((v['denom'] > _r(_r(1.e-15))))
                    _p593 = (((_p588 & (v['_go']==0))) & (v['_go']==0)) & ~((v['denom'] > _r(_r(1.e-15))))
                    # WRF:5490 ufnclu(i,kk)=ufnclu(i,kk+1)*alfau(i,kk+1)/denom
                    v['ufnclu'] = v['ufnclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where(((_p592) & (v['_go']==0)), _r(((v['ufnclu'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] * v['alfau'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]) / v['denom'])), v['ufnclu'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5492 ufnclu(i,kk)=0.
                    v['ufnclu'] = v['ufnclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where(((_p593) & (v['_go']==0)), _r(_r(0.)), v['ufnclu'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5494 __if__ alfa(i,kk).gt.rthresh
                    _p594 = (((_p588 & (v['_go']==0))) & (v['_go']==0)) & ((v['alfa'][(v['i'])-1, (v['kk'])-1] > _r(_r(1.e-15))))
                    _p595 = (((_p588 & (v['_go']==0))) & (v['_go']==0)) & ~((v['alfa'][(v['i'])-1, (v['kk'])-1] > _r(_r(1.e-15))))
                    # WRF:5495 dfnclu(i,kk)=ufnclu(i,kk)/alfa(i,kk)
                    v['dfnclu'] = v['dfnclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where(((_p594) & (v['_go']==0)), _r((v['ufnclu'][(v['i'])-1, (v['kk'])-1] / v['alfa'][(v['i'])-1, (v['kk'])-1])), v['dfnclu'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5497 dfnclu(i,kk)=0.
                    v['dfnclu'] = v['dfnclu'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where(((_p595) & (v['_go']==0)), _r(_r(0.)), v['dfnclu'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==900, 0, v['_go'])
                    # WRF:5499 continue
                    return v
                v = lax.fori_loop(0,_p589,_p590,v)
                v['kk'] = jnp.where((_p588 & (v['_go']==0)), jnp.int32((v['kclds'])+_p589*(jnp.int32(-1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==901, 0, v['_go'])
                # WRF:5500 continue
                return v
            v = lax.fori_loop(0,_p584,_p585,v)
            v['i'] = jnp.where((_p583 & (v['_go']==0)), jnp.int32((v['myis'])+_p584*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5502 do i=myis,myie
            _p596 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p597 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p598(_p599,v):
                v = dict(v)
                v['i'] = jnp.where((_p596 & (v['_go']==0)), jnp.int32((v['myis'])+_p599*(jnp.int32(1))), v['i'])
                # WRF:5503 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p596 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5504 if(kclds.eq.0) go to 911
                _p600 = (((_p596 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5504 go to 911
                v['_go'] = jnp.where(((_p600) & (v['_go']==0)), 911, v['_go'])
                # WRF:5505 do kk=1,kclds+1
                _p601 = (((_p596 & (v['_go']==0))) & (v['_go']==0))
                _p602 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
                def _p603(_p604,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p601 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p604*(jnp.int32(1))), v['kk'])
                    # WRF:5506 ufntrn(i,kk) = ufnclu(i,kk)*ufntrn(i,kk)
                    v['ufntrn'] = v['ufntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p601 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufnclu'][(v['i'])-1, (v['kk'])-1] * v['ufntrn'][(v['i'])-1, (v['kk'])-1])), v['ufntrn'][(v['i'])-1, (v['kk'])-1]))
                    # WRF:5507 dfntrn(i,kk) = dfnclu(i,kk)*dfntrn(i,kk)
                    v['dfntrn'] = v['dfntrn'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p601 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfnclu'][(v['i'])-1, (v['kk'])-1] * v['dfntrn'][(v['i'])-1, (v['kk'])-1])), v['dfntrn'][(v['i'])-1, (v['kk'])-1]))
                    v['_go'] = jnp.where(v['_go']==910, 0, v['_go'])
                    # WRF:5508 continue
                    return v
                v = lax.fori_loop(0,_p602,_p603,v)
                v['kk'] = jnp.where((_p601 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p602*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==911, 0, v['_go'])
                # WRF:5509 continue
                return v
            v = lax.fori_loop(0,_p597,_p598,v)
            v['i'] = jnp.where((_p596 & (v['_go']==0)), jnp.int32((v['myis'])+_p597*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5510 do i=myis,myie
            _p605 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p606 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p607(_p608,v):
                v = dict(v)
                v['i'] = jnp.where((_p605 & (v['_go']==0)), jnp.int32((v['myis'])+_p608*(jnp.int32(1))), v['i'])
                # WRF:5511 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p605 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5512 if(kclds.eq.0) go to 930
                _p609 = (((_p605 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5512 go to 930
                v['_go'] = jnp.where(((_p609) & (v['_go']==0)), 930, v['_go'])
                # WRF:5513 j2=kbtmsw(i,2)
                v['j2'] = jnp.where((((_p605 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtmsw'][(v['i'])-1, (jnp.int32(2))-1]), v['j2'])
                # WRF:5514 do k=j2,lp1
                _p610 = (((_p605 & (v['_go']==0))) & (v['_go']==0))
                _p611 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(v['j2']))//(jnp.int32(1))+jnp.int32(1))
                def _p612(_p613,v):
                    v = dict(v)
                    v['k'] = jnp.where((_p610 & (v['_go']==0)), jnp.int32((v['j2'])+_p613*(jnp.int32(1))), v['k'])
                    # WRF:5515 ufn(i,k) = ufntrn(i,1)*ttu(i,k)
                    v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p610 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufntrn'][(v['i'])-1, (jnp.int32(1))-1] * v['ttu'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                    # WRF:5516 dfn(i,k) = dfntrn(i,1)*ttd(i,k)
                    v['dfn'] = v['dfn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p610 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfntrn'][(v['i'])-1, (jnp.int32(1))-1] * v['ttd'][(v['i'])-1, (v['k'])-1])), v['dfn'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==920, 0, v['_go'])
                    # WRF:5517 continue
                    return v
                v = lax.fori_loop(0,_p611,_p612,v)
                v['k'] = jnp.where((_p610 & (v['_go']==0)), jnp.int32((v['j2'])+_p611*(jnp.int32(1))), v['k'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==930, 0, v['_go'])
                # WRF:5518 continue
                return v
            v = lax.fori_loop(0,_p606,_p607,v)
            v['i'] = jnp.where((_p605 & (v['_go']==0)), jnp.int32((v['myis'])+_p606*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5519 do i=myis,myie
            _p614 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p615 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p616(_p617,v):
                v = dict(v)
                v['i'] = jnp.where((_p614 & (v['_go']==0)), jnp.int32((v['myis'])+_p617*(jnp.int32(1))), v['i'])
                # WRF:5520 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p614 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5521 if(kclds.eq.0) go to 970
                _p618 = (((_p614 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5521 go to 970
                v['_go'] = jnp.where(((_p618) & (v['_go']==0)), 970, v['_go'])
                # WRF:5522 do kk=2,kclds+1
                _p619 = (((_p614 & (v['_go']==0))) & (v['_go']==0))
                _p620 = jnp.maximum(jnp.int32(0), (((v['kclds'] + jnp.int32(1)))-(jnp.int32(2)))//(jnp.int32(1))+jnp.int32(1))
                def _p621(_p622,v):
                    v = dict(v)
                    v['kk'] = jnp.where((_p619 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p622*(jnp.int32(1))), v['kk'])
                    # WRF:5523 j1 = ktopsw(i,kk)
                    v['j1'] = jnp.where((((_p619 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['ktopsw'][(v['i'])-1, (v['kk'])-1]), v['j1'])
                    # WRF:5524 j2 = kbtmsw(i,kk+1)
                    v['j2'] = jnp.where((((_p619 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtmsw'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1]), v['j2'])
                    # WRF:5525 if (j1.eq.1) go to 965
                    _p623 = (((_p619 & (v['_go']==0))) & (v['_go']==0)) & ((v['j1'] == jnp.int32(1)))
                    # WRF:5525 go to 965
                    v['_go'] = jnp.where(((_p623) & (v['_go']==0)), 965, v['_go'])
                    # WRF:5526 do k=j2,j1
                    _p624 = (((_p619 & (v['_go']==0))) & (v['_go']==0))
                    _p625 = jnp.maximum(jnp.int32(0), ((v['j1'])-(v['j2']))//(jnp.int32(1))+jnp.int32(1))
                    def _p626(_p627,v):
                        v = dict(v)
                        v['k'] = jnp.where((_p624 & (v['_go']==0)), jnp.int32((v['j2'])+_p627*(jnp.int32(1))), v['k'])
                        # WRF:5527 ufn(i,k) = ufntrn(i,kk)*ttu(i,k)
                        v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p624 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufntrn'][(v['i'])-1, (v['kk'])-1] * v['ttu'][(v['i'])-1, (v['k'])-1])), v['ufn'][(v['i'])-1, (v['k'])-1]))
                        # WRF:5528 dfn(i,k) = dfntrn(i,kk)*ttd(i,k)
                        v['dfn'] = v['dfn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p624 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfntrn'][(v['i'])-1, (v['kk'])-1] * v['ttd'][(v['i'])-1, (v['k'])-1])), v['dfn'][(v['i'])-1, (v['k'])-1]))
                        v['_go'] = jnp.where(v['_go']==940, 0, v['_go'])
                        # WRF:5529 continue
                        return v
                    v = lax.fori_loop(0,_p625,_p626,v)
                    v['k'] = jnp.where((_p624 & (v['_go']==0)), jnp.int32((v['j2'])+_p625*(jnp.int32(1))), v['k'])
                    v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                    # WRF:5530 j3 = kbtmsw(i,kk)
                    v['j3'] = jnp.where((((_p619 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['kbtmsw'][(v['i'])-1, (v['kk'])-1]), v['j3'])
                    # WRF:5531 __if__ (j3-j1).gt.1
                    _p628 = (((_p619 & (v['_go']==0))) & (v['_go']==0)) & (((v['j3'] - v['j1']) > jnp.int32(1)))
                    _p629 = (((_p619 & (v['_go']==0))) & (v['_go']==0)) & ~(((v['j3'] - v['j1']) > jnp.int32(1)))
                    # WRF:5532 tempf = (ufnclu(i,kk)-ufn(i,j3))*dpcld(i,kk-1)
                    v['tempf'] = jnp.where(((_p628) & (v['_go']==0)), _r(((v['ufnclu'][(v['i'])-1, (v['kk'])-1] - v['ufn'][(v['i'])-1, (v['j3'])-1]) * v['dpcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])), v['tempf'])
                    # WRF:5533 tempg = (dfnclu(i,kk)-dfn(i,j3))*dpcld(i,kk-1)
                    v['tempg'] = jnp.where(((_p628) & (v['_go']==0)), _r(((v['dfnclu'][(v['i'])-1, (v['kk'])-1] - v['dfn'][(v['i'])-1, (v['j3'])-1]) * v['dpcld'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])), v['tempg'])
                    # WRF:5534 do k=j1+1,j3-1
                    _p630 = ((_p628) & (v['_go']==0))
                    _p631 = jnp.maximum(jnp.int32(0), (((v['j3'] - jnp.int32(1)))-((v['j1'] + jnp.int32(1))))//(jnp.int32(1))+jnp.int32(1))
                    def _p632(_p633,v):
                        v = dict(v)
                        v['k'] = jnp.where((_p630 & (v['_go']==0)), jnp.int32(((v['j1'] + jnp.int32(1)))+_p633*(jnp.int32(1))), v['k'])
                        # WRF:5535 ufn(i,k) = ufnclu(i,kk)+tempf*(pp(i,k)-pptop(i,kk-1))
                        v['ufn'] = v['ufn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p630 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufnclu'][(v['i'])-1, (v['kk'])-1] + (v['tempf'] * (v['pp'][(v['i'])-1, (v['k'])-1] - v['pptop'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])))), v['ufn'][(v['i'])-1, (v['k'])-1]))
                        # WRF:5536 dfn(i,k) = dfnclu(i,kk)+tempg*(pp(i,k)-pptop(i,kk-1))
                        v['dfn'] = v['dfn'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p630 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfnclu'][(v['i'])-1, (v['kk'])-1] + (v['tempg'] * (v['pp'][(v['i'])-1, (v['k'])-1] - v['pptop'][(v['i'])-1, ((v['kk'] - jnp.int32(1)))-1])))), v['dfn'][(v['i'])-1, (v['k'])-1]))
                        v['_go'] = jnp.where(v['_go']==950, 0, v['_go'])
                        # WRF:5537 continue
                        return v
                    v = lax.fori_loop(0,_p631,_p632,v)
                    v['k'] = jnp.where((_p630 & (v['_go']==0)), jnp.int32(((v['j1'] + jnp.int32(1)))+_p631*(jnp.int32(1))), v['k'])
                    v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                    v['_go'] = jnp.where(v['_go']==965, 0, v['_go'])
                    # WRF:5539 continue
                    return v
                v = lax.fori_loop(0,_p620,_p621,v)
                v['kk'] = jnp.where((_p619 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p620*(jnp.int32(1))), v['kk'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==970, 0, v['_go'])
                # WRF:5540 continue
                return v
            v = lax.fori_loop(0,_p615,_p616,v)
            v['i'] = jnp.where((_p614 & (v['_go']==0)), jnp.int32((v['myis'])+_p615*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5541 do i=myis,myie
            _p634 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p635 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p636(_p637,v):
                v = dict(v)
                v['i'] = jnp.where((_p634 & (v['_go']==0)), jnp.int32((v['myis'])+_p637*(jnp.int32(1))), v['i'])
                # WRF:5542 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p634 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5543 if(kclds.eq.0) go to 980
                _p638 = (((_p634 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5543 go to 980
                v['_go'] = jnp.where(((_p638) & (v['_go']==0)), 980, v['_go'])
                # WRF:5544 do k=1,lp1
                _p639 = (((_p634 & (v['_go']==0))) & (v['_go']==0))
                _p640 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
                def _p641(_p642,v):
                    v = dict(v)
                    v['k'] = jnp.where((_p639 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p642*(jnp.int32(1))), v['k'])
                    # WRF:5545 dfswc(i,k) = dfswc(i,k) + dfn(i,k)*dfntop(i,n)
                    v['dfswc'] = v['dfswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p639 & (v['_go']==0))) & (v['_go']==0)), _r((v['dfswc'][(v['i'])-1, (v['k'])-1] + (v['dfn'][(v['i'])-1, (v['k'])-1] * v['dfntop'][(v['i'])-1, (v['n'])-1]))), v['dfswc'][(v['i'])-1, (v['k'])-1]))
                    # WRF:5546 ufswc(i,k) = ufswc(i,k) + ufn(i,k)*dfntop(i,n)
                    v['ufswc'] = v['ufswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p639 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufswc'][(v['i'])-1, (v['k'])-1] + (v['ufn'][(v['i'])-1, (v['k'])-1] * v['dfntop'][(v['i'])-1, (v['n'])-1]))), v['ufswc'][(v['i'])-1, (v['k'])-1]))
                    v['_go'] = jnp.where(v['_go']==981, 0, v['_go'])
                    # WRF:5547 continue
                    return v
                v = lax.fori_loop(0,_p640,_p641,v)
                v['k'] = jnp.where((_p639 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p640*(jnp.int32(1))), v['k'])
                v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
                v['_go'] = jnp.where(v['_go']==980, 0, v['_go'])
                # WRF:5548 continue
                return v
            v = lax.fori_loop(0,_p635,_p636,v)
            v['i'] = jnp.where((_p634 & (v['_go']==0)), jnp.int32((v['myis'])+_p635*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            # WRF:5549 do i=myis,myie
            _p643 = (((_p486 & (v['_go']==0))) & (v['_go']==0))
            _p644 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
            def _p645(_p646,v):
                v = dict(v)
                v['i'] = jnp.where((_p643 & (v['_go']==0)), jnp.int32((v['myis'])+_p646*(jnp.int32(1))), v['i'])
                # WRF:5550 kclds=nclds(i)
                v['kclds'] = jnp.where((((_p643 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
                # WRF:5551 if(kclds.eq.0) go to 990
                _p647 = (((_p643 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
                # WRF:5551 go to 990
                v['_go'] = jnp.where(((_p647) & (v['_go']==0)), 990, v['_go'])
                # WRF:5552 gdfnd(i) = gdfnd(i) + ccmax(i)*dfn(i,lp1)*dfntop(i,n)
                v['gdfnd'] = v['gdfnd'].at[(v['i'])-1].set(jnp.where((((_p643 & (v['_go']==0))) & (v['_go']==0)), _r((v['gdfnd'][(v['i'])-1] + ((v['ccmax'][(v['i'])-1] * v['dfn'][(v['i'])-1, (v['lp1'])-1]) * v['dfntop'][(v['i'])-1, (v['n'])-1]))), v['gdfnd'][(v['i'])-1]))
                v['_go'] = jnp.where(v['_go']==990, 0, v['_go'])
                # WRF:5553 continue
                return v
            v = lax.fori_loop(0,_p644,_p645,v)
            v['i'] = jnp.where((_p643 & (v['_go']==0)), jnp.int32((v['myis'])+_p644*(jnp.int32(1))), v['i'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1000, 0, v['_go'])
            # WRF:5554 continue
            return v
        v = lax.fori_loop(0,_p487,_p488,v)
        v['n'] = jnp.where((_p486 & (v['_go']==0)), jnp.int32((jnp.int32(2))+_p487*(jnp.int32(1))), v['n'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5555 do i=myis,myie
        _p648 = ((jnp.bool_(True)) & (v['_go']==0))
        _p649 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p650(_p651,v):
            v = dict(v)
            v['i'] = jnp.where((_p648 & (v['_go']==0)), jnp.int32((v['myis'])+_p651*(jnp.int32(1))), v['i'])
            # WRF:5556 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p648 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5557 if(kclds.eq.0) go to 1100
            _p652 = (((_p648 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5557 go to 1100
            v['_go'] = jnp.where(((_p652) & (v['_go']==0)), 1100, v['_go'])
            # WRF:5558 do k=1,lp1
            _p653 = (((_p648 & (v['_go']==0))) & (v['_go']==0))
            _p654 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p655(_p656,v):
                v = dict(v)
                v['k'] = jnp.where((_p653 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p656*(jnp.int32(1))), v['k'])
                # WRF:5559 dfswc(i,k) = tmp1(i)*dfswl(i,k) + ccmax(i)*dfswc(i,k)
                v['dfswc'] = v['dfswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p653 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tmp1'][(v['i'])-1] * v['dfswl'][(v['i'])-1, (v['k'])-1]) + (v['ccmax'][(v['i'])-1] * v['dfswc'][(v['i'])-1, (v['k'])-1]))), v['dfswc'][(v['i'])-1, (v['k'])-1]))
                # WRF:5560 ufswc(i,k) = tmp1(i)*ufswl(i,k) + ccmax(i)*ufswc(i,k)
                v['ufswc'] = v['ufswc'].at[(v['i'])-1, (v['k'])-1].set(jnp.where((((_p653 & (v['_go']==0))) & (v['_go']==0)), _r(((v['tmp1'][(v['i'])-1] * v['ufswl'][(v['i'])-1, (v['k'])-1]) + (v['ccmax'][(v['i'])-1] * v['ufswc'][(v['i'])-1, (v['k'])-1]))), v['ufswc'][(v['i'])-1, (v['k'])-1]))
                v['_go'] = jnp.where(v['_go']==1101, 0, v['_go'])
                # WRF:5561 continue
                return v
            v = lax.fori_loop(0,_p654,_p655,v)
            v['k'] = jnp.where((_p653 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p654*(jnp.int32(1))), v['k'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1100, 0, v['_go'])
            # WRF:5562 continue
            return v
        v = lax.fori_loop(0,_p649,_p650,v)
        v['i'] = jnp.where((_p648 & (v['_go']==0)), jnp.int32((v['myis'])+_p649*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5563 do i=myis,myie
        _p657 = ((jnp.bool_(True)) & (v['_go']==0))
        _p658 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p659(_p660,v):
            v = dict(v)
            v['i'] = jnp.where((_p657 & (v['_go']==0)), jnp.int32((v['myis'])+_p660*(jnp.int32(1))), v['i'])
            # WRF:5564 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p657 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5565 if(kclds.eq.0) go to 1200
            _p661 = (((_p657 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5565 go to 1200
            v['_go'] = jnp.where(((_p661) & (v['_go']==0)), 1200, v['_go'])
            # WRF:5566 do kk=1,lp1
            _p662 = (((_p657 & (v['_go']==0))) & (v['_go']==0))
            _p663 = jnp.maximum(jnp.int32(0), ((v['lp1'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p664(_p665,v):
                v = dict(v)
                v['kk'] = jnp.where((_p662 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p665*(jnp.int32(1))), v['kk'])
                # WRF:5567 fswc(i,kk) = ufswc(i,kk)-dfswc(i,kk)
                v['fswc'] = v['fswc'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p662 & (v['_go']==0))) & (v['_go']==0)), _r((v['ufswc'][(v['i'])-1, (v['kk'])-1] - v['dfswc'][(v['i'])-1, (v['kk'])-1])), v['fswc'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==1201, 0, v['_go'])
                # WRF:5568 continue
                return v
            v = lax.fori_loop(0,_p663,_p664,v)
            v['kk'] = jnp.where((_p662 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p663*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1200, 0, v['_go'])
            # WRF:5569 continue
            return v
        v = lax.fori_loop(0,_p658,_p659,v)
        v['i'] = jnp.where((_p657 & (v['_go']==0)), jnp.int32((v['myis'])+_p658*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        # WRF:5570 do i=myis,myie
        _p666 = ((jnp.bool_(True)) & (v['_go']==0))
        _p667 = jnp.maximum(jnp.int32(0), ((v['myie'])-(v['myis']))//(jnp.int32(1))+jnp.int32(1))
        def _p668(_p669,v):
            v = dict(v)
            v['i'] = jnp.where((_p666 & (v['_go']==0)), jnp.int32((v['myis'])+_p669*(jnp.int32(1))), v['i'])
            # WRF:5571 kclds=nclds(i)
            v['kclds'] = jnp.where((((_p666 & (v['_go']==0))) & (v['_go']==0)), jnp.int32(v['nclds'][(v['i'])-1]), v['kclds'])
            # WRF:5572 if(kclds.eq.0) go to 1250
            _p670 = (((_p666 & (v['_go']==0))) & (v['_go']==0)) & ((v['kclds'] == jnp.int32(0)))
            # WRF:5572 go to 1250
            v['_go'] = jnp.where(((_p670) & (v['_go']==0)), 1250, v['_go'])
            # WRF:5573 do kk=1, l
            _p671 = (((_p666 & (v['_go']==0))) & (v['_go']==0))
            _p672 = jnp.maximum(jnp.int32(0), ((v['l'])-(jnp.int32(1)))//(jnp.int32(1))+jnp.int32(1))
            def _p673(_p674,v):
                v = dict(v)
                v['kk'] = jnp.where((_p671 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p674*(jnp.int32(1))), v['kk'])
                # WRF:5574 hswc(i,kk) = radcon*(fswc(i,kk+1)-fswc(i,kk))/dp(i,kk)
                v['hswc'] = v['hswc'].at[(v['i'])-1, (v['kk'])-1].set(jnp.where((((_p671 & (v['_go']==0))) & (v['_go']==0)), _r(((v['radcon'] * (v['fswc'][(v['i'])-1, ((v['kk'] + jnp.int32(1)))-1] - v['fswc'][(v['i'])-1, (v['kk'])-1])) / v['dp'][(v['i'])-1, (v['kk'])-1])), v['hswc'][(v['i'])-1, (v['kk'])-1]))
                v['_go'] = jnp.where(v['_go']==1251, 0, v['_go'])
                # WRF:5575 continue
                return v
            v = lax.fori_loop(0,_p672,_p673,v)
            v['kk'] = jnp.where((_p671 & (v['_go']==0)), jnp.int32((jnp.int32(1))+_p672*(jnp.int32(1))), v['kk'])
            v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
            v['_go'] = jnp.where(v['_go']==1250, 0, v['_go'])
            # WRF:5576 continue
            return v
        v = lax.fori_loop(0,_p667,_p668,v)
        v['i'] = jnp.where((_p666 & (v['_go']==0)), jnp.int32((v['myis'])+_p667*(jnp.int32(1))), v['i'])
        v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])
        return (v['fswc'], v['hswc'], v['ufswc'], v['dfswc'], v['fswl'], v['hswl'], v['ufswl'], v['dfswl'], v['press'], v['coszro'], v['taudar'], v['rh2o'], v['rrco2'], v['ssolar'], v['qo3'], v['nclds'], v['ktopsw'], v['kbtmsw'], v['camt'], v['crr'], v['ctt'], v['alvb'], v['alnb'], v['alvd'], v['alnd'], v['gdfvb'], v['gdfnb'], v['gdfvd'], v['gdfnd'], v['abcff'], v['pwts'], v['h35e1'], v['h1224e3'], v['one'], v['zero'], v['haf'], v['h69766e5'], v['hp219'], v['hp816'], v['rrayav'], v['ginv'], v['cfco2'], v['cfo3'], v['two'], v['h235m3'], v['hp26'], v['h129m2'], v['h75826m4'], v['h1036e2'], v['h1p082'], v['hmp805'], v['h1386e2'], v['h658m2'], v['h2118m2'], v['h42m2'], v['h323m4'], v['hm1ez'], v['diffctr'], v['o3difctr'], v['fifty'], v['radcon'], v['ids'], v['ide'], v['jds'], v['jde'], v['kds'], v['kde'], v['ims'], v['ime'], v['jms'], v['jme'], v['kms'], v['kme'], v['its'], v['ite'], v['jts'], v['jte'], v['kts'], v['kte'],)
    return _swr93
