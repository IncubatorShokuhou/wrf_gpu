import sys, json, numpy as np
z = np.load(sys.argv[1])
rv = 461.6 / 287.0
I = (slice(None), slice(5, -5), slice(5, -5))
def rel(a, b, sl=I):
    e = (a - b)[sl]; return float(np.sqrt(np.mean(e**2)) / np.sqrt(np.mean(b[sl]**2)))
P, W = z['port_rthften'], z['wrf_RTHFTEN']
Q, WQ = z['port_rqvften'], z['wrf_RQVFTEN']
print('shapes', P.shape, W.shape, z['msfty'].shape)
print('RTHFTEN rel', rel(P, W), 'RQVFTEN rel', rel(Q, WQ))
# state at end of step 1 (port theta = theta_m; WRF T is dry perturbation with use_theta_m? compare both)
thm = z['port_theta']; qv = z['port_qv']; thd = thm / (1 + rv * qv)
wT = z['wrf_T'] + 300.0
print('state: dry-theta rms diff', float(np.sqrt(np.mean((thd - wT)[I]**2))), 'moist-theta rms diff', float(np.sqrt(np.mean((thm - wT)[I]**2))))
print('qv rel', rel(qv, z['wrf_QVAPOR']), 'mu rms diff', float(np.sqrt(np.mean((z['port_mu'] - (z['wrf_MU'] + z['wrf_MUB']))[5:-5, 5:-5]**2))))
print('w rms diff', float(np.sqrt(np.mean((z['port_w'] - z['wrf_W'])[I]**2))), 'w rms', float(np.sqrt(np.mean(z['wrf_W'][I]**2))))
# map-factor hypotheses: port * msfty^p vs WRF
m = z['msfty'][None]
for p in (-2, -1, 0, 1, 2):
    print(f'RTHFTEN rel with port*msfty^{p}', rel(P * m**p, W), ' RQVFTEN', rel(Q * m**p, WQ))
print('msfty range', float(z['msfty'].min()), float(z['msfty'].max()))
# least-squares scale and per-level error
a = float(np.sum((P * W)[I]) / np.sum((P * P)[I])); print('LS scale WRF ~ a*port: a =', a, ' rel after scale', rel(a * P, W))
lev = [rel(P[k:k+1], W[k:k+1], (slice(None), slice(5, -5), slice(5, -5))) for k in range(P.shape[0])]
print('per-level rel (k: rel)', [(k, round(v, 3)) for k, v in enumerate(lev)][:45])
# error vs convection / PBL
cu = np.abs(z['wrf_RTHCUTEN'])[I].max(axis=0) > 0; e2 = ((P - W)[I]**2).mean(axis=0)
print('convective cols', int(cu.sum()), 'of', cu.size, ' err share in convective cols', float(e2[cu].sum() / e2.sum()))
# dry conversion check with WRF state: WRF RTHFTEN = th/(T) * (C - rv*th*RQV); invert using port state
print('WRF RTHFTEN vs port no-dry form rel', rel(P * thm / thd + rv * thd * Q, W))
