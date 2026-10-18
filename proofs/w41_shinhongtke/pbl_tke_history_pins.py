"""Per-scheme TKE_PBL history pins vs original CPU-WRF (lane w41-shinhongtke), no JAX.

The v0.4.0 v-prep GPU runs wrote the post-RK held State.qke at every output time as the superset
variable QKE (wrfout_writer QKE <- state.qke, float32; lead 0 is the wrfinput overlay). The new
writer emits TKE_PBL[:-1] = factor(bl) * qke with a zero Z top, so factor * GPU QKE (leads >= 1) is
byte-for-byte what the fixed writer writes for those states. This compares it with the original
CPU-WRF TKE_PBL of the same composite per lead (chaotic-trajectory statistics, not bytes), and
records the WRF lead-0 cold seed against the port's initial qke (from source; see README).
usage: python pbl_tke_history_pins.py <out.json>
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import netCDF4

VPREP = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
CASES = {'c03': (2, 'MYJ', .5), 'c26': (4, 'QNSE', .5), 'c06': (8, 'BouLac', 1.), 'c57': (8, 'BouLac', 1.),
         'c08': (12, 'GBM', 1.), 'c07': (11, 'Shin-Hong', 1.)}
# Port initial qke (cold start) per scheme, from source: QNSE operational_mode initial carry .02;
# GBM d02_replay GBMPBLINIT .001; Shin-Hong _initial_carry_for_run .005; MYJ/BouLac: MYNN cold-start
# equilibrium profile (d02_replay._wrf_mynn_coldstart_qke, no scheme-own seed) -> not a constant.
PORT_SEED = {'c03': None, 'c26': .02, 'c06': None, 'c57': None, 'c08': .001, 'c07': .005}
out = {}
for case, (bl, name, factor) in CASES.items():
    cpu_files = sorted((VPREP / case / 'cpu').glob('wrfout_d01_*'))
    rows = []
    for k, cf in enumerate(cpu_files):
        gf = sorted(glob.glob(str(VPREP / case / 'gpu' / '**' / cf.name), recursive=True))
        if not gf:
            continue
        with netCDF4.Dataset(cf) as c, netCDF4.Dataset(gf[0]) as g:
            t = np.asarray(c['TKE_PBL'][0], np.float32)
            q = np.asarray(g['QKE'][0], np.float32) if 'QKE' in g.variables else None
        row = dict(lead=k, cpu_top_zero=bool(not t[-1].any()), cpu_mass_unique=[float(x) for x in np.unique(t[:-1])[:3]])
        if k == 0:
            seed = PORT_SEED[case]
            row['wrf_lead0_constant'] = float(t[:-1].flat[0]) if np.unique(t[:-1]).size == 1 else None
            row['port_initial_tke'] = None if seed is None else float(np.float32(factor) * np.float32(seed))
            row['lead0_byte_match'] = (None if seed is None else
                                       bool(np.full_like(t[:-1], np.float32(factor) * np.float32(seed)).tobytes() == t[:-1].tobytes()))
        elif q is not None:
            m = (np.float32(factor) * q).astype(np.float64); a = t[:-1].astype(np.float64)
            d = m - a
            row.update(cpu_mean=float(a.mean()), mapped_mean=float(m.mean()), ratio=float(m.mean() / a.mean()),
                       bias=float(d.mean()), rmse=float(np.sqrt((d * d).mean())), cpu_max=float(a.max()),
                       pearson_r=(float(np.corrcoef(a.ravel(), m.ravel())[0, 1]) if a.std() > 0 and m.std() > 0 else None),
                       byte_equal=bool(m.astype(np.float32).tobytes() == t[:-1].tobytes()))
        rows.append(row)
    later = [r for r in rows if r['lead'] > 0 and 'pearson_r' in r]
    out[case] = dict(bl=bl, scheme=name, factor=factor, rows=rows,
                     min_r=min((r['pearson_r'] for r in later if r['pearson_r'] is not None), default=None),
                     constant_byte_leads=sum(r['pearson_r'] is None and r['byte_equal'] for r in later), leads=len(later), ratio_range=[min(r['ratio'] for r in later), max(r['ratio'] for r in later)],
                     cpu_tops_zero=all(r['cpu_top_zero'] for r in rows), lead0=rows[0])
out['reading'] = ('[M] leads>=1: factor*GPU QKE (= new writer TKE_PBL bytes) vs CPU-WRF TKE_PBL, chaotic GPU/CPU '
                  'trajectories; lead 0: WRF cold seed vs port initial qke (source). CAM-UW (bl=9) holds TKE in '
                  'its carry (no GPU file evidence) -> unit/producer tests only.')
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')
for case, v in out.items():
    if case == 'reading':
        continue
    print(case, v['scheme'], 'factor', v['factor'], 'min_r', v['min_r'], 'const_byte', v['constant_byte_leads'], '/', v['leads'], 'ratio %.3f..%.3f' % tuple(v['ratio_range']),
          'tops0', v['cpu_tops_zero'], 'lead0 WRF', v['lead0'].get('wrf_lead0_constant'), 'port', v['lead0'].get('port_initial_tke'),
          'byte', v['lead0'].get('lead0_byte_match'))
