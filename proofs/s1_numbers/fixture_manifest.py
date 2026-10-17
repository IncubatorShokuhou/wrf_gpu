"""Attach provenance/units/tolerance metadata to the tracked 76 KiB input crop."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT/'tests/v040/numbers'
previous = json.loads((P/'swiss_transport.json').read_text())
units = {'U': 'm s-1', 'V': 'm s-1', 'QVAPOR': 'kg kg-1', 'MU': 'Pa', 'MUB': 'Pa',
         'C2H': 'Pa', 'dx': 'm', 'dy': 'm'}
with np.load(P/'swiss_transport.npz') as arrays:
    variables = {name: {'units': units.get(name, '1'), 'shape': list(value.shape), 'dtype': str(value.dtype),
                        'absolute_tolerance': 0, 'relative_tolerance': 0}
                 for name, value in arrays.items()}
manifest = {
    'fixture_id': 's1_numbers_swiss_crop_v1',
    'source': previous['source'], 'source_sha256': previous['source_sha256'],
    'source_commit': 'WRF f52c197ed39d12e087d02c50f412d90d418f6186; exact routine hashes in oracle_manifests',
    'scenario': 'Swiss RD11 cropped complete operator domain, specified nonperiodic h5/v3, dt18, RK3; not a forecast',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'crop_yx': [12, 12], 'mass_shape': [44, 12, 12], 'variables': variables,
    'files': {'swiss_transport.npz': {'sha256': hashlib.sha256((P/'swiss_transport.npz').read_bytes()).hexdigest()}},
    'generation_command': 'JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8 nice -n 19 python proofs/s1_numbers/make_fixture.py; '
                          'python proofs/s1_numbers/fixture_manifest.py',
    'license_notes': 'Pristine WRF source remains external under its original license. Small input subset tracked per S1 asset rule.',
    'staggering': 'U: z,y,x-face; V: z,y-face,x; QVAPOR/mass/maps: mass grid; U/V map factors on their faces; eta vectors mass levels',
    'constructed_operands': {
        'number_units': 'kg-1 (qh: kg kg-1)',
        'number_shape': [44, 12, 12], 'dtype': 'float32',
        'profiles': 'scale*(QVAPOR/max(QVAPOR))**(1+i%3)*(1+.3*sin(x/2)); exact order/scales in test_number_transport.py',
        'scales': {'Ns': 2e5, 'Ng': 4e3, 'Nc': 1.5e8, 'Nr': 3e4, 'Nn': 2e8, 'qh': 3e-4},
        'vertical_flux': 'ROM=100*sin(linspace(0,pi,45)), endpoints exactly zero; shape45x12x12; unitsPa s-1',
        'ccn_conc': '0/1e8/2.7e8, Registry units m-3; assigned literally to QNN inflow as pristine WRF does',
    },
    'comparison': {
        'reference': 'unchanged pristine advect_scalar/advect_scalar_pd/rk_update_scalar/flow_dep_bdy[_qnn], isolated REAL4 lane build',
        'boundary': 'byte exact, nine width/inflow cases, mixed signs and exact zero velocities',
        'scalar_chain': 'max absolute error <=32*spacing(float32(max(abs(reference)))), relative tolerance0; both inputs finite',
        'tolerance_frozen_at': '16f97cf96 before NS02 results; independent noncontracting gfortran vs XLA CPU REAL4 composition',
    },
}
(P/'swiss_transport.json').write_text(json.dumps(manifest, indent=2)+'\n')
(P/'swiss_transport.yaml').write_text(''.join(key+': '+json.dumps(value)+'\n' for key, value in manifest.items()))
