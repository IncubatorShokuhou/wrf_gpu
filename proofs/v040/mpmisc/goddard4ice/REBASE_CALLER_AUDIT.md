# Goddard4ice integration source audit after main3716340b rebase

Audit baseline: root d152cc02c (106 selected shared tests and default2bf identity
already measured by root; this audit executed no JAX/numerical science).
No own core/proof/test blobs differ from the pre-rebase backup. All previously
completed original-WRF numerical qualification remains reusable.

Confirmed caller mapping: dry TH, total pressure/exner, rho=(1+qv)/ALT, direct
PH/g face differences for DZ, mass-average geometric Z, bottom staggered W,
terrain and native DX. Post-RK caller supplies one-based own step index;
ACPHYS6 resets only at step1. Separate QHAIL transport/diffusion/root-flow and
no unregistered number transport survive. Specified-ring masking includes
all new mass/surface carries and raw precipitation.

State selection remains qh/hail_acc +6ACPHYS+6GSFC radii+SR; defaults carry
none of those new diagnostics. Exact State prefix/main schema is preserved
by the root's independent proof. MP7 binds in dispatch and post-RK paths;
WRF-order caller is mandatory. Native outputs supply exact rain+ice separately
from original total; Noah previous-step forcing uses original total override,
snow-only/graupel/hail channels. SR remains the held original frozen fraction.

RRTMG4/4 mediation retains GSFC micron values and multiplies cloud/ice/snow
by 1e-6 for general metre radii. Cold GSFC/general radii are zero for mp7.
The shared radius helper accepts4/5 flags, but actual new Goddard5/5 config
qualification remains MP8-only; MP7+5/5 is refused and is not claimed here.

Two concrete integration gaps were reported to root: restart validation did
not require qh/hail_acc (nor shape/dtype-check mp7 new fields on pickle/store),
and original PHYS_TOT history was omitted. Root owns restart fixes/tests.
Own ddc43c72c adds only PHYS_TOT output, literal left-associated original
physc+physe+physd+physs+physm+physf, same mass-grid shape, K/s. Production still
elides the individual six rates; cost is one REAL4 output/held field (4B/cell),
no scratch, no extra kernel launch. PHYS_TOT_SOURCE_PROOF.json proves all
prior output equation ASTs unchanged. Root owns State/history/spec wiring;
new ABI lowering remains queued until its sole CPU-JAX slot is released.

Registry census found no additional MP7 history omissions: six GSFC radii,
QHAIL, RAINNC/SNOWNC/GRAUPELNC/HAILNC/SR and PHYS_TOT have h/rh flags.
Individual six instantaneous budgets and five precipitation fluxes are '-'.
ACPHYS6 and derived ACPHYS_TOT are restart-only r; general radii are r only.
ACPHYS_TOT can be reconstructed exactly by original c+e+d+s+m+f association.

The coupled-smoke proof uses the real loader and official donating
_advance_chunk for own steps1/2. It changes only MP in the source namelist,
uses a read-only symlink input view, asserts CPU and records JIT mode, actual
work, finite state, full F2 and RSS. Launch before imports with JAX_PLATFORMS=cpu,
GPUWRF_JAX_CACHE=0, GPUWRF_FAST_DEFAULTS=0, GPUWRF_CENSUS=1 and
JAX_DISABLE_JIT unset/0. The global-disable trial already failed memory and
must not be repeated. The script asserts finite outputs, not zero guard
counts; final F2 disposition needs a manual matched release control whenever
nonzero guards appear. This is a genuine CPU execution smoke, not WRF
forecast fidelity or GPU performance evidence.
