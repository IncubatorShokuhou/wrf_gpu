# AV_I — separate Thompson runtime-coefficient follow-up

**CPU-qualified candidate; GPU/E41 and the next composed program remain open.**
The public runtime binder refuses this experimental control until its own
GPU/E41/cost gates pass; the low-level CPU qualification path is retained.
The frozen four-parameter Nt candidate, snapshots and night launchers were not
changed. This report covers `av_i` alone, added after that candidate was sealed.

## WRF meaning and implementation

Pristine WRF 4.7.1 `phys/module_mp_thompson.F:161` declares private REAL
PARAMETER `av_i=1493.9`. Its two active uses are the ice mass and number
fall-speed expressions at :3681/:3686; :3684 is a commented alternative.
There is no initialization or table dependency. With `bv_i=1`, units are s⁻¹.
This is the actual coefficient, not a new multiplicative tuning factor.
Changing it is a **gpuwrf extension**; CPU-WRF needs recompilation for comparison.

The existing native namespace reads `av_i` as a REAL scalar operand and binds
`AV_I` in the unchanged equations. Only admission metadata, the existing catalog
tuple and private-oracle declaration allowlist changed. No kernel, adapter,
Nt closure, State schema or table changed. The input's original REAL values
are 1493.9000244140625 and 2987.800048828125 after rounding.

## Evidence

Same frozen mixed-phase fixture/24 columns as the Nt work: 44 levels, dt=18 s;
nonzero qc/qr/qi/qs/qg cells 198/265/256/360/153. These are CPU results [M].

| Gate | Result |
|---|---|
| AI02 original WRF, av_i=2987.8 | All 10 fields pass unchanged mixed-phase increment bounds on 24 columns |
| Default-value omission against changed WRF | Killed; the changed coefficient is active |
| AI01 compiled value reuse | OFF+ON produce exactly 2 JIT traces; changing av_i adds no trace and changes 4 output leaves |
| AI04 OFF native component HLO | BYTE vs original TP10 control: 628,924 bytes, SHA `12a9494b…`; canonical `0426b161…` |
| AI05 ON1493.9 vs ON2987.8 native HLO | BYTE: 629,202 bytes, SHA `bee99272…`; canonical `ee553ec0…`; one Pallas call each |
| AI06 host admission/catalog/restart/key checks | 65 passed in 2.61 s |
| Value-independent keys | static `6bd75efb…`, aval `f99c2260…`; changed values change restart digests; av_c/av_i names cannot collide |

No GPU execution, native CUDA compilation, warm-root timing or E41 scan was
performed for this follow-up. CPU CUDA-target lowering is compiler preflight.

## Localized ON-default rounding

AI01 intentionally retained the generic checker failure. The existing checker
allows two REAL ulps and only treats cloudw as widened REAL precipitation; no
shared checker or frozen GPU tool was changed.

| Output | Changed cells | Max absolute difference | Precision difference |
|---|---:|---:|---:|
| qc | 1 | 1.3877787807814457e-17 kg/kg | 1 REAL ulp |
| qi | 22 | 3.552713678800501e-15 kg/kg | 3 REAL ulps |
| Ni | 17 | 0.000244140625 #/kg | 4 REAL ulps |
| ice precipitation | 1 | 1.1368683772161603e-13 mm | 1 REAL ulp; 536,870,912 float64-carrier ulps |

The other 16 output leaves are byte-identical. Maximum relative differences
in these four fields are below 4.3e-7. The manager's documented few-REAL-ulp
rule admits this localized class after cause and original-WRF evidence; the
next composed validator must encode it explicitly without broadening unrelated
fields or disguising the float64-carrier difference.

AI03 uses the original ice fall-speed code objects with only the private scalar
binding changed. Eager static/runtime mass and number speeds are BYTE. Compiled
speeds differ by at most 2 REAL ulps (200 mass and 45 number cells). Optimized
CPU HLO shows the cause: static products fold into 5975.60059 / 3734.75049;
runtime paths multiply the scalar and then 4 / 2.50000024. Sedimentation and
final rounding propagate that small difference into qi/Ni. The ice precipitation
arrays are exactly representable as widened REAL32 in both arms:
`phys_thompson_prep._sed_registers` initializes and carries ppt in REAL32, while
`phys_thompson_full` has float64 output buffers. This is source/rounding evidence,
not a JAX-vs-JAX fidelity claim; AI02 supplies the original-WRF comparison.

## Cost and integration boundary

The source adds one scalar REAL input (4 bytes) to the existing column program,
with no new table or prognostic storage. Native component call count stays one.
That does not establish GPU kernel/transfer counts, speed, VRAM or readiness;
those remain unmeasured. Whole-root/GPU work is deferred until the parent
composes this follow-up after the frozen Nt gate/scheduling decision.

Product prototype: `f1d7b9f2d2`; attribution probe: `50bf00baef`; key test:
`ad00d1c6e8`. Measured source tree `1dcc704c7af603232d84a85b2e15a1dc6d129125`.
All experiment rows AI00–AI06 are in main EXPERIMENTS; source/hunk notice was
posted before the prototype commit. Own Fortran binaries, snapshot and caches
were cleaned after their last dependent arm.

Receipts: `<USER_HOME>/wrf_gpu2_lanes/w44-thparams/AV_I/AI01` through `AI06.xml`;
combined `AV_I/SUMMARY.json`, SHA256
`174c4eacc71216c5110ba77bcab99d0b07e2d67a21380168b6605ca2ded119ea`.
