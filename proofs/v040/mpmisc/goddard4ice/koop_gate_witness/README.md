# Independent original mp38 Koop threshold witness (fresh10/11)

The original Fortran condition at module_mp_thompson.F:2634–2640 uses explicit
snow number NS, not the separately fitted zeroth snow moment SMO0. NS is the
two-gamma integral at2083–2088; inactive NS starts at zero at1790. Substituting
SMO0 changes the genuinely active near-999e3 condition.

Fresh input profile: T230K (subject only to original TH*PII rounding),
QV=1.5*originalRSIF(P,230), QI=1e-5, QS=1e-4, cloud/rain/graupel0, W1m/s,
DT18s, positive production aerosol inputs. A pilot original execution supplies
NS/RHO. The two callers set initial NI=(999000-NS +/-2)/RHO in REAL4.
The original size band remains active; actual inputs are preserved in the
full public savepoints. These profiles were designed from original quantities
before evaluating the repaired port; no candidate result sets the target.

Original fresh10 has38/40 Koop gates active and38 nonzero PNI_IHA values.
Original fresh11 has0/40 active and all PNI_IHA zero. The old SMO0 expression
would pass39/40 cells in each: it changes39 genuine fresh11 gates. Exact
original per-level gate prints and staged fields are retained. Counts are
measured, not an assertion that every target hits its intended side. The
complete O0-debug and O2 original public38 column arrays are BYTE equal for
both callers; the original gate facts therefore apply to the normal reference.

original_columns.npz contains all original input/output arrays and scalars;
original_stage_fields.npz contains original NS/SMO0/RHO/NI/T/SSATI and five
DOUBLE freezing-rate arrays. receipt.json pins their hashes, profile targets,
actual original gate prints and cached-table preservation. DEN in the legacy
project driver is diagnostic only: production density is
0.622*P/(287.04*T*(QV+0.622)), also retained from GDB RHO.

run.py changes only the project-authored driver, links cached unmodified
original module objects and reads existing original tables. Source SHA
fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802;
no original body/table was edited. Cached674698000B table inode/mtime/size
is unchanged. Init has both production aerosol and hail switches enabled.
Every arm is CPU-only, nice19, bounded and uses no JAX.

initialized_scalars.json contains exact original initialized REAL bits from
GDB at1648: T1_QR_EV0x3f47ae14, T2_QR_EV0x42135207, SC30x3f5bb0e7,
CSG150x3f65e565 and CSE150x3fd15e9e. The initial C-language dump attempt
failed because module globals need qualification; GDB Python parse_and_eval
under Fortran language preserves their exact REAL values and bits.

Independent snow coefficient audit finds native REAL casts of all10SA,
10SB and17CSE values BYTE exact. The remaining polynomial defect is expression
association, not coefficient data: coefficient-first original left products
versus cached TC2/order2 products. A tiny84-evaluation arithmetic witness has
four non-BYTE coefficient results (maximum4.768e-7); this is an association
counterexample, not a forecast-fidelity or candidate residual claim.

Full mp38 qualification remains the Thompson helper/root gate. These original
active holdouts and constants do not by themselves admit the option or claim
GPU execution/performance.


## Original cloud gamma coefficient audit (E83)

A separate initialized-original GDB read at1648/run4 captures CCE(5,15),
CCG(5,15), OCG1(15) and OCG2(15), first Fortran index varying fastest.
GDB confirms every array element is real(kind=4),4bytes. Original source and
debug-module hashes, asset hash and each non-BYTE index/raw hex pair are
pinned in cloud_coefficients_audit.json. No JAX or physical-column rerun.

Native REAL asset casts match only94/180 original initialized values:
CCE75/75 BYTE; CCG12/75 BYTE (63differences,max21ULP); OCG1 5/15BYTE
(10differences,max5ULP); OCG2 2/15BYTE (13differences,max18ULP). For example,
original CCG(1,5)=120.00000762939453 (0x42f00001), while the old asset
casts to120.0 (0x42f00000). Original OCG2(2)=0.00833333283662796
(0x3c088888), while the asset has0.008333333767950535 (0x3c088889).

original_cloud_coefficients.npz stores the exact original initialized REAL4
arrays (stable SHA31d4c7df... across the independent dtype readback). The
coefficient defect is proven independently of candidate outputs; whether it
explains remaining mp38 number residuals is for the helper's changed-artifact
measurement. No shared asset or production code was edited. Private mp38
bindings can use this reference without changing older mp8/mp28 programs.

## Selected original frozen coefficient audit (E83)

The independent GDB audit at1648/run4 reads initialized CIE/CIG(7),
CRE/CRG(13), ice mass/fall parameters, and the hail coefficient rows2,3,6,7,
9,11,12 for all9 density bins. It also checks AM_G and distribution inverses,
Brown--Lawler parameters and the actual original T2_QR_QI prefix.
All selected source coefficients are REAL(kind=4),4bytes; original DOUBLE
distribution/rate locals remain separate. Raw bits, exact source/object and
product-source hashes and mappings are in frozen_coefficients_audit.json.

All167 effective selected bindings match original bits. The extra legacy
CRG8 diagnostic differs by1ULP: original720.0000610351562(0x44340001),
legacy720(0x44340000). It is not imported by the active column code; its
actual consumer uses the precomputed original WRF_REAL_CONSTANTS T2_QR_QI
prefix, independently checked BYTE here. CIG7/CRG12 are original
1.3293403387069702 and need no replacement. This closes the selected
frozen-coefficient theory; it does not prove remaining expression order.
No JAX or physics rerun, production edits, full table dumps or allocations
were required. The original cached table inode/size/mtime are unchanged.

## Narrow sublimation scalar promotion audit

Original initialized t1_qg_sd and t1_qs_sd are REAL(kind=4),4bytes,
0.8600000143051147 (0x3f5c28f6). Their exact promotion to DOUBLE is
0x3feb851ec0000000; the Python0.86 constant is0x3feb851eb851eb85.
The difference is1.4305114759416426e-8, despite identical REAL casts.
The original hail term at2699/2829 multiplies this REAL coefficient by
DOUBLE ilamg**REAL cge(10,1), so the promoted original value is required.
TC's Python T1_SUBL_QG enters the DOUBLE power term at3205 directly.
This is a concrete source binding defect; a paired physical-rate comparison
is still needed to assign the observed hail residual to it.

Optional t2_qs_sd is original REAL1.5197088718414307 (0x3fc285d2),
DOUBLE promotion0x3ff850ba40000000; the Python expression is
0x3ff850ba345c5d31. Its REAL cast also agrees. Snow relevance depends
on operand kinds and is not claimed here. No nonexistent ice coefficient
is included. Source/object hashes, exact original types and all raw bits
are retained in sublimation_scalars_audit.json; cached table unchanged.
The earlier frozen audit checked REAL binding bits, which do not establish
DOUBLE promotion identity for a Python coefficient in a DOUBLE island.

## Actual case5 selected rain-freezing table audit

All40 original index components at the10 actual rain-table lookups
(levels8–17) are BYTE equal to the current saved candidate indices.
Only16 DOUBLE entries (TPI/TNI atlevels10–17) were read: all16 match
the exact original freezeH2O.dat records, and all16 differ from the legacy
thompson-tables-v1.npz asset. The maximum relative differences are
3.7439867164823036e-7 (mass) and3.564510264624593e-7 (number).
Both original and asset values are DOUBLE; this is a data difference, not
a cast or indexing error. selected_qrfz_audit.json pins each index/value/raw
bit pair, original source/object hashes and unchanged cached-file provenance.

At original level15, indices(19,14,13,19) select TPI
1.4217468001067902e-12 versus legacy1.4217471390947867e-12, and TNI
0.00015961535041319477 versus legacy0.0001596153884381044.
All16 candidate pre-limiter rates exactly equal their legacy entry times
original REAL odts. All8 original PNI_RFZ rates at2960 exactly equal
the original TNI entry times REAL odts. Original PRI_RFZ at2960 has
already passed the rain-availability limiter at2904–2910;5/8 selected
levels differ from the pre-limiter mass product and are not projected as
unchanged. The table defect therefore directly explains the selected
number-rate differences; downstream limited mass needs the composed gate.

The read-only cached table is the original CPU case's freezeH2O.dat, read
by pristine freezeH2O at4630/4631. Its first records are big-endian
DOUBLE with4byte record markers and27,106,200bytes per rain record.
The legacy extractor script uses NVFORTRAN; this audit does not establish
the precise historical math/build difference that generated its values.
No table regeneration, full dump, original-body or production edit, JAX
execution or full-domain allocation was used. The Thompson helper owns
the authorized exact private table extraction and affected science gate.

## Original threshold-number stage trace

Fresh10/11 each has one negative incoming Ni level: level1 is
-10015.9638671875 / -10013.291015625 because original NS there exceeds
the999000 target. This level is explicitly a negative-input stress case;
the other39 input levels remain positive and the original measured Koop
gate counts38/40 versus0/40 remain unchanged. The target was derived from
original NS, not fitted to candidate results.

The relinked original caller's80 NI inputs match the committed witness
BYTE, and its two final level1 Ni values match the old original reference.
Original raw NI1D remains the held baseline while its private working NI
is floored. The candidate clips Ni tozero in _clip_species_aero before
HailContext captures the baseline. Original3035–3053 explicitly uses raw
NI1D in its size/number balance. This changes tendency subtraction and
subsequent cancellation, even though the adjusted working moments are
close. Source, preSED and postSED fields each match the other39 levels.

For fresh10 level1, source NITEN is1066.990234375 versus510.5478820800781;
preSED NI is13752.2646484375 versus13752.267578125. Applied final Ni before
rebalancing is9236.1787109375 versus9236.1806640625 (two REAL ULP).
The original final DOUBLE slope is13431.298828125 versus13431.2998046875;
final Ni is9236.1884765625 versus9236.1904296875. These slope/output
differences are downstream of the held-baseline difference. Fresh11 has
the same baseline defect but final rounding happens to coincide.

threshold_number_audit.json retains original source/object/caller/compiled
executable hashes, saved candidate-stage hash, every original stage and
final scalar kind/raw bits. original_threshold_number_stages.npz packs
118 small original arrays/scalars. The original physics module was not
edited, cached tables were read only, and only the authored caller was
compiled. Its executable/object/table symlinks were removed after the arm.
No JAX execution or repeated full numerical qualification was performed;
the Thompson helper owns the source correction and final composed gate.
