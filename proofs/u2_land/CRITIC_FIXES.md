# R1–R4 closure

The independent verdict on `a7efa75a2` was MERGE-WITH-FIXES. Its pristine
references, numerical/native receipts, key guards and default identities remain
accepted; the fixes below address the specific public-reader and cold-input seams.

* **R1:** checkpoint reads authenticate requested NoahMPOptions, SST-skin
  enablement and all four Noah-MP group selections, in addition to the saved
  bundle and existing bound-table/map digest. Public controls restore all five
  groups byte-exactly and refuse all five changed requests. Stateless DVEG8→9
  is refused even when the table digest is identical.
* **R2:** both reduced readers refuse active SST/irrigation/carbon/groundwater/
  tile-drain groups and direct callers to `read_checkpoint_with_runtime_state`.
  Ten negative calls and legacy/inactive positive controls exercise the public APIs.
* **R3:** cold DVEG7/8/9 loading applies pristine NOAHMP_INIT:2163–2182:
  natural LAI=`MAX(input,.05)` and XSAI=`MAX(.1*LAI,.05)`, in REAL4. Nonvegetated
  categories receive zero. Three actual Swiss loader regressions cover 0/.01/.04,
  both adjacent REAL4 values around .05, .05 itself, larger LAI, and urban/ice/
  barren/water cells. First PHENOLOGY now preserves the initialized canopy.
  Evolved below-threshold values still enter the original zero-canopy branch;
  full checkpoint continuation preserves them without cold reinitialization.
* **R4:** crop phenology sets/restores its own REAL flag and clears JIT caches.
  The SST prefix assertion names the retained pre-MMF schema. Twenty-four
  advanced-water literal tests retain 2,496/2,496 BYTE parity. The compiled
  test checks the unchanged independent original bounds. Sixteen modes are
  within bounds; eight remain explicitly **XFAIL**, with raw indices and maximum
  excess exactly matching the published LAND48 receipt. The largest excess is
  8.278402674477547e-7. Nominal tolerances and frozen fixture bytes are unchanged.

LAND58 on `1f0250a45` passes 187 affected checks, with those eight explicit
compiled XFAILs, zero failures/errors, in 313.58 s CPU. It includes contracts,
legacy MP fail-closed, D31, NativeRestartStore, shared checkpoint consumers,
owned carry roundtrips and all regressions above. Raw XML/log/JSON are retained;
compiled XFAIL is not a physics PASS or a forecast-fidelity claim.

Only three production files change relative to the reviewed candidate:
checkpoint reader host logic, cold land initialization, and the loader's
DVEG7/8/9 branch. Physics, coupling, dynamics, kernels and hot operational step/
carry sources are byte-identical. The source seal records the exact unchanged
AST of the loader after removing only that new conditional branch, and the
unchanged existing initializer definitions. A/C/D smoke evidence and all
numerical/native receipts are reused. The affected B smoke and both official
default identity probes are refreshed separately and recorded in the fix receipt.

The guards execute only during checkpoint reading. Cold initialization uses
NumPy scratch arrays and two replacement LAI/XSAI maps once, with no additional
prognostic fields or timestep launches/transfers. The default branch skips it.
The 12/48/24-or-36/12/4 B/cell figures describe prognostic groups only, not total
option memory: crop parameter maps alone are 524 B/cell by schema before
aliasing/compiler elimination, with initial seeds and the 784 B/cell canopy
input temporary additional. Whole-run GPU peak memory and speed remain unknown.
