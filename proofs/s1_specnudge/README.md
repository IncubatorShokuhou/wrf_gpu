# Spectral FDDA CPU qualification

The reference is unmodified pristine WRF `phys/module_fdda_spnudging.F` and its
FFTPACK5 dependency closure, compiled in a lane-owned directory. The wrapper
reproduces the `fddagd_driver` SPNUDGING branch and the literal caller operations
in `calculate_phy_tend`, `phy_fg_ten` and `phy_prep_part2`. Domain stubs provide
storage and diagnostics only; no reference arithmetic is replaced.

`savepoints/MANIFEST.json` binds 40 compressed fixtures to source SHA256s,
original CPU-WRF Swiss h12/h13/h15 crops and option values. Two horizontal sheets
exercise odd/even FFT lengths. Ten regimes cover basic filtering, DC-only,
all modes, negative cutoff, PBL precedence/lid, fixed height ramps, both signed
end ramps, cadence holding and zeroing. There are 336 calls across the REAL32
and precision-promoted REAL64 variants, with uncoupled, coupled, owned and
decoupled outputs for U/V/T/PH/Q. REAL32 is original WRF precision. REAL64 is a
diagnostic build that also promotes FFTPACK's explicit `REAL(kind=4)` using
`-freal-4-real-8`; `-fdefault-real-8` alone gives an invalid ABI.

The frozen comparison bounds are in `manifest.yaml` and the committed tests.
FFTPACK and XLA use different radix trees and FMA association; the limits scale
with machine epsilon and original operands, with no arbitrary absolute floor.
They do not establish bitwise equality. Active-field census and real deletion
controls prevent zero-forcing or weight-deletion false passes.

Generate with CPU affinity and the QUIET guard required by AGENTS.md:

```sh
bash proofs/s1_specnudge/oracle/build.sh <USER_HOME>/wrf_gpu2_lanes/s1-specnudge/oracle
python proofs/s1_specnudge/oracle/make_savepoints.py <USER_HOME>/wrf_gpu2_lanes/s1-specnudge/oracle proofs/s1_specnudge/savepoints
python .agent/skills/building-wrf-oracles/scripts/validate_fixture_manifest.py proofs/s1_specnudge/manifest.yaml
python -m pytest tests/v040/fdda/test_s1_spectral_oracle.py tests/v040/fdda/test_s1_spectral_wiring.py --basetemp <USER_HOME>/wrf_gpu2_lanes/s1-specnudge/pytest
```

The wave's explicit requirement to track every test asset takes precedence over
the general oracle skill's manifest-only storage preference. These are small
operator fixtures, not large WRF output files; full CPU-WRF outputs stay outside
the repository and are removed after their final dependent comparison.

`prepare_cpu_cases.py` creates original CPU-WRF Swiss control/spectral runs.
`SPECTRAL_WRF_TABLE_DIR` points to existing original WRF Thompson lookup tables;
they are copied locally to avoid cold regeneration and reference writes.
The executable is an exact RD11 CPU reference copy and is checked for E104
instrumented dump paths. The assembled o2 analysis values are retained unchanged;
only truthful V4-compatible metadata and required `FieldType=104` attributes are
added. CPU-WRF otherwise silently leaves analyses zero despite a SUCCESS log.
Verify ingested analysis values via debug output and restart before trusting
such a run. Packed WRF fdda3d I/O has a mass footprint for U/V/PH; unused
east/north/top public-layout slots do not enter the spectral operator.

`probe.py` reuses the established o2 CLI probe with lane-owned artifacts,
spectral keys and saved PH/initial states. `O1_EXEC_STEPS=0` with release defaults
checks the existing Swiss program hash; a legacy CPU execution is a finite
coupled smoke, not native GPU qualification. The original CPU-WRF state
comparison, performance/memory receipts and any limitations are reported
separately from the standalone oracle gate. GPU qualification remains pending.

Final numerical source is byte-identical to the tested early-return candidate;
the main rebase preserves both legacy coupled program hashes (OFF `3b9e3c56`,
spectral `a365cc4f`) and the released default `2bf12cb8` hash. Composed restart
checks retain both the lake and spectral schema guards. The qualification packet
in `receipts/` records the frozen gates, short original-CPU comparison and CPU
cost/memory evidence. Full model outputs are deliberately not checked in.
