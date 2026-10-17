# Regional stream qualification

`QUALIFICATION.json` binds the model source tree, scope and compact receipts.
The original CPU-WRF auxhist7 fixture has five frames, thirteen surface fields,
two mass centers, units, tolerances and input/output checksums. `manifest.yaml`
also passes the repository fixture-manifest validator.

The component gates cover all three region types, every-step/exact-step-aligned
cadence, all five statistics, partial-window continuation in both restart
formats, real native two-domain orchestration, authenticated output replay,
foreign-file refusal, active-state admission and a backend-dark CLI. One genuine
production-hook deletion fails by assertion. Same-shape renamed/moved streams
reuse one JIT variant; the old/new metadata yields identical reducer IR.

The supported CPU path ran three real Swiss steps per arm: 173 shared carry
leaves byte-exact, finite, zero guards. The native GPU ran four comparison steps
and nine interleaved 16-step timing pairs. Its measured source is recorded in the
receipt; source/IR and full composed ON-program/constant seals carry forward the
implementation evidence. Native ELF identity across different compilations is
not asserted. Both measured executables pass full E41; the ON trace has zero
DtoH and only 24 bytes of HtoD before the first kernel. Timing ranges overlap.
Compiled footprint means arguments + outputs + temporaries - aliases; ON adds
956 argument bytes, 2360 temporary bytes, and 3388 footprint bytes in this case.

The existing native model QC guard visits and F2 negatives remain explicitly
reported. This packet does not clear the global F2, long forecast, station RMSE,
S1, R1 or cadence-weighted S3 gates. Large caches, snapshots and raw profiler
exports are disposable; retained JSON reports and checksums are the audit record.
