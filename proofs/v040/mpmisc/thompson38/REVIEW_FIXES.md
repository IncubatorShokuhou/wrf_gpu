# mp38 reviewer source corrections

Review rv-s2-mpmisc accepted the experimental CPU scope with two required
source corrections. Production commit4da8ed1fb closes both on one candidate.

Cloud sedimentation now retains raw positive DZQ throughout the mp38
return-rates route, for both the original500 m band and flux divergence
(module_mp_thompson.F:3646–3655,3830–3833). The legacy mp28 route retains its
previous specialization. The isolated quarter-metre column and a500 m band
edge are compared with an independent no-FMA REAL Fortran statement oracle,
using actual source-qualified velocity operands.

HailContext retains an alias of incoming raw qv1d while rates use the existing
private working floor. The first mixed-phase reconstruction (:3192), each
subsequent cumulative reconstruction and final writeback (:3974) use that raw
baseline. No new prognostic field or input table is introduced. Constructed
negative(-1e-7) and nonnegative subfloor(5e-11) cases exercise128 active source
cells and800 stage reconstructions. Every saved baseline and reconstructed
vapor agrees with original REAL statements. These are operator witnesses,
not pristine full-column replays of the constructed low-vapor inputs or
actual forecast-event claims.

MP59 commits the tests before the correction: all three fail by numerical
assertions on the reviewed production source, with no setup failures.
MP60 passes72 affected source/operator/caller/carry/contract/legacy/NetCDF
checks in22.82 s. The final test also separates the isolated quarter-metre
flux from the band-edge witness. Both tests fail if their source correction
is removed; the vapor test checks the intermediate stage, not only output.
MP65 passes the final three source tests and nine genuine table-provider
key/HLO/output collision regressions:12 passed25.93 s on immutable3f0d4bab8,
whose production source is byte-identical to4da8ed1fb.

MP62 replays the unchanged original references on immutable4da8ed1fb:
6160 physical+77 surface+440 direct-theta values BYTE, and three evolving
own steps1680 physical+21 surface+120 direct-theta values BYTE. All eleven
compiled cases are finite REAL32 and byte-identical to their previous H66
outputs. Independently frozen compiler intervals still have115/6237 excesses;
no tolerance or interval changes. Fifteen default8/28 AST specializations
remain exact. See review_fixes_cpu.json and the MP62 artifact directory.

[M CPU] Changed six-column executable: input18240 B and output18552 B,
temporary312424 B versus previous314536 B (2112 B fewer), no aliases.
Cold compile+first execution for6/3/2 columns is21.957/17.963/18.742 s.
Six-column warm median1.396 ms over nine calls (1.038–2.337 ms); this is
component timing, not forecast or GPU speed. Buffer statistics exclude the
929643480 B table payload and possible legacy duplication. [I] Raw QV aliases
an already-live input; removing the DZ floor introduces no host transfer or
new scratch array at source. Actual GPU launches, memory, E41 and D6 remain
pending. No whole-model fp32 claim is made for retained-wide CPU smoke mode.

The first artifact arm MP61 was discarded for configuration before a compiled
receipt: wrong FAST environment spelling and default shared-cache activation.
The retained checker forces FAST_DEFAULTS0/JAX_CACHE0 before imports and
asserts caching is disabled. MP62a completed nine groups, then encountered a
threshold-reference path setup error; MP62b reuses those same-commit receipts
and runs the remaining affected gates. No science failure is concealed and
no original source, reference or compiler bound was altered.

MP63/64 repeat the official native-facade default trace identities on
immutable4da8ed1fb. Swiss is exactly
f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d
(59.5 s CPU load+trace); PROD d01 is exactly
6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b
(91.9 s). The owned wrapper changes only the temporary-directory location.
The PROD WN2 inputs establish code identity only. These are not execution,
executable-readiness or GPU measurements.

MP66 repeats the authentic Swiss42x42x44/dt18/MYNN5/Noah-MP4/RRTMG4/4/CU0
two-step official CPU smoke on immutable3f0d4bab8 (production BYTE4da8ed1fb).
Nonfinite0, all guards0, pre-mask events0 and all guard water changes0.
Load56.730 s; compile+first293.815 s; next3.553 s; peak18119564 KiB.
The five held surfaces are REAL32; qvolg/nwfa/nifa retain wide CPU storage.
The namelist SHA matches MP56. The observed times are higher than its
233.826/2.347 s, while peak RSS is slightly lower than18128108 KiB.
These single observations on the shared CPU host do not isolate the cause
of the timing difference. Actual GPU cost remains unknown; the measured
component temporary buffer decreases2112 B and no new transfer is introduced
at source. Do not infer a GPU speed claim from either CPU smoke.

All owned pytest/oracle binaries, temporary input views and detached snapshots
were removed after the last dependent arm. The pre-existing shared CPU cache
accidentally selected by early MP59/60/61 is disclosed in MAIN INBOX and the
lane notes; it was not broadly deleted because other workers own entries.
All final artifact, identity, key-regression and coupled arms explicitly
disable persistent caching. Ferrier/Goddard/unaffected results are reused
without duplicate validation. Manager may merge under the existing
MERGE-WITH-FIXES experimental CPU verdict; GPU/full-forecast gates remain open.
MP67 also validates the corrected clean archivee396b760e:92 assets,
738789201 compressed bytes, original asset completeness, NPZ CRC/finite,
strict JSON and production package-data checks all pass4.999 s. No extraction
or persistent cache is created. Subsequent handoff changes are Markdown only.
