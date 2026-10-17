# Grid FDDA reference correction and held heat-flux history

The reported PSFC failure came from an invalid original-WRF reference. The
assembled `wrffdda_d01` omitted `FieldType=104`; WRF returned before reading
the analyses and nudged toward zeros while still reporting SUCCESS. Correcting
TITLE alone had not repaired ingestion. Paired original-WRF three-step runs
prove zero NEW fields in the old file and byte-exact ingestion after adding
metadata. All 13 original forcing payloads were preserved.

The corrected original-WRF 24-hour reference compared with the completed GPU
run passes all 62 frozen D6 bounded fields at all 25 frames. PSFC hour-one RMSE
is **0.317180734 Pa**, previously 203.505694 Pa; the worst hourly RMSE is
0.981511024 Pa. The old GPU inventory remains **INCOMPLETE** solely because
`HFX_FDDA` is absent. This is not a GPU qualification of the changed executable.

The implementation also:

- Writes authentic WRF ingestion metadata and the singleton `fdda_levels`
  dimension for `MU_NDG`. Original WRF ingests all six NEW fields; all 363
  history arrays at each of four short-run frames remain byte-identical.
  PSU grid nudging supplies zero mass tendency.
- Retains `HFX_FDDA` at the FDDA call, before mass coupling, using WRF's literal
  `p/(287*th_phy)*1004*RTHNDGDTEN*dz8w`. It uses dry potential temperature.
  Cadence holds, rate decoupling and the past-end early return preserve the
  diagnostic. The PSU-only carry subclass leaves the spectral schema intact.
- Publishes held history through both production output consumers and validates
  its restart lifetime in pickle, NetCDF, checkpoint and native storage.
- Uses a continuous full carry and clock for daily FDDA, including auxiliary
  output intervals. Active old restarts without held history are refused.

CPU evidence includes 106 FDDA/spectral/oracle checks, 415 distinct passing
consumer checks, 12 unchanged radiation/land checks inherited through the
recorded source/default-program identity proof, and seven genuine deletion
assertions with no setup errors. Default Swiss native-facade identity remains
`2bf12cb8`; the enabled FDDA full-step trace is `403e8f5f`. The existing coupled
CPU smoke is carried forward with unchanged physical-source code and four
compiled tendency arrays byte-identical on real Swiss operands.

At 44×42×42, the new REAL history carry is 310,464 bytes. CPU compiler scratch
remains 708 bytes; interleaved component wall medians are 0.383608 ms before and
0.419988 ms after. These are CPU component measurements, not GPU speed or
whole-model memory claims. No new host transfer is introduced inside a step;
history/restart validation runs at publication boundaries. GPU launch count,
peak memory, compile cost and timing need the manager's changed-artifact arm.

The small receipts and source hashes are in `receipts/` and `QUALIFICATION.json`.
Raw original-WRF reference frames remain at
`<USER_HOME>/wrf_gpu2_lanes/t-fdda/TF02/full24`; the full frozen comparison is at
`<USER_HOME>/wrf_gpu2_lanes/t-fdda/TF03_score`. Temporary executable copies,
writable lookup tables, snapshots and test caches are disposable; reference
frames remain needed by the manager's GPU comparison.

Next: one independent critic, then manager-scheduled GPU qualification of the
new held diagnostic, actual-executable E41 scan, finite/census checks, and full
24-hour D6 plus inventory against the corrected original-WRF reference.

Current-main rebase composition and the 591 passing affected checks are sealed separately in [REBASE_FIX.json](REBASE_FIX.json). The reviewed `QUALIFICATION.json` remains byte-identical.
