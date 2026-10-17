# MERGE — exact 583dc78ca748e320b4be51034767699bb73c7a08

Same critic follow-up to `c0b054bb7` / the original MERGE-WITH-FIXES.
All three qualification conditions are closed for this diagnostic-retention
lane. Model tree: `53334b3c965fe03238d4df55ec8795dee82ee567`.
The Jensen exception is accepted as a small disclosed compiled deviation
under the standing FMA policy, as specified in the manager's review request.
**Its strict enclosure result remains 165 failures; no bound is widened.**

I independently audited the 84 sealed delta receipts, including compressed
raw-byte hashes, and rescored the saved native outputs with NumPy. No model
compile, repeated numerical run or GPU work was necessary.

- MP50/51/52/53: all saved expanded-native outputs at three evolving steps
  have zero violations of the unchanged original P348/P360 bounds. Original
  literal/FMA controls match the frozen hashes and fixture bytes; comparison
  policy code is byte-identical to the previously reviewed source.
- MP55: independently reproduced 165/8160 new-diagnostic values outside the
  original input-ULP/FMA interval, maximum 138 float32 boundary steps and
  relative excess `1.4297135713903125e-5`. All values and interval endpoints
  are finite; no excess has a zero boundary. Maximum absolute excesses by
  quantity: density 0.00323486328125 kg m-3, diameter 3.6670826375484467e-9 m,
  fall speed 2.86102294921875e-6 m s-1, aspect ratio 1.3113021850585938e-6.
- The affected outputs are exclusively `di1..3`, `rho1..3`, `vm1..3`,
  `phi1..3`, mapped to the twelve held D_ICE/RHO_ICE/V_ICE/PHI_ICE history
  leaves. The adapter builds prognostic inputs before extending its output
  mapping; held diagnostics never feed the scheme. Their values are not
  transported or consumed as forcing/radiation radii. All prior saved state
  outputs, including effective radii, independently match the old native
  observations byte-for-byte; the sealed runtime comparison additionally
  attests precipitation BYTE. No new physical producer arithmetic changed.
- All five actual Swiss full-RK runs complete two steps, own indices 1/2,
  zero nonfinite values, zero census guards, current producer-to-carry
  interior BYTE and full restart BYTE on the final model tree. This is CPU
  coupled smoke with FAST0, not GPU or forecast-trajectory qualification.
  Some diagnostics remain inactive during the short window; active original
  column and IO evidence are retained separately. MP50 step 2 has four
  nonzero cells per diagnostic, so the earlier blanket zero shorthand is
  not used as evidence.
- All five requested `test_microphysics_operational_runs_and_mutates`
  singleton XML records pass, with explicit 12G/no-swap wrapper logs.
  Final composition reuse is justified by exact complete canonical program
  hashes, every input fingerprint, nested closed constants and active output
  ABI for all five options. The 145 affected State/E78/writer checks pass.
- Both final registered CENSUS0 default hashes match: Swiss
  `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` and PROD
  `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
  The wrong CENSUS1 attempt remains preserved and explicitly superseded.

Source reuse was independently checked: 352 original producer/asset/oracle
blobs, 47 native/adapter/selector dependencies across measured and final
sources (including each actual numerical arm), and five shared column-helper
ASTs. The composed optional writer-profile fields precede the twelve ice
fields; both E78 validations remain present. The original qualification
packet is unmodified.

Disclosure is accurate in `docs/physics/ice_diagnostics.md`,
`DELTA_QUALIFICATION.json`, and `IC14-jensen-excess.json`: strict-enclosure
failures and magnitudes are explicit, old physical outputs remain BYTE,
and neither values nor bounds were adjusted. Integration may now replace
the text's "awaiting independent assessment" with this acceptance; retain
the numerical disclosure and the strict-failure classification.

Added diagnostic carry remains 12/24/48 B per cell. Native frontend call
count is unchanged; actual GPU layouts, stores, register pressure, compile
readiness, peak VRAM and performance are unmeasured. P3's separately accepted
ZERO-reference disposition does not waive its eight GPU44 component
residuals or MP52 compiler blocker. GPU/E41/full-forecast D6 and radar remain
separate qualification work, not claims of this MERGE verdict.

Evidence: `DELTA_AUDIT.json` and `delta_audit.py` in this review directory;
read-only rescore/audit wall 0.525 s, peak RSS 72,328 KiB. All prior review
checks and the genuine mutant are reused on their verified unchanged scope.
