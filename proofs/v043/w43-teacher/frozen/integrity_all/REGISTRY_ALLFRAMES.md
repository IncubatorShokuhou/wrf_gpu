# All-frames integrity registry — FROZEN 2026-10-06T21:14:40Z (manager 2026-10-06T21:13:54Z, pre-registered before the final set)

Applies identically to every final case (F1'' + 72 h 0227/0408/0115 + Monica). Degeneracy on ALL paired frames per domain.

## Tools (sha256)
- integrity COPY with --all-frames: integrity_all/scripts/wn3_integrity.py 413299de211072b0a38ca9dc6c9e3d3db504f2af0f16b1ba3ea0f5056f7b639b
- its scorer module:                integrity_all/scripts/wn3_score.py     cbc95179c95dd08d53ad564f2b81135d4eab14fd1a295d52e94fd2d0aff783a6
- frozen originals (untouched):     tools_src/scripts/wn3_integrity.py fa037a8bc7c024ae29f567f47519428b81679aa99d382197f7225ac3e68e68d5, wn3_score.py ba1b5972f191edfdac825341bc39d855085a34d083f30740f9a114a6c3dff24d
- registry (class rules):           integ_classes.py d3cfb72e21ef014adc41953a956627b09f818b52204f4c04deddd319bf406397  (UNCHANGED code)
- R4 cell-watch data:               r4_eval.py 73deaf41563304b40b3649103e9b81cfaeed6119a241e54864b13b2e446fd32e  (run with --integrity <all-frames JSON> => every frame)
- driver:                           integrity_all/classify_all.sh 738be63ce5bdeb7222a6d9a40e64918404fe1d94b7176cb35c33e5b7575ee5bd
- parity proof (2 cases x 3 domains, 6/6): integrity_all/proof/proof_result.json 78d18653d44fd2dd402681c0efc3c008cc483670297bb1a4af3b47c9b31bfe43

## Decision
(a) Companion window = +-1 h of an ACCEPTED parent on ANY frame of the same domain; caps UNCHANGED.
    Implementation: no code change — classify() pass 1 collects accepted parents from every frame present in the integrity JSON,
    so fed the --all-frames JSON it already applies (a). (The 21:1xZ wn3 0227 all-frames classification already used it.)
(b) R4 rain watch 0.01 mm stays (CPU and GPU hourly rain in the flagged cells).
(c) Owner verdicts (b-thompson, pristine/source evidence) BEFORE the final set for: d02 CANLIQ/CANWAT/FWET@03-02_11 (R4 rain watch) and
    d03 CANICE@03-02_03 (CPU 8.4e-4 > R5 cap 1e-6). No cap is relaxed without that evidence.

## Known consequence (0227 RC frames, bitwise the gated RC; P0227-GPU in brackets)
Under (a) the orphan companions stay NEW because no accepted parent is flagged within +-1 h on ANY frame:
d03 QRAINXY@03-02_10 (CPU 2.7e-5, also > R4 companion watch 1e-6), QGRAUP@03-02_19/03-03_00 (<= 6.6e-12, 03-03_00 = documented tail
34e1d8ec6), SNOW/SNOWC/SNOWH/TAUSS@03-02_20-21 (<= 2.3e-4 kg/m2 / 6.8e-5 / 1.9e-6 m / 5e-3) [member also d01 QNICE@03-02_15, CPU .0024].
These are NOT covered by (c) => a further owner/manager disposition is needed, or they remain FAIL-new-class in the final 0227 verdict.
