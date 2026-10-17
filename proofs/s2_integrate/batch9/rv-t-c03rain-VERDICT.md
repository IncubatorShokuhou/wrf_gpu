# MERGE — t-c03rain after 1612acc49

Reviewed candidate `9a483f816242c87cd5a5e95ca049434016763965` against
`1612acc495b890832147622557f70edff2bf47ae`. Only four product source files
change: QNSE adapter, operational history initialization/forwarding, and nested
loader/lead-zero writer. Lin through 1612 and E41 fixes are outside this review.
No product fixes are required.

Independent CPU evidence on a clean detached candidate, cores 16–19, nice 19,
`JAX_PLATFORMS=cpu`, `GPUWRF_JAX_CACHE=0`, OMP 1, explicit QUIET guards:

| Gate | Result [M] | Receipt under `<USER_HOME>/wrf_gpu2_lanes/rv-t-c03rain/` |
|---|---|---|
| QNSE coupling, MYJ consumers, lead-zero, three writer modes, restart and NetCDF locking | 133 PASS, one known non-strict XPASS; zero failures/errors | RV01B.xml |
| Original QNSE REAL4 literal and independent compiled-input enclosure checks | 62 PASS; 29 existing literal-backend warnings | RV02.xml |
| Real c03/SF2 and c26/SF4 Noah-MP loader preserves both REAL history leaves | 2 PASS | RV05-loader.xml |
| Four public restart formats reject each missing SF4 wind-history leaf | 8 PASS | RV08-legacy.xml |
| Delete SF4 forwarding before Noah/no-LSM overlay | 2 numerical assertion failures; zero setup errors | RV06-history.xml |
| Substitute ordinary winds for equivalent-roughness producer outputs | 1 numerical assertion failure; zero setup errors | RV06-producer.xml |
| Delete actual loader reseed wind kwargs | 1 SF2 and 1 SF4 array assertion failure; zero setup errors | RV07-reseed-mutant.xml (SF2 only), RV07B-reseed-sf4-mutant.xml |
| Unchanged official native-facade release probe | Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` EXACT | RV03-swiss.json, RV03-prod.json |
| Reuse/source admission | 9 numerical/kernel/oracle files byte-identical to 1612; all 6 cited producer receipt hashes verified | RV04-source-audit.json |

Total: **205 independent PASS, one known XPASS, five genuine mutant assertions**.
RV01A selected a nonexistent oracle test filename and ran no tests. RV07's SF4
case encountered a critic harness `inspect.getsource` error after its preceding
SF2 mutation; that error is not counted as a kill. Corrected per-test mutation
isolation was rerun only for SF4 (RV07B). Raw failed launches remain retained.

Source/reference findings:

- MYJSFC `module_sf_myjsfc.F:985–1018` and QNSESFC
  `module_sf_qnsesfc.F:928–961` produce special equivalent-roughness winds:
  sea retains U10/V10; land uses `UMFLX/EKMS10+UZ0` and `VMFLX/EKMS10+VZ0`.
  These are not earth-rotation diagnostics. Both adapter paths expose the
  existing surface solve's outputs at its actual time level.
- History forwards these outputs before the land overlay; its write method
  preserves absent keys. The real loader's later reseed retains the leaves for
  SF2/SF4. This site is now independently deletion-sensitive on both real cases.
- Pristine `module_physics_init.F:1323–1327` initializes both special winds from
  U10/V10 for SF2/SF4. Lead-zero writer follows this even when input special
  fields already exist. Other schemes retain absent fields, rather than zero
  outputs. The common held-history map serves both pipelines and all three
  writer modes; metadata matches Registry.EM_COMMON:1069–1070.
- Existing QNSE E78 validation iterates every history field and rejects missing
  special winds. All four public restart formats preserve the populated leaves
  byte-exact; eight additional missing-leaf refusals passed. Old SF4 restarts
  lacking the new history are intentionally refused.

Performance and scope: the existing fixed QNSE surface kernel already returns
both winds. Forwarding introduces no extra surface solve, Pallas call, profile
array, timestep host transfer, table or static input [M source]. Two extra REAL
surface history buffers are `2*ny*nx*4` bytes (14,112 bytes for 42×42) [I]; this
is not measured peak VRAM. Added carry stores/traffic and any resulting fusion,
compile/readiness or timing change for SF4 remain [U]. No SF4 speed or GPU/D6
qualification is asserted. The default canonical programs are independently
unchanged; no native-binary identity is inferred from CPU jaxpr identity.

The producer's sealed full c03 25-frame D6 receipt and selected-c03 canonical
program reuse proof are hash-verified, not rerun here. E41 remains disclosed and
assigned to t-e41opt; this verdict neither changes that gate nor approves a
release. No reviewer GPU work occurred.
