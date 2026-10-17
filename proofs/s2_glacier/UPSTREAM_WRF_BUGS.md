# Pristine glacier source definedness

The standalone oracle compiles the unchanged WRF 4.7.1 glacier module with
production-style FP traps off and compiler initialization to zero. Rebuilding
with NaN REAL initialization changes **zero defined outputs** across all 4,896
option/regime cases. Inactive HCPCT snow slots are undefined OUT values and are
excluded using the entry ISNOW, rather than interpreted as physical capacities.
The port writes zero at those inactive slots; active values are byte exact.

The surrounding Noah-MP driver reads QSNOWXY before the glacier call and writes
the returned QSNOW afterward (:822/:1258). This is persistent model state.
It cannot be replaced by this step's precipitation or a history-only packet.

WATER_GLACIER has OPT_SNF branches 1/4, 2 and 3, with no branch for 5. Glacier
domains using opt_snf=5 are refused during input preparation. Natural-soil
wet-bulb partition qualification does not qualify that absent glacier branch.

## Defined per-point QRAINXY hold

The pristine `module_sf_noahmpdrv.F:823` loads `QRAIN = QRAINXY(I,J)` before
selecting the glacier branch. `NOAHMP_GLACIER` has no QRAIN argument and leaves
that loaded value unchanged. The driver writes it back to the same point's
`QRAINXY` at :1259; `NOAHMP_INIT` initializes that field to zero at :2141.
The port's glacier history zero therefore matches WRF's defined per-point
hold on the qualified cold-initialized path. Compiler initialization is not
needed to define this value. Arbitrary nonzero externally seeded `QRAINXY`
is outside this qualification. The glacier water budget uses its own local
rain partition.

The three assignments are source-bound in
[`receipts/QRAINXY_SOURCE.json`](receipts/QRAINXY_SOURCE.json), with pristine
driver SHA256
`29f16622740d7388a17ff9ba6c2f4640487e44f3f26b8c4bdd47aad42e0a4572`.
