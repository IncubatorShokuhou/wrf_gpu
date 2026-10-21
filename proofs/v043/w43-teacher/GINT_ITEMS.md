| case | G-INT pass | REAL FAIL items | annex-EXPLAINED items |
|---|---|---|---|
| wn3_20260227 | True | 0 | 4 |
| wn3_20260502 | False | 7 | 0 |
| wn3_20260614 | True | 0 | 0 |
| wn3_20260220 | False | 1 | 2 |
| wn3_20260608 | True | 0 | 0 |
| wn3_20260120 | True | 0 | 1 |
| era5_20240302 | False | 5 | 2 |
| era5_20240218 | False | 6 | 1 |

| case | G-INT | domain | field | frame (valid UTC) | annex (i)-(iv) check | registry note (frozen classify) | annex result |
|---|---|---|---|---|---|---|---|
| wn3_20260227 | PASS | d01 | QNICE | 2026-03-01_13 | flagged 4 (iv True) ; GPU exact 0 True (ii) ; CPU max 4.05e+00 cap 1e+01 (iii True) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | EXPLAINED |
| wn3_20260227 | PASS | d03 | CANICE | 2026-03-01_11 | flagged 6 (iv True) ; GPU exact 0 True (ii) ; CPU max 4.19e-05 cap 1e-04 (iii True) | GPU 0.0, CPU max abs 4.19e-05; R5 cap 1e-06 | EXPLAINED |
| wn3_20260227 | PASS | d03 | QGRAUP | 2026-03-02_23 | flagged 8 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.31e-11 cap 1e-10 (iii True) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | EXPLAINED |
| wn3_20260227 | PASS | d03 | QGRAUP | 2026-03-03_00 | flagged 3 (iv True) ; GPU exact 0 True (ii) ; CPU max 6.58e-12 cap 1e-10 (iii True) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | EXPLAINED |
| wn3_20260502 | FAIL | d01 | QICE | 2026-05-03_07 | flagged 93 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.59e-08 cap 1e-10 (iii False) | GPU 0.0, CPU max abs 1.59e-08; R2 needs GPU 0 and CPU max abs <= 1e-9 | REAL FAIL |
| wn3_20260502 | FAIL | d01 | QNICE | 2026-05-03_07 | flagged 101 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.26e+02 cap 1e+01 (iii False) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| wn3_20260502 | FAIL | d01 | QGRAUP | 2026-05-04_11 | flagged 1 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.88e-06 cap 1e-10 (iii False) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| wn3_20260502 | FAIL | d01 | QGRAUP | 2026-05-04_12 | flagged 82 (iv True) ; GPU exact 0 True (ii) ; CPU max 5.63e-06 cap 1e-10 (iii False) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| wn3_20260502 | FAIL | d01 | QGRAUP | 2026-05-05_04 | flagged 34 (iv True) ; GPU exact 0 True (ii) ; CPU max 7.31e-10 cap 1e-10 (iii False) | R2 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| wn3_20260502 | FAIL | d01 | QRAINXY | 2026-05-05_08 | flagged 2 (iv True) ; GPU exact 0 True (ii) ; CPU max 4.77e-07 cap 1e-07 (iii False) | R4 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| wn3_20260502 | FAIL | d03 | RAINNC | 2026-05-04_07 | (i) outside trace family | GPU 0.0, CPU max abs 1.86e-06 | REAL FAIL |
| wn3_20260220 | FAIL | d02 | QRAINXY | 2026-02-21_13 | flagged 1 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.62e-12 cap 1e-07 (iii True) | R4 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | EXPLAINED |
| wn3_20260220 | FAIL | d02 | QRAINXY | 2026-02-21_14 | flagged 1 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.56e-12 cap 1e-07 (iii True) | R4 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | EXPLAINED |
| wn3_20260220 | FAIL | d03 | QCLOUD | 2026-02-22_22 | (i) outside trace family | GPU 0.0, CPU max abs 8.87e-08 | REAL FAIL |
| wn3_20260120 | PASS | d03 | CANICE | 2026-01-23_19 | flagged 3 (iv True) ; GPU exact 0 True (ii) ; CPU max 3.82e-05 cap 1e-04 (iii True) | GPU 0.0, CPU max abs 3.82e-05; R5 cap 1e-06 | EXPLAINED |
| era5_20240302 | FAIL | d02 | CANICE | 2024-03-03_11 | flagged 1 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.15e-06 cap 1e-04 (iii True) | GPU 0.0, CPU max abs 1.15e-06; R5 cap 1e-06 | EXPLAINED |
| era5_20240302 | FAIL | d02 | CANLIQ | 2024-03-04_14 | flagged 4 (iv True) ; GPU exact 0 True (ii) ; CPU max 4.92e-04 cap 1e-04 (iii False) | GPU 0.0, CPU max abs 0.000492; R4 cell watch: <= 12 cells, CPU CANLIQ 0.000492 CANWAT 0.000492 mm, rain 1 h CPU 0.0551 GPU 0.0012 mm, GPU canopy 0/0 | REAL FAIL |
| era5_20240302 | FAIL | d02 | CANLIQ | 2024-03-04_15 | flagged 6 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.03e-04 cap 1e-04 (iii False) | GPU 0.0, CPU max abs 0.000103; R4 cell watch: <= 18 cells, CPU CANLIQ 0.000103 CANWAT 0.000103 mm, rain 1 h CPU 0.117 GPU 0.00535 mm, GPU canopy 0/0 | REAL FAIL |
| era5_20240302 | FAIL | d02 | CANWAT | 2024-03-04_14 | flagged 4 (iv True) ; GPU exact 0 True (ii) ; CPU max 4.92e-04 cap 1e-04 (iii False) | GPU 0.0, CPU max abs 0.000492; R4 cell watch: <= 12 cells, CPU CANLIQ 0.000492 CANWAT 0.000492 mm, rain 1 h CPU 0.0551 GPU 0.0012 mm, GPU canopy 0/0 | REAL FAIL |
| era5_20240302 | FAIL | d02 | CANWAT | 2024-03-04_15 | flagged 6 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.03e-04 cap 1e-04 (iii False) | GPU 0.0, CPU max abs 0.000103; R4 cell watch: <= 18 cells, CPU CANLIQ 0.000103 CANWAT 0.000103 mm, rain 1 h CPU 0.117 GPU 0.00535 mm, GPU canopy 0/0 | REAL FAIL |
| era5_20240302 | FAIL | d02 | FWET | 2024-03-04_14 | flagged 4 (iv True) ; GPU exact 0 True (ii) ; CPU max 9.47e-02 cap 5e-02 (iii False) | GPU 0.0, CPU max abs 0.0947; R4 cell watch: <= 12 cells, CPU CANLIQ 0.000492 CANWAT 0.000492 mm, rain 1 h CPU 0.0551 GPU 0.0012 mm, GPU canopy 0/0 | REAL FAIL |
| era5_20240302 | FAIL | d02 | FWET | 2024-03-04_15 | flagged 6 (iv True) ; GPU exact 0 True (ii) ; CPU max 4.50e-02 cap 5e-02 (iii True) | GPU 0.0, CPU max abs 0.045; R4 cell watch: <= 18 cells, CPU CANLIQ 0.000103 CANWAT 0.000103 mm, rain 1 h CPU 0.117 GPU 0.00535 mm, GPU canopy 0/0 | EXPLAINED |
| era5_20240218 | FAIL | d01 | QNRAIN | 2024-02-19_06 | (i) outside trace family | GPU 0.0, CPU max abs 8.79e-20 | REAL FAIL |
| era5_20240218 | FAIL | d01 | QRAIN | 2024-02-19_06 | (i) outside trace family | GPU 0.0, CPU max abs 1.39e-27 | REAL FAIL |
| era5_20240218 | FAIL | d01 | RAINNC | 2024-02-19_06 | (i) outside trace family | GPU 0.0, CPU max abs 0.00591 | REAL FAIL |
| era5_20240218 | FAIL | d02 | QRAINXY | 2024-02-21_15 | flagged 1 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.06e-07 cap 1e-07 (iii False) | R4 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| era5_20240218 | FAIL | d03 | QCLOUD | 2024-02-19_16 | (i) outside trace family | GPU 0.0, CPU max abs 9.84e-06 | REAL FAIL |
| era5_20240218 | FAIL | d03 | QRAINXY | 2024-02-21_10 | flagged 2 (iv True) ; GPU exact 0 True (ii) ; CPU max 1.26e-06 cap 1e-07 (iii False) | R4 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | REAL FAIL |
| era5_20240218 | FAIL | d03 | QRAINXY | 2024-02-21_22 | flagged 1 (iv True) ; GPU exact 0 True (ii) ; CPU max 2.03e-12 cap 1e-07 (iii True) | R4 companion invalid: no accepted parent-class flag within +-1 h; GPU 0.0 | EXPLAINED |
