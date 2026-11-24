====================================================================================================
FINAL SUMMARY

=== CROSS-SYSTEM ranking, orbit-consistent estimator (80 combos common to 4 systems) ===
   L  m  q | mean err  max err | lorenz_r0 rossler_r logistic_ henon_map
   2  4 10 |     0.8%     2.1% |   +0.9126   +0.0699   +0.3465   +0.4199
   1  4  8 |     1.1%     2.7% |   +0.8808   +0.0720   +0.3466   +0.4229
   2  4  9 |     2.3%     7.9% |   +0.9075   +0.0771   +0.3466   +0.4234
   2  4  7 |     2.9%     8.8% |   +0.9279   +0.0651   +0.3466   +0.4215
   2  4  8 |     3.5%     9.5% |   +0.9917   +0.0685   +0.3470   +0.4214
   2  4  6 |     3.9%    10.3% |   +0.9287   +0.0641   +0.3467   +0.4302
   1  4  6 |     4.1%     8.4% |   +0.8397   +0.0654   +0.3467   +0.4229
   1  3  5 |     4.4%    16.1% |   +0.9089   +0.0599   +0.3467   +0.4229
  reference (5, 6, 5) is not in the common grid.
  theory: lorenz_r01=0.9056, rossler_r01=0.0714, logistic_map_r01=0.3466, henon_map_r01=0.4192

=== CROSS-SYSTEM ranking, legacy (consecutive Jacobians) estimator (80 combos common to 4 systems) ===
   L  m  q | mean err  max err | lorenz_r0 rossler_r logistic_ henon_map
   1  4  8 |     1.1%     2.7% |   +0.8808   +0.0720   +0.3466   +0.4229
   1  4  6 |     4.1%     8.4% |   +0.8397   +0.0654   +0.3467   +0.4229
   1  3  5 |     4.4%    16.1% |   +0.9089   +0.0599   +0.3467   +0.4229
   1  3  8 |     4.7%    14.8% |   +0.9350   +0.0609   +0.3466   +0.4229
   1  4  7 |     5.1%    13.2% |   +0.9633   +0.0808   +0.3466   +0.4229
   1  3  7 |     8.7%    18.2% |   +1.0469   +0.0584   +0.3467   +0.4230
   1  4  4 |     9.5%    32.7% |   +0.8648   +0.0480   +0.3467   +0.4229
   1  4  5 |     9.9%    20.1% |   +1.0744   +0.0570   +0.3467   +0.4229
  reference (5, 6, 5) is not in the common grid.
  theory: lorenz_r01=0.9056, rossler_r01=0.0714, logistic_map_r01=0.3466, henon_map_r01=0.4192

saved sweep_out/sweep_all.csv, sweep_summary.json and sweep_<name>.json
CAUTION: the best (L,m,q) of one system on one realisation is partly selected by luck (min over many combos). Trust combos that rank well across systems, then validate the few winners on NEW realisations (other initial conditions / seeds).
