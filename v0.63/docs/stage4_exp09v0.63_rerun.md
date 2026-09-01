# Stage 4 — Exp09 ControlSim v0.63 Rerun

## Reference Audit

The frozen v0.62 Exp09 YAML, v0.63 canonical controller/configuration, and final GKE S5 actuator implementation/tests were inspected before execution. The v0.62 and v0.63 Exp09 configurations differ only in the approved AHBN `max_fanout` migration from 4 to 6. The v0.63 score, mode boundary, and exact S5 fanout boundaries match the final GKE reference.

## Frozen Exp09 Design

ER topology; N=100; p={0.04, 0.06, 0.08, 0.10, 0.12}; four clusters; source 0; base delay 1.0 s; jitter 0.2 s; 20 formal runs per cell; strategies Gossip, Cluster, DC-SoC, and AHBN. Expected formal total: 400 runs.

## Canonical/GKE Parity

Canonical score: `z = -d_hat + l_hat + u_hat + c_hat`; alpha=0.30; zero centers; mode is Gossip at weight >=0.50 and Cluster otherwise. Fanout boundaries are exactly <=-0.25 -> 2, (-0.25,0.25) -> 3, [0.25,0.90) -> 4, [0.90,1.50) -> 5, and >=1.50 -> 6.

## Environment

Pinned project and Python are enforced by `scripts/run_stage4_v063.sh`. Smoke uses seed 42 and one run per ER density × strategy cell; formal preserves 20 runs per setting.

## GKE Canonical Source

Live GKE final S5 mapping inspected and matched.

## v0.62 -> v0.63 Migration

Only AHBN actuator range/mapping changed; Exp09 factors and all baselines remain frozen.

## Regression Validation

See Exp07 gate documentation.

## Smoke Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp09_v063_smoke.sh
```

## Smoke Terminal Output

Not run. The timestamped output directory will contain `terminal.log` with stdout, stderr, exit code, and output path.

## Smoke Validation

Pending manual execution and only after Exp07 review.

## Formal Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp09_v063_formal.sh
```

## Formal Terminal Output

Not run.

## Aggregation

The prepared analyzer reports each strategy × ER probability and AHBN trace occupancy.

## Statistical Analysis

Pending manual formal run.

## Scientific Interpretation

Do not assume density raises fanout: increased duplicate pressure can lower `z = -d_hat + l_hat + u_hat + c_hat`.

## Final Status

Prepared; not executed.

## Plotting Correction

Original failure: `ValueError: density/n mismatch`.

Root cause: the plotting validator compared floating-point density values using strict Python set equality. Reading the analysis CSV materialized the 0.06 cell as `0.0599999999999999`; the complete 20-row strategy × density grid and every `n=20` value were otherwise correct. The plotter now converts density and n columns with `pd.to_numeric`, validates the sorted density grid with `numpy.allclose`, and validates n numerically. The same tolerance-aware density check is applied to the five-row AHBN adaptive summary.

Only `scripts/plot_stage4_exp09_v063.py` validation was changed. Exp09 smoke, formal simulation, and statistical analysis were not rerun. Raw formal data and validated summary statistics remain unchanged.

## Corrected Plot Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp09_v063_plots.sh \
    outputs/stage4_exp09_v063-20260901_123157
```

## Corrected Plot Terminal Output

The repository wrapper completed with `FIGURE GENERATION EXIT CODE: 0`. Its complete stdout/stderr and all eight generated paths are preserved above in this document and in `exp09_v063_figure_generation_terminal.log` inside the formal directory. The original failed transcript is also preserved.

## Generated Figures

- `exp09_v063_delivery.png`
- `exp09_v063_delay.png`
- `exp09_v063_duplicates.png`
- `exp09_v063_forwards.png`
- `exp09_v063_ahbn_z_by_density.png`
- `exp09_v063_ahbn_fanout_by_density.png`
- `exp09_v063_ahbn_mode_by_density.png`
- `exp09_v063_realized_mean_degree.png`

## Visual Validation

PASS. All five p values and four treatments are present in the primary plots; Student-t 95% confidence intervals correspond to the machine-readable summary; axes and units are correct; there is no clipping, artificial smoothing, hidden treatment, or reordered condition. AHBN z, fanout, and mode panels match the adaptive summary. The realized-mean-degree panel matches the topology summary.

## Scientific Interpretation

Increasing p strongly improves Gossip delay (6.0299 to 3.2420 s) but amplifies Gossip duplicates (209.0 to 984.1) and forwards (306.6 to 1083.1). Structured and DC-SoC retain zero duplication and nearly constant cost/delay. AHBN delivery remains approximately 0.962–0.970, its delay is non-monotonic but improves modestly at the endpoints (8.0757 to 7.7141 s), and duplicate/forward growth plateaus near p=0.10–0.12. AHBN therefore regulates density-driven amplification far more strongly than Gossip, with a propagation-delay and delivery tradeoff. Scientific classification: EXPECTED PARTIAL.

Endpoint p=0.04 to p=0.12:

| Strategy | Delivery delta | Delay delta (%) | Duplicate delta (%) | Forward delta (%) |
|---|---:|---:|---:|---:|
| Gossip | 0.0000 | -2.7879 s (-46.24%) | +775.1 (+370.86%) | +776.5 (+253.26%) |
| Structured | 0.0000 | -0.0011 s (-0.02%) | 0 (undefined from zero) | +1.4 (+1.43%) |
| DC-SoC | 0.0000 | 0.0000 s (0.00%) | 0 (undefined from zero) | +1.4 (+1.43%) |
| AHBN | +0.0056 | -0.3616 s (-4.48%) | +62.4 (+47.31%) | +64.3 (+28.48%) |

## Formula Interpretation

The observed behavior is consistent with `z = -d_hat + l_hat + u_hat + c_hat`. From p=0.04 to 0.12, mean d_hat rises 0.1538 to 0.2014, mean l_hat rises 0.2389 to 0.2619, mean u_hat rises 0.0493 to 0.0616, and c_hat remains zero. The stronger negative duplicate term outweighs the positive latency/utilization changes, reducing mean z from 0.1344 to 0.1221. Fanout remains almost entirely 3, with fanout 2 changing from 0% to 0.052% and no fanout 4/5/6. Gossip-mode share falls from 95.63% to 91.02% while Cluster share rises from 4.37% to 8.98%. Density affects AHBN indirectly through local observations; it is not a controller input.

## Manuscript Impact

- Density adds propagation paths and generally lowers delay: RETAIN WITH QUALIFICATION; Gossip improves strongly, AHBN improves modestly at endpoints but is non-monotonic, and structured comparators are flat.
- AHBN delay falls from about 7.56 s toward 6.0 s: UPDATE to 8.0757 to 7.7141 s.
- AHBN duplicates peak around 315.5: UPDATE to a peak of 194.45 at p=0.10.
- Duplicates plateau/contract slightly at p=0.12: RETAIN; AHBN changes 194.45 to 194.30 from p=0.10 to 0.12.
- Structured duplicate overhead remains near zero: RETAIN; it is exactly zero in these formal means.
- AHBN balances propagation efficiency and duplicate amplification: RETAIN WITH QUALIFICATION; it strongly limits amplification relative to Gossip but carries higher delay and slightly lower delivery.

## Exp09 Freeze Decision

TECHNICAL VALIDATION: PASS

SCIENTIFIC RESULT: EXPECTED PARTIAL

EXP09 FREEZE: PASS

Exp07(sim): FROZEN

Exp08(sim): FROZEN

Exp09(sim): FROZEN

Stage-4 controlled simulation: COMPLETE

Exp09 smoke and formal simulation were not rerun during plotting correction. Statistical analysis was not rerun. Raw formal data, validated summaries, canonical AHBN, actuator thresholds, Exp07, and Exp08 remain unchanged. Figures were generated through the reproducible repository wrapper. No experiment beyond Exp09 was run.
Stage 4 exp09 ControlSim v0.63 smoke
Command: cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Command: bash scripts/run_stage4_exp09_v063_smoke.sh
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123147/exp09_dense_topology_smoke.yaml
Output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123147
EXP09 DESIGN
project root: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
topology: er
N: 100
density levels: [0.04, 0.06, 0.08, 0.1, 0.12]
strategies: ['gossip', 'cluster', 'dcsoc', 'ahbn']
runs per cell: 1
expected run count: 20
Saved outputs/csv/exp09_results_20260901_123150.csv
Saved outputs/csv/exp09_adaptive_trace_20260901_123150.csv
EXP09 DATASET / TOPOLOGY AUDIT: PASS
expected runs: 20
actual runs: 20
cells: 20; replicates per cell: 1
AHBN trace rows: 1352
controller violations: score=0, fanout=0, mode=0
realized topology by configured p:
 density_p  runs  mean_nodes  mean_edge_count  mean_degree
      0.04     1        99.0            220.0     4.444444
      0.06     1       100.0            310.0     6.200000
      0.08     1       100.0            412.0     8.240000
      0.10     1       100.0            517.0    10.340000
      0.12     1       100.0            604.0    12.080000
TECHNICAL VALIDATION: PASS
EXIT CODE: 0
OUTPUT DIRECTORY: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123147
Stage 4 exp09 ControlSim v0.63 formal
Command: cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Command: bash scripts/run_stage4_exp09_v063_formal.sh
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/configs/exp09_dense_topology.yaml
Output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157
EXP09 DESIGN
project root: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
topology: er
N: 100
density levels: [0.04, 0.06, 0.08, 0.1, 0.12]
strategies: ['gossip', 'cluster', 'dcsoc', 'ahbn']
runs per cell: 20
expected run count: 400
Saved outputs/csv/exp09_results_20260901_123202.csv
Saved outputs/csv/exp09_adaptive_trace_20260901_123202.csv
EXP09 DATASET / TOPOLOGY AUDIT: PASS
expected runs: 400
actual runs: 400
cells: 20; replicates per cell: 20
AHBN trace rows: 27444
controller violations: score=0, fanout=0, mode=0
realized topology by configured p:
 density_p  runs  mean_nodes  mean_edge_count  mean_degree
      0.04    20       98.60           202.10     4.098975
      0.06    20       99.85           303.10     6.071364
      0.08    20      100.00           399.95     7.999000
      0.10    20      100.00           495.40     9.908000
      0.12    20      100.00           591.05    11.821000
TECHNICAL VALIDATION: PASS
EXIT CODE: 0
OUTPUT DIRECTORY: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157
Analysis command: bash scripts/run_stage4_exp09_v063_analysis.sh outputs/stage4_exp09_v063-20260901_123157
Selected dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157
selected formal output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157
expected rows: 400
actual rows: 400
conditions: 20; n=20 each
DATASET AUDIT: PASS
DENSITY EVIDENCE: PASS
AHBN TRACE VALIDATION: PASS

PRIMARY RESULTS
gossip  p=0.04 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=6.029857 [5.547599, 6.512116]  duplicates=209.000000 [199.054513, 218.945487]  total_forwards=306.600000 [296.511345, 316.688655]
gossip  p=0.06 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.442501 [4.246074, 4.638928]  duplicates=408.500000 [398.622556, 418.377444]  total_forwards=507.350000 [497.518443, 517.181557]
gossip  p=0.08 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.871893 [3.670704, 4.073082]  duplicates=601.900000 [587.359487, 616.440513]  total_forwards=700.900000 [686.359487, 715.440513]
gossip  p=0.10 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.323639 [3.293090, 3.354189]  duplicates=792.800000 [775.114517, 810.485483]  total_forwards=891.800000 [874.114517, 909.485483]
gossip  p=0.12 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.241952 [3.211246, 3.272659]  duplicates=984.100000 [965.314323, 1002.885677]  total_forwards=1083.100000 [1064.314323, 1101.885677]
cluster p=0.04 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.520881 [4.471440, 4.570322]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=97.600000 [97.159914, 98.040086]
cluster p=0.06 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.519793 [4.467753, 4.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=98.850000 [98.678544, 99.021456]
cluster p=0.08 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.519793 [4.467753, 4.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
cluster p=0.10 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.519793 [4.467753, 4.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
cluster p=0.12 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.519793 [4.467753, 4.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   p=0.04 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.197891 [1.197079, 1.198703]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=97.600000 [97.159914, 98.040086]
dcsoc   p=0.06 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.197891 [1.197079, 1.198703]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=98.850000 [98.678544, 99.021456]
dcsoc   p=0.08 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.197891 [1.197079, 1.198703]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   p=0.10 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.197891 [1.197079, 1.198703]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   p=0.12 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.197891 [1.197079, 1.198703]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
ahbn    p=0.04 n=20  delivery_ratio=0.961943 [0.952031, 0.971856]  propagation_delay=8.075682 [7.738070, 8.413295]  duplicates=131.900000 [128.729769, 135.070231]  total_forwards=225.750000 [222.540108, 228.959892]
ahbn    p=0.06 n=20  delivery_ratio=0.966949 [0.958887, 0.975012]  propagation_delay=7.814579 [7.435467, 8.193692]  duplicates=179.200000 [176.660981, 181.739019]  total_forwards=274.750000 [271.725008, 277.774992]
ahbn    p=0.08 n=20  delivery_ratio=0.965500 [0.957541, 0.973459]  propagation_delay=8.257322 [7.750557, 8.764088]  duplicates=190.700000 [189.001654, 192.398346]  total_forwards=286.250000 [283.878730, 288.621270]
ahbn    p=0.10 n=20  delivery_ratio=0.969500 [0.961988, 0.977012]  propagation_delay=7.724285 [7.312389, 8.136182]  duplicates=194.450000 [192.811246, 196.088754]  total_forwards=290.400000 [288.026179, 292.773821]
ahbn    p=0.12 n=20  delivery_ratio=0.967500 [0.960398, 0.974602]  propagation_delay=7.714127 [7.322803, 8.105451]  duplicates=194.300000 [192.924157, 195.675843]  total_forwards=290.050000 [287.970930, 292.129070]

AHBN ADAPTIVE BY DENSITY
 density_p  trace_rows     z_min   z_mean    z_max  d_hat_min  d_hat_mean  d_hat_max  l_hat_min  l_hat_mean  l_hat_max  u_hat_min  u_hat_mean  u_hat_max  c_hat_min  c_hat_mean  c_hat_max  fanout_2_count  fanout_2_proportion  fanout_3_count  fanout_3_proportion  fanout_4_count  fanout_4_proportion  fanout_5_count  fanout_5_proportion  fanout_6_count  fanout_6_proportion  gossip_mode_count  gossip_mode_proportion  cluster_mode_count  cluster_mode_proportion  mean_edge_count  mean_degree  fanout_violations  mode_violations
      0.04        4535 -0.197267 0.134388 0.190541        0.0    0.153786   0.752604        0.0    0.238882   0.479842        0.0    0.049292    0.10925        0.0         0.0        0.0               0             0.000000            4535             1.000000               0                  0.0               0                  0.0               0                  0.0               4337                0.956340                 198                 0.043660           202.10     4.098975                  0                0
      0.06        5515 -0.238435 0.125880 0.190801        0.0    0.189894   0.793489        0.0    0.256803   0.485239        0.0    0.058971    0.10925        0.0         0.0        0.0               0             0.000000            5515             1.000000               0                  0.0               0                  0.0               0                  0.0               5095                0.923844                 420                 0.076156           303.10     6.071364                  0                0
      0.08        5745 -0.328984 0.121850 0.190602        0.0    0.201099   0.870376        0.0    0.261781   0.491200        0.0    0.061168    0.10925        0.0         0.0        0.0               4             0.000696            5741             0.999304               0                  0.0               0                  0.0               0                  0.0               5230                0.910357                 515                 0.089643           399.95     7.999000                  0                0
      0.10        5828 -0.267776 0.121400 0.190724        0.0    0.202669   0.825443        0.0    0.262544   0.490564        0.0    0.061525    0.10925        0.0         0.0        0.0               1             0.000172            5827             0.999828               0                  0.0               0                  0.0               0                  0.0               5279                0.905800                 549                 0.094200           495.40     9.908000                  0                0
      0.12        5821 -0.296494 0.122078 0.190677        0.0    0.201403   0.850537        0.0    0.261912   0.493435        0.0    0.061568    0.10925        0.0         0.0        0.0               3             0.000515            5818             0.999485               0                  0.0               0                  0.0               0                  0.0               5298                0.910153                 523                 0.089847           591.05    11.821000                  0                0

P=0.04 TO P=0.12
strategy  delivery_ratio_p004  delivery_ratio_p012  delivery_ratio_delta  delivery_ratio_change_pct  propagation_delay_p004  propagation_delay_p012  propagation_delay_delta  propagation_delay_change_pct  duplicates_p004  duplicates_p012  duplicates_delta  duplicates_change_pct  total_forwards_p004  total_forwards_p012  total_forwards_delta  total_forwards_change_pct
  gossip             1.000000               1.0000              0.000000                   0.000000                6.029857                3.241952                -2.787905                    -46.235007            209.0            984.1             775.1             370.861244               306.60              1083.10                 776.5                 253.261579
 cluster             1.000000               1.0000              0.000000                   0.000000                4.520881                4.519793                -0.001088                     -0.024057              0.0              0.0               0.0                    NaN                97.60                99.00                   1.4                   1.434426
   dcsoc             1.000000               1.0000              0.000000                   0.000000                1.197891                1.197891                 0.000000                      0.000000              0.0              0.0               0.0                    NaN                97.60                99.00                   1.4                   1.434426
    ahbn             0.961943               0.9675              0.005557                   0.577658                8.075682                7.714127                -0.361556                     -4.477092            131.9            194.3              62.4              47.308567               225.75               290.05                  64.3                  28.482835
TECHNICAL VALIDATION: PASS
ANALYSIS EXIT CODE: 0
EXP09 FORMAL FIGURE GENERATION
dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157
python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/plot_stage4_exp09_v063.py", line 102, in <module>
    if __name__ == "__main__": main()
                               ~~~~^^
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/plot_stage4_exp09_v063.py", line 92, in main
    require(set(summary["density_p"].astype(float)) == set(P_VALUES) and (summary["n"] == 20).all(), "density/n mismatch")
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/plot_stage4_exp09_v063.py", line 27, in require
    if not ok: raise ValueError(message)
               ^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: density/n mismatch
FIGURE GENERATION EXIT CODE: 1
SUMMARY DIAGNOSTIC
columns: ['strategy', 'density_p', 'n', 'delivery_ratio_mean', 'delivery_ratio_sd', 'delivery_ratio_ci95_low', 'delivery_ratio_ci95_high', 'propagation_delay_mean', 'propagation_delay_sd', 'propagation_delay_ci95_low', 'propagation_delay_ci95_high', 'duplicates_mean', 'duplicates_sd', 'duplicates_ci95_low', 'duplicates_ci95_high', 'total_forwards_mean', 'total_forwards_sd', 'total_forwards_ci95_low', 'total_forwards_ci95_high']
row count: 20
strategies: ['ahbn', 'cluster', 'dcsoc', 'gossip']
density repr: ['np.float64(0.04)', 'np.float64(0.0599999999999999)', 'np.float64(0.08)', 'np.float64(0.1)', 'np.float64(0.12)']
n dtype: int64
n values:
strategy  density_p
ahbn      0.04         20
          0.06         20
          0.08         20
          0.10         20
          0.12         20
cluster   0.04         20
          0.06         20
          0.08         20
          0.10         20
          0.12         20
dcsoc     0.04         20
          0.06         20
          0.08         20
          0.10         20
          0.12         20
gossip    0.04         20
          0.06         20
          0.08         20
          0.10         20
          0.12         20
all n numeric 20: True
CORRECTED STATIC PLOT CHECK: PASS
normalized summary densities: [np.float64(0.04), np.float64(0.0599999999999999), np.float64(0.08), np.float64(0.1), np.float64(0.12)]
normalized adaptive densities: [np.float64(0.04), np.float64(0.0599999999999999), np.float64(0.08), np.float64(0.1), np.float64(0.12)]
strategy-density cells: 20
all n == 20: True
EXP09 FORMAL FIGURE GENERATION
dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157
python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_delivery.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_delay.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_duplicates.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_forwards.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_ahbn_z_by_density.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_ahbn_fanout_by_density.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_ahbn_mode_by_density.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp09_v063-20260901_123157/exp09_v063_realized_mean_degree.png
FIGURE GENERATION EXIT CODE: 0
