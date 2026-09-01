# Stage 4 — Exp08 ControlSim v0.63 Rerun

## Reference Audit

Inspected v0.62 `configs/exp08_ch_bottleneck.yaml`, `run_batch.py`, `scripts/validate_exp08_final.py`, `scripts/aggregate_exp08_e7.py`, `scripts/plot_exp08_e8.py`, and the Exp08 section of `scripts/plot_results.py`. Inspected v0.63 `ahbn/control.py`, `run_one.py`, `run_batch.py`, `configs/exp08_ch_bottleneck.yaml`, `scripts/validate_canonical_ahbn_frozen.py`, the shared runner, and analyzer. Inspected the live GKE final actuator policy and boundary tests.

Canonical equation confirmed: `z = -d_hat + l_hat + u_hat + c_hat`. Bounds confirmed: 2–6. Boundaries confirmed: `z <= -0.25 -> 2`, `<0.25 -> 3`, `<0.90 -> 4`, `<1.50 -> 5`, otherwise 6. No stale active `max_fanout=4` was found in the v0.63 Exp08 path.

## Final Exp08 Design

- BA topology, N=100, m=3
- Four clusters; configured source node 0
- Base delay 1.0 s; jitter 0.2 s
- Overload factors `[1.0, 1.5, 2.0, 3.0]`
- Strategies: Gossip, Cluster/Structured, DC-SoC, AHBN
- Seeds 42–61; 20 runs per strategy × overload cell
- Expected formal count derived from config: 4 × 4 × 20 = 320

The v0.62 and v0.63 designs match exactly except for the already-approved AHBN maximum migration from 4 to 6. No 2.5 overload treatment exists.

## Environment

Pinned project and Python are enforced by `scripts/run_stage4_v063.sh`. Smoke uses seed 42 and one run per overload × strategy cell; formal preserves 20 runs per setting.

## GKE Canonical Source

Live GKE final S5 mapping inspected and matched.

## v0.62 -> v0.63 Migration

Only AHBN actuator range/mapping changed; Exp08 factors and all baselines remain frozen.

## Regression Validation

See Exp07 gate documentation.

## Smoke Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp08_v063_smoke.sh
```

## Smoke Terminal Output

Not run. The timestamped output directory will contain `terminal.log` with stdout, stderr, exit code, and output path.

## Smoke Validation

Pending manual execution and only after Exp07 review.

## Formal Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp08_v063_formal.sh
```

## Formal Terminal Output

Not run.

## Aggregation

The prepared analyzer reports each strategy × overload condition and AHBN trace occupancy.

## Statistical Analysis

Pending manual formal run.

## Scientific Interpretation

Preserve technically valid unexpected results; do not force fanout 5/6 or retune thresholds.

## Final Status

Prepared; not executed.

## Analysis Correction

The first analysis attempt failed because `group.mode` resolved to pandas `DataFrame.mode` rather than the trace column named `mode`. The active count and proportion expressions therefore compared a method with a string and produced a scalar boolean. The only correction was:

```text
old: group.mode
new: group["mode"]
```

Both affected adjacent expressions were corrected. A non-writing regression check confirmed that the mode column exists and all four overload groups consume the existing 19,363 trace rows. Raw formal data was unchanged; smoke and formal Exp08 were not rerun.

## Corrected Analysis Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp08_v063_analysis.sh \
    outputs/stage4_exp08_v063-20260901_111532
```

## Corrected Analysis Terminal Output

The corrected command exited 0. Dataset audit, evidence/overload contract, and AHBN trace validation all passed. The complete transcript, including the preserved earlier failure, is in `outputs/stage4_exp08_v063-20260901_111532/exp08_v063_analysis_terminal.log`.

## Formal Aggregated Results

Values are mean ± Student-t 95% CI half-width; n=20 per cell.

| Strategy | Overload | Delivery | Delay | Duplicates | Forwards |
|---|---:|---:|---:|---:|---:|
| Gossip | 1.0 | 1.0000 ± 0 | 3.2792 ± 0.0277 | 384.00 ± 0 | 483.00 ± 0 |
| Gossip | 1.5 | 1.0000 ± 0 | 3.2792 ± 0.0277 | 384.00 ± 0 | 483.00 ± 0 |
| Gossip | 2.0 | 1.0000 ± 0 | 3.2792 ± 0.0277 | 384.00 ± 0 | 483.00 ± 0 |
| Gossip | 3.0 | 1.0000 ± 0 | 3.2792 ± 0.0277 | 384.00 ± 0 | 483.00 ± 0 |
| Structured | 1.0 | 1.0000 ± 0 | 4.5198 ± 0.0520 | 0 ± 0 | 99.00 ± 0 |
| Structured | 1.5 | 1.0000 ± 0 | 6.0198 ± 0.0520 | 0 ± 0 | 99.00 ± 0 |
| Structured | 2.0 | 1.0000 ± 0 | 7.5198 ± 0.0520 | 0 ± 0 | 99.00 ± 0 |
| Structured | 3.0 | 1.0000 ± 0 | 10.5198 ± 0.0520 | 0 ± 0 | 99.00 ± 0 |
| DC-SoC | 1.0 | 1.0000 ± 0 | 1.1983 ± 0.0007 | 0 ± 0 | 99.00 ± 0 |
| DC-SoC | 1.5 | 1.0000 ± 0 | 1.6983 ± 0.0007 | 0 ± 0 | 99.00 ± 0 |
| DC-SoC | 2.0 | 1.0000 ± 0 | 2.1983 ± 0.0007 | 0 ± 0 | 99.00 ± 0 |
| DC-SoC | 3.0 | 1.0000 ± 0 | 3.1983 ± 0.0007 | 0 ± 0 | 99.00 ± 0 |
| AHBN | 1.0 | 0.9100 ± 0.0157 | 9.2524 ± 0.5876 | 153.15 ± 3.43 | 243.15 ± 4.86 |
| AHBN | 1.5 | 0.8990 ± 0.0190 | 9.6795 ± 0.7217 | 151.85 ± 4.12 | 240.75 ± 5.94 |
| AHBN | 2.0 | 0.9035 ± 0.0148 | 9.0028 ± 0.5207 | 151.85 ± 3.63 | 241.20 ± 5.05 |
| AHBN | 3.0 | 0.8935 ± 0.0223 | 10.0333 ± 0.7154 | 150.70 ± 4.56 | 239.05 ± 6.72 |

Machine-readable outputs: `exp08_v063_summary.csv`, `exp08_v063_ahbn_adaptive_summary.csv`, and `exp08_v063_baseline_to_highest.csv` in the formal output directory.

## AHBN Adaptive Behavior

| Overload | z min/mean/max | f2 | f3 | f4 | f5/f6 | Gossip mode | Cluster mode |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0 | -0.4207 / 0.0733 / 0.1907 | 310 (6.35%) | 4573 (93.65%) | 0 | 0 | 78.07% | 21.93% |
| 1.5 | -0.4172 / 0.0834 / 0.2328 | 212 (4.38%) | 4623 (95.62%) | 0 | 0 | 79.96% | 20.04% |
| 2.0 | -0.4236 / 0.0902 / 0.2633 | 160 (3.30%) | 4632 (95.62%) | 52 (1.07%) | 0 | 81.23% | 18.77% |
| 3.0 | -0.4153 / 0.0957 / 0.3036 | 157 (3.27%) | 4546 (94.69%) | 98 (2.04%) | 0 | 82.02% | 17.98% |

At overload 1.0→3.0, mean d_hat was 0.2554→0.2567, l_hat 0.2752→0.2987, u_hat 0.0535→0.0536, and c_hat remained 0. Fanout and mode violations were zero at every level.

## Structured Bottleneck Effect

Structured delay increased from 4.5198 to 10.5198 seconds: +6.0000 seconds, or +132.75%. Delivery remained 1.0, duplicates remained 0, and forwards remained 99. This is a strong isolated CH-path delay sensitivity.

## AHBN Bottleneck Effect

AHBN delay increased from 9.2524 to 10.0333 seconds: +0.7810 seconds, or +8.44%. Delivery changed by -0.0165, duplicates by -2.45, and forwards by -4.10. AHBN therefore showed much less delay amplification than Structured, with a small delivery decline rather than extra traffic cost.

Gossip was invariant across overload. DC-SoC delay rose from 1.1983 to 3.1983 seconds (+166.91%) while retaining full delivery and fixed traffic, but remained the lowest-delay treatment at every factor.

## Formula Interpretation

The observed response is coherent with `z = -d_hat + l_hat + u_hat + c_hat`. Mean latency pressure increased with overload and pushed mean z upward from 0.0733 to 0.0957. Mean utilization pressure was nearly constant and churn pressure was zero. Duplicate pressure remained substantial and nearly flat, providing negative feedback that damped escalation. Consequently, Gossip-mode share rose and fanout-4 appeared at factors 2.0/3.0, but most decisions remained fanout 3 and no fanout 5/6 was required.

## Figure Generation Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp08_v063_plots.sh \
    outputs/stage4_exp08_v063-20260901_111532
```

The command exited 0. Complete stdout/stderr is recorded in `outputs/stage4_exp08_v063-20260901_111532/exp08_v063_figure_generation_terminal.log`.

## Generated Figures

- `exp08_v063_delay.png`
- `exp08_v063_delivery.png`
- `exp08_v063_duplicates.png`
- `exp08_v063_forwards.png`
- `exp08_v063_ahbn_z_by_overload.png`
- `exp08_v063_ahbn_fanout_by_overload.png`
- `exp08_v063_ahbn_mode_by_overload.png`

All figures are non-empty, include all four overload levels, use all four frozen strategies where applicable, show Student-t 95% CIs, and agree with the summary CSVs. Axes, units, treatments, and legends are readable; no smoothing or fabricated values were introduced.

## Scientific Interpretation

**EXPECTED STRONG.** Increasing CH overload exposed Structured dissemination's concentrated-path sensitivity: its delay amplification was +132.75%, versus +8.44% for AHBN. AHBN adapted coherently through a modest upward score/mode/fanout shift rather than permanent high fanout. Its absolute baseline delay and delivery remain worse than the static baselines, so the result supports bottleneck-amplification mitigation, not universal performance dominance. DC-SoC retained the best absolute delay despite a large relative increase from a low baseline.

## Manuscript Impact

| Old claim | New v0.63 result | Status |
|---|---|---|
| Structured delay 4.50→10.52 s, about +134% | 4.5198→10.5198 s, +132.75% | RETAIN |
| AHBN delay 6.02→6.54 s, about +8.6% | 9.2524→10.0333 s, +8.44% | UPDATE absolute values; RETAIN relative-stability mechanism |
| Delivery remains stable | Static baselines remain 1.0; AHBN changes 0.9100→0.8935 | UPDATE with AHBN qualification |
| AHBN limits structural bottleneck amplification | +8.44% versus Structured +132.75% | RETAIN |

## Exp08 Freeze Decision

```text
TECHNICAL VALIDATION: PASS
SCIENTIFIC RESULT: EXPECTED STRONG
EXP08 FREEZE: PASS
```

The formal dataset, evidence, canonical traces, corrected reproducible analysis, and figures are valid with no unresolved technical anomaly. Exp08(sim) is ready to freeze. Do not run Exp09 until user review.
Stage 4 exp08 ControlSim v0.63 smoke
Command: cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Command: bash scripts/run_stage4_exp08_v063_smoke.sh
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111518/exp08_ch_bottleneck_smoke.yaml
Output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111518
Saved outputs/csv/exp08_results_20260901_111521.csv
Saved outputs/csv/exp08_execution_evidence_20260901_111521.csv
Saved outputs/csv/exp08_ahbn_adaptive_trace_20260901_111521.csv
Saved outputs/csv/exp08_s3_manifest.json
Exp08 evidence: PASS (16 rows; paired topologies; DC-SoC Master/Core targets)
{
  "validation": "PASS",
  "results": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111518/outputs/csv/exp08_results_20260901_111521.csv",
  "trace": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111518/outputs/csv/exp08_ahbn_adaptive_trace_20260901_111521.csv",
  "trace_rows": 958,
  "z_min": -0.4206938926514327,
  "z_mean": 0.07530315181697401,
  "z_max": 0.3017116388275755,
  "d_hat_min": 0.0,
  "l_hat_min": 0.0,
  "u_hat_min": 0.0,
  "c_hat_min": 0.0,
  "d_hat_max": 0.943693553630658,
  "l_hat_max": 0.7398605753292031,
  "u_hat_max": 0.1092499999999999,
  "c_hat_max": 0.0,
  "fanout_2_count": 55,
  "fanout_3_count": 894,
  "fanout_4_count": 9,
  "fanout_5_count": 0,
  "fanout_6_count": 0,
  "fanout_2_proportion": 0.05741127348643006,
  "fanout_3_proportion": 0.9331941544885177,
  "fanout_4_proportion": 0.009394572025052192,
  "fanout_5_proportion": 0.0,
  "fanout_6_proportion": 0.0,
  "gossip_mode_count": 751,
  "gossip_mode_proportion": 0.7839248434237995,
  "cluster_mode_count": 207,
  "cluster_mode_proportion": 0.2160751565762004,
  "fanout_violations": 0,
  "mode_violations": 0
}
TECHNICAL VALIDATION: PASS
EXIT CODE: 0
OUTPUT DIRECTORY: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111518
Stage 4 exp08 ControlSim v0.63 formal
Command: cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Command: bash scripts/run_stage4_exp08_v063_formal.sh
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/configs/exp08_ch_bottleneck.yaml
Output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532
EXP08 FORMAL RUN
project root: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
treatments: ['gossip', 'cluster', 'dcsoc', 'ahbn']
overload factors: [1.0, 1.5, 2.0, 3.0]
runs per cell: 20
expected run count: 320
Saved outputs/csv/exp08_results_20260901_111536.csv
Saved outputs/csv/exp08_execution_evidence_20260901_111536.csv
Saved outputs/csv/exp08_ahbn_adaptive_trace_20260901_111536.csv
Saved outputs/csv/exp08_s3_manifest.json
Exp08 evidence: PASS (320 rows; paired topologies; DC-SoC Master/Core targets)
{
  "validation": "PASS",
  "results": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/outputs/csv/exp08_results_20260901_111536.csv",
  "trace": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/outputs/csv/exp08_ahbn_adaptive_trace_20260901_111536.csv",
  "trace_rows": 19363,
  "z_min": -0.4236084208363476,
  "z_mean": 0.08561253524162588,
  "z_max": 0.3036062494685149,
  "d_hat_min": 0.0,
  "l_hat_min": 0.0,
  "u_hat_min": 0.0,
  "c_hat_min": 0.0,
  "d_hat_max": 0.9523746359308888,
  "l_hat_max": 0.7398605753292031,
  "u_hat_max": 0.1092499999999999,
  "c_hat_max": 0.0,
  "fanout_2_count": 839,
  "fanout_3_count": 18374,
  "fanout_4_count": 150,
  "fanout_5_count": 0,
  "fanout_6_count": 0,
  "fanout_2_proportion": 0.04333006249031658,
  "fanout_3_proportion": 0.9489232040489594,
  "fanout_4_proportion": 0.007746733460724062,
  "fanout_5_proportion": 0.0,
  "fanout_6_proportion": 0.0,
  "gossip_mode_count": 15551,
  "gossip_mode_proportion": 0.8031296803181325,
  "cluster_mode_count": 3812,
  "cluster_mode_proportion": 0.19687031968186747,
  "fanout_violations": 0,
  "mode_violations": 0
}
TECHNICAL VALIDATION: PASS
EXIT CODE: 0
OUTPUT DIRECTORY: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532
Analysis command: bash scripts/run_stage4_exp08_v063_analysis.sh outputs/stage4_exp08_v063-20260901_111532
Selected dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/analyze_stage4_exp08_v063.py", line 185, in <module>
    if __name__ == "__main__": main()
                               ~~~~^^
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/analyze_stage4_exp08_v063.py", line 163, in main
    summary, adaptive = aggregate(results), adaptive_summary(trace); trend = changes(summary)
                                            ~~~~~~~~~~~~~~~~^^^^^^^
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/analyze_stage4_exp08_v063.py", line 133, in adaptive_summary
    row[f"{mode}_mode_count"] = int((group.mode == mode).sum())
                                    ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'bool' object has no attribute 'sum'
ANALYSIS EXIT CODE: 1
Analysis command: bash scripts/run_stage4_exp08_v063_analysis.sh outputs/stage4_exp08_v063-20260901_111532
Selected dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532
selected formal output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532
results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/outputs/csv/exp08_results_20260901_111536.csv
evidence: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/outputs/csv/exp08_execution_evidence_20260901_111536.csv
trace: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/outputs/csv/exp08_ahbn_adaptive_trace_20260901_111536.csv
expected rows: 320
actual rows: 320
conditions: 16; n=20 each
DATASET AUDIT: PASS
EVIDENCE/OVERLOAD CONTRACT: PASS
AHBN TRACE VALIDATION: PASS

PRIMARY RESULTS
gossip  overload=1.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.279219 [3.251534, 3.306905]  duplicates=384.000000 [384.000000, 384.000000]  total_forwards=483.000000 [483.000000, 483.000000]
gossip  overload=1.5 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.279219 [3.251534, 3.306905]  duplicates=384.000000 [384.000000, 384.000000]  total_forwards=483.000000 [483.000000, 483.000000]
gossip  overload=2.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.279219 [3.251534, 3.306905]  duplicates=384.000000 [384.000000, 384.000000]  total_forwards=483.000000 [483.000000, 483.000000]
gossip  overload=3.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.279219 [3.251534, 3.306905]  duplicates=384.000000 [384.000000, 384.000000]  total_forwards=483.000000 [483.000000, 483.000000]
cluster overload=1.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=4.519793 [4.467753, 4.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
cluster overload=1.5 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=6.019793 [5.967753, 6.071834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
cluster overload=2.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=7.519793 [7.467753, 7.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
cluster overload=3.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=10.519793 [10.467753, 10.571834]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   overload=1.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.198268 [1.197598, 1.198937]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   overload=1.5 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=1.698268 [1.697598, 1.698937]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   overload=2.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=2.198268 [2.197598, 2.198937]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
dcsoc   overload=3.0 n=20  delivery_ratio=1.000000 [1.000000, 1.000000]  propagation_delay=3.198268 [3.197598, 3.198937]  duplicates=0.000000 [0.000000, 0.000000]  total_forwards=99.000000 [99.000000, 99.000000]
ahbn    overload=1.0 n=20  delivery_ratio=0.910000 [0.894293, 0.925707]  propagation_delay=9.252378 [8.664768, 9.839989]  duplicates=153.150000 [149.716598, 156.583402]  total_forwards=243.150000 [238.292715, 248.007285]
ahbn    overload=1.5 n=20  delivery_ratio=0.899000 [0.879980, 0.918020]  propagation_delay=9.679536 [8.957877, 10.401194]  duplicates=151.850000 [147.732607, 155.967393]  total_forwards=240.750000 [234.805015, 246.694985]
ahbn    overload=2.0 n=20  delivery_ratio=0.903500 [0.888679, 0.918321]  propagation_delay=9.002786 [8.482134, 9.523438]  duplicates=151.850000 [148.217549, 155.482451]  total_forwards=241.200000 [236.151104, 246.248896]
ahbn    overload=3.0 n=20  delivery_ratio=0.893500 [0.871221, 0.915779]  propagation_delay=10.033341 [9.317894, 10.748788]  duplicates=150.700000 [146.141899, 155.258101]  total_forwards=239.050000 [232.325054, 245.774946]

AHBN ADAPTIVE BY OVERLOAD
 ch_overload_factor  trace_rows     z_min   z_mean    z_max  d_hat_min  d_hat_mean  d_hat_max  l_hat_min  l_hat_mean  l_hat_max  u_hat_min  u_hat_mean  u_hat_max  c_hat_min  c_hat_mean  c_hat_max  fanout_2_count  fanout_2_proportion  fanout_3_count  fanout_3_proportion  fanout_4_count  fanout_4_proportion  fanout_5_count  fanout_5_proportion  fanout_6_count  fanout_6_proportion  gossip_mode_count  gossip_mode_proportion  cluster_mode_count  cluster_mode_proportion  fanout_violations  mode_violations
                1.0        4883 -0.420694 0.073345 0.190677        0.0    0.255413   0.946949        0.0    0.275242   0.506467        0.0    0.053516    0.10925        0.0         0.0        0.0             310             0.063486            4573             0.936514               0             0.000000               0                  0.0               0                  0.0               3812                0.780668                1071                 0.219332                  0                0
                1.5        4835 -0.417224 0.083393 0.232808        0.0    0.253649   0.943694        0.0    0.283253   0.600075        0.0    0.053789    0.10925        0.0         0.0        0.0             212             0.043847            4623             0.956153               0             0.000000               0                  0.0               0                  0.0               3866                0.799586                 969                 0.200414                  0                0
                2.0        4844 -0.423608 0.090226 0.263305        0.0    0.250738   0.952375        0.0    0.287167   0.657281        0.0    0.053797    0.10925        0.0         0.0        0.0             160             0.033031            4632             0.956235              52             0.010735               0                  0.0               0                  0.0               3935                0.812345                 909                 0.187655                  0                0
                3.0        4801 -0.415258 0.095670 0.303606        0.0    0.256690   0.946949        0.0    0.298715   0.739861        0.0    0.053645    0.10925        0.0         0.0        0.0             157             0.032702            4546             0.946886              98             0.020412               0                  0.0               0                  0.0               3938                0.820246                 863                 0.179754                  0                0

BASELINE TO HIGHEST OVERLOAD
strategy  delivery_ratio_baseline  delivery_ratio_highest  delivery_ratio_delta  delivery_ratio_change_pct  propagation_delay_baseline  propagation_delay_highest  propagation_delay_delta  propagation_delay_change_pct  duplicates_baseline  duplicates_highest  duplicates_delta  duplicates_change_pct  total_forwards_baseline  total_forwards_highest  total_forwards_delta  total_forwards_change_pct
  gossip                     1.00                  1.0000                0.0000                   0.000000                    3.279219                   3.279219                 0.000000                      0.000000               384.00               384.0              0.00               0.000000                   483.00                  483.00                   0.0                   0.000000
 cluster                     1.00                  1.0000                0.0000                   0.000000                    4.519793                  10.519793                 6.000000                    132.749432                 0.00                 0.0              0.00                    NaN                    99.00                   99.00                   0.0                   0.000000
   dcsoc                     1.00                  1.0000                0.0000                   0.000000                    1.198268                   3.198268                 2.000000                    166.907637                 0.00                 0.0              0.00                    NaN                    99.00                   99.00                   0.0                   0.000000
    ahbn                     0.91                  0.8935               -0.0165                  -1.813187                    9.252378                  10.033341                 0.780963                      8.440669               153.15               150.7             -2.45              -1.599739                   243.15                  239.05                  -4.1                  -1.686202
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_summary.csv
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_ahbn_adaptive_summary.csv
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_baseline_to_highest.csv
TECHNICAL VALIDATION: PASS
ANALYSIS EXIT CODE: 0
EXP08 FORMAL FIGURE GENERATION
dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532
python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_delay.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_delivery.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_duplicates.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_forwards.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_ahbn_z_by_overload.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_ahbn_fanout_by_overload.png
Generated figure: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp08_v063-20260901_111532/exp08_v063_ahbn_mode_by_overload.png
FIGURE GENERATION EXIT CODE: 0
