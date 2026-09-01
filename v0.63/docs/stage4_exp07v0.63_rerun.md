# Stage 4 — Exp07 ControlSim v0.63 Rerun

## Environment

- Project: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63`
- Python: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Smoke: seed 42, one run per relevant setting.
- Formal: frozen 120-run design (Gossip 5 × 20; AHBN adaptive × 20).

## GKE Canonical Source

The live `AHBN_GKEProj/ahbn2_gke` final S5 policy, runtime, tests, runner, analyzer, and report were inspected. The canonical actuator is `z <= -0.25 -> 2`, `z < 0.25 -> 3`, `z < 0.90 -> 4`, `z < 1.50 -> 5`, otherwise `6`.

## v0.62 -> v0.63 Migration

Only the canonical AHBN maximum and final score-to-fanout mapping changed. The controller equation, observations, EWMA, normalization, sigmoid, mode rule, topology, baselines, and Exp07 design remain frozen.

## Regression Validation

Record the validator commands and output here before the smoke run.

## Regression Gate Repair

Original command:

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_stage4_exp07_execution.py
```

Original failure:

```text
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/scripts/validate_stage4_exp07_execution.py", line 5, in <module>
    from ahbn.config import load_yaml_config
ModuleNotFoundError: No module named 'ahbn'
```

Cause: direct execution placed `scripts/`, not the project root, on `sys.path`. Modified `scripts/validate_stage4_exp07_execution.py` to derive `PROJECT_ROOT = Path(__file__).resolve().parents[1]` and insert it into `sys.path` before project-local imports, matching the established canonical-validator bootstrap.

Canonical validator rerun:

```text
PASS: frozen canonical AHBN defaults, signs, mode, and final GKE actuator boundaries
```

Exp07 execution validator rerun:

```text
Gossip fixed-fanout sweep: [2, 3, 4, 5, 6]
Gossip scheduled runs    : 100
AHBN scheduled runs      : 20
AHBN receives sweep value: NO
AHBN min_fanout          : 2
AHBN max_fanout          : 6
AHBN default_fanout      : 3
AHBN result fanout       : None
Expected total runs      : 120
EXP07 EXECUTION VALIDATION: PASS
```

```text
CANONICAL REGRESSION: PASS
EXP07 EXECUTION REGRESSION: PASS
EXP07 SMOKE: NOT RUN
```

## Smoke Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp07_v063_smoke.sh
```

## Smoke Terminal Output

The complete stdout/stderr transcript and real exit code are written to the timestamped output directory's `terminal.log`. Paste or reference that transcript here after manual execution.

## Smoke Validation

Pending manual execution. `analyze_stage4_v063.py` validates every AHBN trace row and writes `ahbn_trace_summary.json`.

## Formal Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp07_v063_formal.sh
```

## Exp07 Formal Release Gate

The user-executed smoke artifacts at `outputs/stage4_exp07_v063-20260901_090457/` were inspected without rerunning the experiment.

```text
CANONICAL REGRESSION: PASS
EXP07 EXECUTION REGRESSION: PASS
EXP07 SMOKE VALIDATION: PASS

Smoke z range: -0.4206938926514327 to 0.1898827145660933

Smoke actuator occupancy:
fanout2 = 15
fanout3 = 218
fanout4 = 0
fanout5 = 0
fanout6 = 0

Smoke fanout violations = 0
Smoke mode violations = 0
Smoke exit code = 0

EXP07 FORMAL: READY FOR MANUAL EXECUTION
```

Because the smoke maximum score remained below `0.25`, the absence of fanouts 4, 5, and 6 is consistent with the final frozen actuator and is not a failure.

Static release verification confirmed the frozen formal design: Gossip fanouts `[2, 3, 4, 5, 6]` × 20 runs = 100 runs; AHBN adaptive with `fanout=None` × 20 runs = 20 runs; 120 total. Seeds remain `42 + run_idx` for corresponding conditions. The pinned interpreter, timestamped output isolation, raw result/trace creation, command/stdout/stderr transcript routing, real exit-code propagation, and post-run analyzer are ready.

## Formal Terminal Output

Not run. Run only after the Exp07 smoke gate is reviewed.

## Aggregation

Pending. The runner writes `aggregate_results.csv` with n, mean, sample SD, and Student-t 95% CI.

## Statistical Analysis

Pending manual formal run.

## Scientific Interpretation

Interpret the final actuator as additional high-pressure control resolution. Fanout 5/6 absence is valid when z does not cross 0.90/1.50; do not retune.

## Final Status

Prepared; smoke and formal runs not executed by Codex.

## Formal Dataset Audit

Selected formal dataset: `outputs/stage4_exp07_v063-20260901_091610`. It is the only completed formal Exp07 v0.63 candidate; the other timestamped Exp07 directory is explicitly labeled smoke. The formal directory contains one result CSV, one AHBN adaptive-trace CSV, the frozen config reference, technical validation PASS, and exit code 0.

The audit found 120 unique result rows: 20 each for Gossip f2, f3, f4, f5, f6, and AHBN adaptive, with seeds 42–61 exactly once per treatment. There were no missing treatments, duplicate treatment/seed keys, malformed or nonfinite primary metrics, invalid metric domains, negative values, fanout violations, or mode violations. The 4,883-row AHBN trace covers all 20 seeds.

## Analysis Script

`scripts/run_stage4_exp07_v063_analysis.sh`, using `scripts/analyze_stage4_exp07_v063.py` and the pinned v0.63 Python environment.

## Analysis Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp07_v063_analysis.sh outputs/stage4_exp07_v063-20260901_091610
```

## Analysis Terminal Output

The complete stdout/stderr transcript and exit code are stored at `outputs/stage4_exp07_v063-20260901_091610/exp07_v063_analysis_terminal.log`. Analysis exit code: 0.

## Aggregated Results

Values are mean with Student-t 95% CI; n=20 per treatment.

| Treatment | Delivery | Delay | Duplicates | Forwards |
|---|---:|---:|---:|---:|
| Gossip f2 | 0.7320 [0.6947, 0.7693] | 11.7258 [11.1601, 12.2915] | 74.20 [70.47, 77.93] | 146.40 [138.94, 153.86] |
| Gossip f3 | 0.9100 [0.8943, 0.9257] | 9.2524 [8.6648, 9.8400] | 153.15 [149.72, 156.58] | 243.15 [238.29, 248.01] |
| Gossip f4 | 0.9600 [0.9509, 0.9691] | 7.4629 [7.0111, 7.9147] | 201.00 [197.24, 204.76] | 296.00 [291.47, 300.53] |
| Gossip f5 | 0.9700 [0.9609, 0.9791] | 6.4785 [6.0606, 6.8963] | 233.25 [229.21, 237.29] | 329.25 [324.45, 334.05] |
| Gossip f6 | 0.9855 [0.9786, 0.9924] | 5.5893 [5.3682, 5.8103] | 258.35 [254.55, 262.15] | 355.90 [351.63, 360.17] |
| AHBN adaptive | 0.9100 [0.8943, 0.9257] | 9.2524 [8.6648, 9.8400] | 153.15 [149.72, 156.58] | 243.15 [238.29, 248.01] |

Machine-readable outputs are `exp07_v063_summary.csv`, `exp07_v063_ahbn_adaptive_summary.csv`, `exp07_v063_ahbn_vs_gossip.csv`, and `exp07_v063_figure_data.csv` in the formal directory.

## AHBN Adaptive Behavior

- z min/mean/max: -0.420694 / 0.073345 / 0.190677
- fanout 2: 310 (6.35%)
- fanout 3: 4,573 (93.65%)
- fanouts 4/5/6: 0 (0%)
- Gossip mode: 3,812 (78.07%)
- Cluster mode: 1,071 (21.93%)
- fanout violations: 0
- mode violations: 0

The observed maximum z remained below 0.25, so the absence of gears 4–6 is the correct frozen-actuator response, not a technical failure.

## AHBN vs Fixed Gossip

Compared with f2, AHBN increased delivery by 0.178 and reduced delay by 21.09%, while adding 78.95 duplicates and 96.75 forwards on average. AHBN and f3 had identical descriptive aggregate metrics. Compared with f4/f5/f6, AHBN used 17.85%/26.15%/31.68% fewer forwards and 23.81%/34.34%/40.72% fewer duplicates, but delivery was 0.050/0.060/0.0755 lower and delay was 23.98%/42.82%/65.54% higher.

## Scientific Interpretation

Fixed Gossip shows the expected trade-off across f2–f6: delivery increases monotonically and delay decreases monotonically, while duplicates and forwarding traffic increase monotonically. Delivery gains diminish after f4, but costs continue to rise.

AHBN operated almost entirely in gear 3 with occasional gear 2 decisions. Its aggregate operating point coincided descriptively with fixed Gossip f3. It therefore represents an intermediate runtime-selected compromise rather than dominance: it improves delivery and delay relative to f2 at greater cost, and reduces overhead relative to f4–f6 while accepting lower delivery and slower propagation. Outcome classification: **D — no convincing aggregate advantage over fixed Gossip f3**, alongside a valid intermediate trade-off against the other fixed settings.

Manuscript mechanism check: **CONSISTENT WITH MANUSCRIPT MECHANISM**. The new data supports the qualitative fanout/performance/duplication mechanism and balanced-operating-point framing, but not all historical numerical claims.

## Manuscript Impact

| Old claim | New v0.63 result | Status |
|---|---|---|
| Gossip reaches approximately 0.98 delivery at f6 | f6 mean delivery = 0.9855 | RETAIN |
| Gossip duplicates increase strongly with fanout | means rise from 74.20 at f2 to 258.35 at f6 | RETAIN |
| AHBN maintains lower duplicate overhead | true relative to f4–f6; equal to f3 and higher than f2 | UPDATE |
| AHBN achieves approximately 62.2% duplicate reduction relative to f6 | current reduction = 40.72% | UPDATE |
| AHBN maintains competitive propagation performance | equal to f3, faster than f2, slower than f4–f6 | UPDATE |

No old claim requires removal, but the three AHBN claims require the stated qualification or revised value.

## Exp07 Freeze Decision

```text
TECHNICAL VALIDATION: PASS
SCIENTIFIC RESULT: MIXED
EXP07 FREEZE: PASS
```

The dataset is complete, the adaptive trace is canonical, the statistics are reproducible, and the mixed outcome is scientifically interpretable. No Exp07 rerun is warranted. Exp07 is ready to freeze; do not run Exp08 until user review.
Stage 4 exp07 ControlSim v0.63 smoke
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_090457/exp07_fanout_smoke.yaml
Output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_090457
Saved outputs/csv/exp07_results_20260901_090459.csv
Saved outputs/csv/exp07_adaptive_trace_20260901_090459.csv
{
  "validation": "PASS",
  "results": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_090457/outputs/csv/exp07_results_20260901_090459.csv",
  "trace": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_090457/outputs/csv/exp07_adaptive_trace_20260901_090459.csv",
  "trace_rows": 233,
  "z_min": -0.4206938926514327,
  "z_mean": 0.07632211140246864,
  "z_max": 0.1898827145660933,
  "d_hat_min": 0.0,
  "l_hat_min": 0.0,
  "u_hat_min": 0.0,
  "c_hat_min": 0.0,
  "d_hat_max": 0.943693553630658,
  "l_hat_max": 0.505102513490026,
  "u_hat_max": 0.1092499999999999,
  "c_hat_max": 0.0,
  "fanout_2_count": 15,
  "fanout_3_count": 218,
  "fanout_4_count": 0,
  "fanout_5_count": 0,
  "fanout_6_count": 0,
  "fanout_2_proportion": 0.06437768240343347,
  "fanout_3_proportion": 0.9356223175965666,
  "fanout_4_proportion": 0.0,
  "fanout_5_proportion": 0.0,
  "fanout_6_proportion": 0.0,
  "gossip_mode_count": 186,
  "gossip_mode_proportion": 0.7982832618025751,
  "cluster_mode_count": 47,
  "cluster_mode_proportion": 0.2017167381974249,
  "fanout_violations": 0,
  "mode_violations": 0
}
TECHNICAL VALIDATION: PASS
EXIT CODE: 0
OUTPUT DIRECTORY: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_090457
Stage 4 exp07 ControlSim v0.63 formal
Command: cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
Command: bash scripts/run_stage4_exp07_v063_formal.sh
Python: /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/configs/exp07_fanout.yaml
Output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610
Saved outputs/csv/exp07_results_20260901_091613.csv
Saved outputs/csv/exp07_adaptive_trace_20260901_091613.csv
{
  "validation": "PASS",
  "results": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/outputs/csv/exp07_results_20260901_091613.csv",
  "trace": "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/outputs/csv/exp07_adaptive_trace_20260901_091613.csv",
  "trace_rows": 4883,
  "z_min": -0.4206938926514327,
  "z_mean": 0.07334546283492155,
  "z_max": 0.190677226716101,
  "d_hat_min": 0.0,
  "l_hat_min": 0.0,
  "u_hat_min": 0.0,
  "c_hat_min": 0.0,
  "d_hat_max": 0.9469491239050968,
  "l_hat_max": 0.5064668593737915,
  "u_hat_max": 0.1092499999999999,
  "c_hat_max": 0.0,
  "fanout_2_count": 310,
  "fanout_3_count": 4573,
  "fanout_4_count": 0,
  "fanout_5_count": 0,
  "fanout_6_count": 0,
  "fanout_2_proportion": 0.06348556215441327,
  "fanout_3_proportion": 0.9365144378455867,
  "fanout_4_proportion": 0.0,
  "fanout_5_proportion": 0.0,
  "fanout_6_proportion": 0.0,
  "gossip_mode_count": 3812,
  "gossip_mode_proportion": 0.7806676223633012,
  "cluster_mode_count": 1071,
  "cluster_mode_proportion": 0.21933237763669874,
  "fanout_violations": 0,
  "mode_violations": 0
}
TECHNICAL VALIDATION: PASS
EXIT CODE: 0
OUTPUT DIRECTORY: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610
Analysis command: bash scripts/run_stage4_exp07_v063_analysis.sh outputs/stage4_exp07_v063-20260901_091610
Selected dataset: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610
selected formal output directory: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610
timestamp: 20260901_091610
result file count: 1
trace file count: 1
config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/configs/exp07_fanout.yaml
results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/outputs/csv/exp07_results_20260901_091613.csv
trace: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/outputs/csv/exp07_adaptive_trace_20260901_091613.csv
expected rows: 120
actual rows: 120
treatment counts: {"AHBN adaptive": 20, "Gossip f2": 20, "Gossip f3": 20, "Gossip f4": 20, "Gossip f5": 20, "Gossip f6": 20}
EXP07 v0.63 FORMAL ANALYSIS

PRIMARY RESULTS
Gossip f2: delivery=0.732000 [0.694695, 0.769305]; delay=11.725782 [11.160110, 12.291453]; duplicates=74.200 [70.469, 77.931]; forwards=146.400 [138.939, 153.861]
Gossip f3: delivery=0.910000 [0.894293, 0.925707]; delay=9.252378 [8.664768, 9.839989]; duplicates=153.150 [149.717, 156.583]; forwards=243.150 [238.293, 248.007]
Gossip f4: delivery=0.960000 [0.950889, 0.969111]; delay=7.462899 [7.011087, 7.914710]; duplicates=201.000 [197.244, 204.756]; forwards=296.000 [291.473, 300.527]
Gossip f5: delivery=0.970000 [0.960889, 0.979111]; delay=6.478463 [6.060637, 6.896290]; duplicates=233.250 [229.207, 237.293]; forwards=329.250 [324.451, 334.049]
Gossip f6: delivery=0.985500 [0.978629, 0.992371]; delay=5.589269 [5.368230, 5.810307]; duplicates=258.350 [254.553, 262.147]; forwards=355.900 [351.630, 360.170]
AHBN adaptive: delivery=0.910000 [0.894293, 0.925707]; delay=9.252378 [8.664768, 9.839989]; duplicates=153.150 [149.717, 156.583]; forwards=243.150 [238.293, 248.007]

AHBN ADAPTIVE STATE
{"c_hat_max": 0.0, "c_hat_mean": 0.0, "c_hat_min": 0.0, "cluster_mode_count": 1071, "cluster_mode_proportion": 0.21933237763669874, "d_hat_max": 0.9469491239050968, "d_hat_mean": 0.25541256665659, "d_hat_min": 0.0, "fanout_2_count": 310, "fanout_2_proportion": 0.06348556215441327, "fanout_3_count": 4573, "fanout_3_proportion": 0.9365144378455867, "fanout_4_count": 0, "fanout_4_proportion": 0.0, "fanout_5_count": 0, "fanout_5_proportion": 0.0, "fanout_6_count": 0, "fanout_6_proportion": 0.0, "fanout_violations": 0, "gossip_mode_count": 3812, "gossip_mode_proportion": 0.7806676223633012, "l_hat_max": 0.5064668593737915, "l_hat_mean": 0.2752416747297723, "l_hat_min": 0.0, "mode_violations": 0, "trace_rows": 4883, "u_hat_max": 0.1092499999999999, "u_hat_mean": 0.05351635476173926, "u_hat_min": 0.0, "z_max": 0.190677226716101, "z_mean": 0.07334546283492155, "z_min": -0.4206938926514327}

AHBN VS FIXED GOSSIP
f2: delivery_delta=+0.178000; delay_delta=-2.473403 (-21.09%); duplicates_delta=+78.950 (reduction=-106.40%); forwards_delta=+96.750 (reduction=-66.09%)
f3: delivery_delta=+0.000000; delay_delta=+0.000000 (+0.00%); duplicates_delta=+0.000 (reduction=+0.00%); forwards_delta=+0.000 (reduction=+0.00%)
f4: delivery_delta=-0.050000; delay_delta=+1.789480 (+23.98%); duplicates_delta=-47.850 (reduction=+23.81%); forwards_delta=-52.850 (reduction=+17.85%)
f5: delivery_delta=-0.060000; delay_delta=+2.773915 (+42.82%); duplicates_delta=-80.100 (reduction=+34.34%); forwards_delta=-86.100 (reduction=+26.15%)
f6: delivery_delta=-0.075500; delay_delta=+3.663110 (+65.54%); duplicates_delta=-105.200 (reduction=+40.72%); forwards_delta=-112.750 (reduction=+31.68%)

DATASET AUDIT: PASS
TECHNICAL VALIDATION: PASS
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/exp07_v063_summary.csv
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/exp07_v063_ahbn_adaptive_summary.csv
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/exp07_v063_ahbn_vs_gossip.csv
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/exp07_v063_figure_data.csv
Saved /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/outputs/stage4_exp07_v063-20260901_091610/exp07_v063_analysis.txt
ANALYSIS EXIT CODE: 0
