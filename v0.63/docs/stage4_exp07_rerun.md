# Exp07 v0.62 rerun record

Control Simulator Version: v0.62

Parent Reference: v0.61

Reason for v0.62: Synchronize the Control Simulator with the latest frozen canonical AHBN controller and regenerate Exp07, Exp08, and Exp09 evidence.

v0.61 Status: Preserved unchanged as the historical pre-correction reference.

## Provenance and canonical correction

- Project root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.62`
- Python: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python` (Python 3.14.6)
- Inspected/modified: `ahbn/control.py`, `run_batch.py`, `run_one.py`, and the Exp07/08/09 configs.
- Added: `scripts/validate_canonical_ahbn_frozen.py`.
- Frozen score: `z = -d + l + u + c`; zero centres; mode is Gossip at `z >= 0`; fanout is 2/3/4 at the frozen score thresholds.
- Comparator algorithms and simulator observation mappings were not modified.

## Canonical regression

Command:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_canonical_ahbn_frozen.py
```

Output:

```text
PASS: frozen canonical AHBN defaults, score, mode, and fanout anchors
```

## Exp07 smoke

Configuration: AHBN only; seed 42; BA m=3; N=100; one run. The smoke-only config was `/private/tmp/v062_exp07_smoke.yaml`; the authoritative config remains 20 runs per setting.

Command:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config /private/tmp/v062_exp07_smoke.yaml
```

Output:

```text
Saved outputs/csv/exp07_results_20260826_075041.csv
Saved outputs/csv/exp07_adaptive_trace_20260826_075041.csv
```

Trace rows: 233.

| Signal | min | mean | max |
|---|---:|---:|---:|
| raw d | 0.000000 | 0.423894 | 0.952381 |
| raw l | 0.000000 | 0.497485 | 0.521675 |
| raw u | 0.000000 | 0.146672 | 0.375000 |
| raw c | N/A | N/A | N/A |
| d_hat | 0.000000 | 0.247990 | 0.943694 |
| l_hat | 0.000000 | 0.271753 | 0.505103 |
| u_hat | 0.000000 | 0.078838 | 0.163875 |
| c_hat | 0.000000 | 0.000000 | 0.000000 |
| z | -0.406817 | 0.102602 | 0.227383 |
| weight | 0.399676 | 0.525621 | 0.556602 |

Raw churn is absent because this non-churn experiment emitted no churn observation updates; the controller's smoothed churn state remained zero. No present raw observation was outside [0,1], and no smoothed/decision field was NaN.

MODE x FANOUT counts:

```text
('cluster', 2) = 13
('cluster', 3) = 22
('cluster', 4) = 0
('gossip', 2) = 0
('gossip', 3) = 198
('gossip', 4) = 0
```

- Mode transitions: 6
- Fanout transitions: 2
- Controller invariant failures: 0
- Delivery ratio: 0.87
- Propagation delay: 9.917798637579718
- Duplicates: 146
- Total forwards: 232

Smoke decision: PASS. Fanout 4 was not reached because the maximum observed score was 0.227383, below the +0.25 threshold; this is consistent with the frozen actuator.

Formal status: NOT STARTED. The overall smoke gate later stopped at Exp08 due to an output-write exception.

## Pre-formal smoke-gate continuation

The Exp08 failure was subsequently diagnosed as a managed sandbox write-authorization issue.

No source-code, filesystem-metadata, experiment-design, or algorithmic change was required.

Exp08 smoke was rerun successfully.

Exp09 smoke was then completed successfully.

The final pre-formal validation state is:

```text
Canonical validator: PASS
Exp07 smoke: PASS
Exp08 smoke: PASS
Exp09 smoke: PASS
```

Therefore the full smoke gate ultimately PASSED and v0.62 was frozen for formal evaluation.

Documentation update/verification command:

```text
apply_patch (append-only update to docs/stage4_exp07_rerun.md), followed by tail -60 docs/stage4_exp07_rerun.md
```

## Formal Exp07 execution

- Start timestamp: `2026-08-26 08:10:46` (from generated result filename)
- End timestamp: `2026-08-26 08:10:47` (from generated trace filename)
- Exact command: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp07_fanout.yaml`

Terminal output:

```text
Saved outputs/csv/exp07_results_20260826_081046.csv
Saved outputs/csv/exp07_adaptive_trace_20260826_081047.csv
```

Evidence:

- Result CSV: `outputs/csv/exp07_results_20260826_081046.csv`
- Adaptive trace CSV: `outputs/csv/exp07_adaptive_trace_20260826_081047.csv`
- Other manifest/evidence files generated: none reported

Validation summary:

```text
VALIDATION PASS
result rows: 120
Gossip rows: 100
AHBN rows: 20
seed range: 42-61
Gossip fanout cells 2,3,4,5,6: 20 rows each
AHBN trace rows: 4883
AHBN mode counts: gossip=4062, cluster=821
AHBN fanout counts: fanout 2=241, fanout 3=4642, fanout 4=0
mode transitions: 1097
fanout transitions: 400
controller invariant failures: 0
```

Descriptive means (`delivery_ratio`, `propagation_delay`, `duplicates`, `total_forwards`):

| Condition | n | Delivery ratio | Propagation delay | Duplicates | Total forwards |
|---|---:|---:|---:|---:|---:|
| Gossip fanout 2 | 20 | 0.7320 | 11.725782 | 74.20 | 146.40 |
| Gossip fanout 3 | 20 | 0.9100 | 9.252378 | 153.15 | 243.15 |
| Gossip fanout 4 | 20 | 0.9600 | 7.462899 | 201.00 | 296.00 |
| Gossip fanout 5 | 20 | 0.9700 | 6.478463 | 233.25 | 329.25 |
| Gossip fanout 6 | 20 | 0.9855 | 5.589269 | 258.35 | 355.90 |
| AHBN canonical | 20 | 0.9100 | 9.252378 | 153.15 | 243.15 |

Formal Exp07 execution validation: PASS
