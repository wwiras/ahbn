# Exp08 v0.62 rerun record

Control Simulator Version: v0.62

Parent Reference: v0.61

Reason for v0.62: Synchronize the Control Simulator with the latest frozen canonical AHBN controller and regenerate Exp07, Exp08, and Exp09 evidence.

v0.61 Status: Preserved unchanged as the historical pre-correction reference.

## Validation state

- Project root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.62`
- Python: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Canonical regression: PASS.
- Exp07 AHBN smoke: PASS.
- Exp08 smoke configuration: AHBN only; seed 42; overload factors 1.0, 1.5, 2.0, and 3.0; four intended runs.

Command:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config /private/tmp/v062_exp08_smoke.yaml
```

Relevant terminal output:

```text
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.62/run_batch.py", line 881, in <module>
    main()
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.62/run_batch.py", line 770, in main
    path = save_results_csv(rows, f"outputs/csv/exp08_results_{ts}.csv", add_timestamp=False)
PermissionError: [Errno 1] Operation not permitted: 'outputs/csv/exp08_results_20260826_075046.csv'
```

Smoke decision: FAIL/STOP. The failure occurred while saving output because the command did not have write permission. No algorithmic correction was attempted, Exp09 was not started, and no formal run was started.

Output files: none from this failed invocation.

Final status: blocked at the strict smoke gate pending a clean rerun with v0.62 output-write permission.

## Continuation: filesystem diagnosis and successful rerun

The project root and interpreter were verified as:

```text
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.62
Python 3.14.6
```

Diagnostic commands inspected `ls -ldeO@`, `stat -x`, the prospective filename, and a harmless create/list/remove test. `outputs` and `outputs/csv` are owned by `wwiras:staff`, mode 0755, with no restrictive ACL or immutable flag. The failed filename did not exist. An ordinary sandboxed `touch` returned `Operation not permitted`; the identical write test with explicit access succeeded and the test file was removed.

Exact diagnosis: the original process was denied by the managed execution sandbox because v0.62 is outside its configured writable workspace. This was not a macOS ownership, ACL, flag, collision, working-directory, or application defect. Correction: grant the run write authorization for the existing v0.62 path. No chmod, chown, source-code, algorithm, metric, config, or output-path change was made.

Rerun command:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config /private/tmp/v062_exp08_smoke.yaml
```

Terminal output:

```text
Saved outputs/csv/exp08_results_20260826_075643.csv
Saved outputs/csv/exp08_execution_evidence_20260826_075643.csv
Saved outputs/csv/exp08_ahbn_adaptive_trace_20260826_075643.csv
Saved outputs/csv/exp08_s3_manifest.json
```

All ranges below are min / mean / max. Raw churn was absent in every condition because no churn occurred; `c_hat` remained 0 / 0 / 0.

| factor | result rows | trace rows | raw d | raw l | raw u | d_hat | l_hat | u_hat | z | weight |
|---:|---:|---:|---|---|---|---|---|---|---|---|
| 1.0 | 1 | 233 | 0 / 0.423894 / 0.952381 | 0 / 0.497485 / 0.521675 | 0 / 0.146672 / 0.375 | 0 / 0.247990 / 0.943694 | 0 / 0.271753 / 0.505103 | 0 / 0.078838 / 0.163875 | -0.406817 / 0.102602 / 0.227383 | 0.399676 / 0.525621 / 0.556602 |
| 1.5 | 1 | 237 | 0 / 0.446028 / 0.947368 | 0 / 0.512203 / 0.606387 | 0 / 0.131723 / 0.375 | 0 / 0.283813 / 0.935672 | 0 / 0.295787 / 0.595306 | 0 / 0.080373 / 0.163875 | -0.386736 / 0.092347 / 0.262724 | 0.404503 / 0.523055 / 0.565306 |
| 2.0 | 1 | 243 | 0 / 0.434231 / 0.950000 | 0 / 0.519376 / 0.666342 | 0 / 0.136604 / 0.375 | 0 / 0.269126 / 0.939970 | 0 / 0.294223 / 0.655217 | 0 / 0.077881 / 0.163875 | -0.389586 / 0.102979 / 0.298366 | 0.403817 / 0.525700 / 0.574043 |
| 3.0 | 1 | 245 | 0 / 0.443590 / 0.950000 | 0 / 0.534671 / 0.743576 | 0 / 0.133124 / 0.375 | 0 / 0.282552 / 0.939970 | 0 / 0.312134 / 0.739861 | 0 / 0.078719 / 0.163875 | -0.386268 / 0.108302 / 0.339212 | 0.404616 / 0.527009 / 0.583999 |

MODE x FANOUT, transitions, and outcomes:

| factor | c2 | c3 | c4 | g2 | g3 | g4 | mode trans. | fanout trans. | invariant failures | delivery | delay | duplicates | forwards |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 13 | 22 | 0 | 0 | 198 | 0 | 6 | 2 | 0 | 0.87 | 9.917799 | 146 | 232 |
| 1.5 | 13 | 32 | 0 | 0 | 189 | 3 | 10 | 8 | 0 | 0.89 | 12.082551 | 148 | 236 |
| 2.0 | 11 | 29 | 0 | 0 | 197 | 6 | 6 | 8 | 0 | 0.92 | 8.716696 | 151 | 242 |
| 3.0 | 9 | 33 | 0 | 0 | 194 | 9 | 7 | 7 | 0 | 0.92 | 9.799653 | 153 | 244 |

All available observations were within [0,1], all decision fields were finite, and fanout stayed in [2,4]. Scientific interpretation: increasing overload most clearly raised raw and smoothed latency maxima. Utilization did not rise monotonically in aggregate, while duplicate pressure also varied and contributes negatively. These competing terms explain the non-monotonic mean score. Factors 1.5, 2.0, and 3.0 reached `z >= 0.25`, producing 3, 6, and 9 Gossip/fanout-4 decisions respectively, exactly as the frozen actuator requires.

Updated smoke decision: PASS.

## Formal Exp08 execution

- Start/end timestamp: `2026-08-26 08:11:47` (generated filenames/manifest)
- Exact command: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp08_ch_bottleneck.yaml`

Terminal output:

```text
Saved outputs/csv/exp08_results_20260826_081147.csv
Saved outputs/csv/exp08_execution_evidence_20260826_081147.csv
Saved outputs/csv/exp08_ahbn_adaptive_trace_20260826_081147.csv
Saved outputs/csv/exp08_s3_manifest.json
```

Validation: 320 result rows; all 16 algorithm x overload cells contain exactly seeds 42-61; no duplicate/missing/extra cells; primary metrics are finite; all trace factors and seeds are represented; observations are bounded where present; decision values are finite; fanout is within [2,4]; controller invariant failures = 0.

Descriptive means:

| Algorithm | Factor | n | Delivery | Delay | Duplicates | Forwards |
|---|---:|---:|---:|---:|---:|---:|
| AHBN | 1.0 | 20 | 0.9100 | 9.252378 | 153.15 | 243.15 |
| AHBN | 1.5 | 20 | 0.8990 | 9.679536 | 151.85 | 240.75 |
| AHBN | 2.0 | 20 | 0.9035 | 9.002786 | 151.85 | 241.20 |
| AHBN | 3.0 | 20 | 0.8935 | 10.033341 | 150.70 | 239.05 |
| Structured | 1.0 | 20 | 1.0000 | 4.519793 | 0.00 | 99.00 |
| Structured | 1.5 | 20 | 1.0000 | 6.019793 | 0.00 | 99.00 |
| Structured | 2.0 | 20 | 1.0000 | 7.519793 | 0.00 | 99.00 |
| Structured | 3.0 | 20 | 1.0000 | 10.519793 | 0.00 | 99.00 |
| DC-SoC | 1.0 | 20 | 1.0000 | 1.198268 | 0.00 | 99.00 |
| DC-SoC | 1.5 | 20 | 1.0000 | 1.698268 | 0.00 | 99.00 |
| DC-SoC | 2.0 | 20 | 1.0000 | 2.198268 | 0.00 | 99.00 |
| DC-SoC | 3.0 | 20 | 1.0000 | 3.198268 | 0.00 | 99.00 |
| Gossip | 1.0 | 20 | 1.0000 | 3.279219 | 384.00 | 483.00 |
| Gossip | 1.5 | 20 | 1.0000 | 3.279219 | 384.00 | 483.00 |
| Gossip | 2.0 | 20 | 1.0000 | 3.279219 | 384.00 | 483.00 |
| Gossip | 3.0 | 20 | 1.0000 | 3.279219 | 384.00 | 483.00 |

AHBN trace summary:

| Factor | Rows | z mean | z min | z max | c2 | c3 | g3 | g4 | Mode transitions | Fanout transitions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 4883 | 0.100104 | -0.406817 | 0.228177 | 241 | 580 | 4062 | 0 | 1097 | 400 |
| 1.5 | 4835 | 0.110287 | -0.403347 | 0.270308 | 147 | 586 | 4050 | 52 | 995 | 351 |
| 2.0 | 4844 | 0.117125 | -0.409731 | 0.300805 | 120 | 554 | 4072 | 98 | 929 | 393 |
| 3.0 | 4801 | 0.122492 | -0.400535 | 0.341106 | 119 | 524 | 4018 | 140 | 912 | 462 |

Formal Exp08 execution validation: PASS
