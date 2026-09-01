# Exp09 v0.62 rerun record

Control Simulator Version: v0.62

Parent Reference: v0.61

Reason for v0.62: Synchronize the Control Simulator with the latest frozen canonical AHBN controller and regenerate Exp07, Exp08, and Exp09 evidence.

v0.61 Status: Preserved unchanged as the historical pre-correction reference.

## Status

- Project root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.62`
- Python: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Canonical regression: PASS.
- Exp07 AHBN smoke: PASS.
- Exp08 AHBN smoke: STOPPED on an output-write `PermissionError`.
- Exp09 smoke: NOT STARTED because the strict smoke gate requires stopping after any exception.
- Formal Exp09: NOT STARTED.

The intended Exp09 smoke remains AHBN only, seed 42, N=100, and ER p values 0.04, 0.06, 0.08, 0.10, and 0.12 (five runs). No experiment-design parameter was changed.

## Continuation: successful smoke

Command:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config /private/tmp/v062_exp09_smoke.yaml
```

Terminal output:

```text
Saved outputs/csv/exp09_results_20260826_075709.csv
Saved outputs/csv/exp09_adaptive_trace_20260826_075709.csv
```

All ranges below are min / mean / max. Raw churn was absent in every condition because no churn occurred; `c_hat` remained 0 / 0 / 0.

| p | result rows | trace rows | raw d | raw l | raw u | d_hat | l_hat | u_hat | z | weight |
|---:|---:|---:|---|---|---|---|---|---|---|---|
| 0.04 | 1 | 228 | 0 / 0.350799 / 0.875000 | 0 / 0.497955 / 0.521675 | 0 / 0.159964 / 0.375 | 0 / 0.154981 / 0.752604 | 0 / 0.239415 / 0.472816 | 0 / 0.074118 / 0.163875 | -0.156006 / 0.158553 / 0.227041 | 0.461077 / 0.539534 / 0.556518 |
| 0.06 | 1 | 272 | 0 / 0.414842 / 0.875000 | 0 / 0.498257 / 0.521709 | 0 / 0.172784 / 0.375 | 0 / 0.204663 / 0.752604 | 0 / 0.261755 / 0.475959 | 0 / 0.089463 / 0.163875 | -0.162619 / 0.146554 / 0.227937 | 0.459435 / 0.536541 / 0.556739 |
| 0.08 | 1 | 285 | 0 / 0.422238 / 0.888889 | 0 / 0.497365 / 0.521578 | 0 / 0.179374 / 0.375 | 0 / 0.203569 / 0.793489 | 0 / 0.262343 / 0.486123 | 0 / 0.092027 / 0.163875 | -0.195719 / 0.150801 / 0.225571 | 0.451226 / 0.537600 / 0.556155 |
| 0.10 | 1 | 278 | 0 / 0.421775 / 0.875000 | 0 / 0.497019 / 0.521628 | 0 / 0.182320 / 0.375 | 0 / 0.202813 / 0.752604 | 0 / 0.262101 / 0.465138 | 0 / 0.091943 / 0.163875 | -0.166103 / 0.151230 / 0.226803 | 0.458569 / 0.537704 / 0.556459 |
| 0.12 | 1 | 289 | 0 / 0.431002 / 0.857143 | 0 / 0.497284 / 0.521709 | 0 / 0.177614 / 0.375 | 0 / 0.215145 / 0.700148 | 0 / 0.266824 / 0.466199 | 0 / 0.093247 / 0.163875 | -0.108669 / 0.144926 / 0.227389 | 0.472859 / 0.536133 / 0.556604 |

MODE x FANOUT, transitions, and outcomes:

| p | c2 | c3 | c4 | g2 | g3 | g4 | mode trans. | fanout trans. | invariant failures | delivery | delay | duplicates | forwards |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.04 | 0 | 5 | 0 | 0 | 223 | 0 | 3 | 0 | 0 | 0.969697 | 8.833894 | 132 | 227 |
| 0.06 | 0 | 13 | 0 | 0 | 259 | 0 | 6 | 0 | 0 | 0.96 | 7.881526 | 176 | 271 |
| 0.08 | 0 | 10 | 0 | 0 | 275 | 0 | 7 | 0 | 0 | 0.96 | 7.621404 | 189 | 284 |
| 0.10 | 0 | 12 | 0 | 0 | 266 | 0 | 7 | 0 | 0 | 0.93 | 7.559361 | 185 | 277 |
| 0.12 | 0 | 16 | 0 | 0 | 273 | 0 | 10 | 0 | 0 | 0.96 | 7.556500 | 193 | 288 |

All available observations were within [0,1], all decision fields were finite, and fanout stayed in [2,4]. Scientific interpretation: duplicate pressure and total forwarding generally increased with density, strengthening the negative `-d` term. Smoothed utilization also increased overall and contributes positively, while latency remained similar. Their competition yields modest, non-monotonic mean scores. Every observed score remained in `(-0.25, 0.25)`, so fanout 3 for every row is the correct frozen-actuator behavior; mode still switched at zero.

Smoke decision: PASS. Formal Exp09 remains NOT STARTED.

## Formal Exp09 execution

- Start/end timestamp: `2026-08-26 08:13:23` (generated filenames)
- Exact command: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp09_dense_topology.yaml`

Terminal output:

```text
Saved outputs/csv/exp09_results_20260826_081323.csv
Saved outputs/csv/exp09_adaptive_trace_20260826_081323.csv
```

Validation: 400 result rows; all 20 algorithm x density cells contain exactly seeds 42-61; no duplicate/missing/extra cells; primary metrics are finite; all trace densities and seeds are represented; observations are bounded where present; decision values are finite; fanout is within [2,4]; controller invariant failures = 0.

Descriptive means:

| Algorithm | Density | n | Delivery | Delay | Duplicates | Forwards |
|---|---:|---:|---:|---:|---:|---:|
| AHBN | 0.04 | 20 | 0.961943 | 8.075682 | 131.90 | 225.75 |
| AHBN | 0.06 | 20 | 0.966949 | 7.814579 | 179.20 | 274.75 |
| AHBN | 0.08 | 20 | 0.965500 | 8.257322 | 190.70 | 286.25 |
| AHBN | 0.10 | 20 | 0.969500 | 7.724285 | 194.45 | 290.40 |
| AHBN | 0.12 | 20 | 0.967500 | 7.714127 | 194.30 | 290.05 |
| Structured | 0.04 | 20 | 1.000000 | 4.520881 | 0.00 | 97.60 |
| Structured | 0.06 | 20 | 1.000000 | 4.519793 | 0.00 | 98.85 |
| Structured | 0.08 | 20 | 1.000000 | 4.519793 | 0.00 | 99.00 |
| Structured | 0.10 | 20 | 1.000000 | 4.519793 | 0.00 | 99.00 |
| Structured | 0.12 | 20 | 1.000000 | 4.519793 | 0.00 | 99.00 |
| DC-SoC | 0.04 | 20 | 1.000000 | 1.197891 | 0.00 | 97.60 |
| DC-SoC | 0.06 | 20 | 1.000000 | 1.197891 | 0.00 | 98.85 |
| DC-SoC | 0.08 | 20 | 1.000000 | 1.197891 | 0.00 | 99.00 |
| DC-SoC | 0.10 | 20 | 1.000000 | 1.197891 | 0.00 | 99.00 |
| DC-SoC | 0.12 | 20 | 1.000000 | 1.197891 | 0.00 | 99.00 |
| Gossip | 0.04 | 20 | 1.000000 | 6.029857 | 209.00 | 306.60 |
| Gossip | 0.06 | 20 | 1.000000 | 4.442501 | 408.50 | 507.35 |
| Gossip | 0.08 | 20 | 1.000000 | 3.871893 | 601.90 | 700.90 |
| Gossip | 0.10 | 20 | 1.000000 | 3.323639 | 792.80 | 891.80 |
| Gossip | 0.12 | 20 | 1.000000 | 3.241952 | 984.10 | 1083.10 |

AHBN trace summary:

| Density | Rows | z mean | z min | z max | c2 | c3 | g3 | Mode transitions | Fanout transitions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.04 | 4535 | 0.159034 | -0.156006 | 0.228041 | 0 | 68 | 4467 | 123 | 0 |
| 0.06 | 5515 | 0.155366 | -0.201220 | 0.228301 | 0 | 150 | 5365 | 274 | 0 |
| 0.08 | 5745 | 0.152433 | -0.301521 | 0.228102 | 2 | 206 | 5537 | 374 | 4 |
| 0.10 | 5828 | 0.152162 | -0.234226 | 0.228224 | 0 | 223 | 5605 | 402 | 0 |
| 0.12 | 5821 | 0.152862 | -0.266190 | 0.228177 | 1 | 206 | 5614 | 371 | 2 |

Formal Exp09 execution validation: PASS
