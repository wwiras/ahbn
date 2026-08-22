# Minimal max_fanout=6 Amendment Validation

## 1. Purpose

This validation does not optimize AHBN's maximum fanout. It evaluates whether a topology-derived ceiling of six, motivated by the expected mean degree 2m=6 of the canonical BA(m=3) topology, is a defensible amendment to the previously frozen [2,4] fanout range.

The validation stopped during V1 repository inspection under the mandatory failure / unexpected-result rule. No validation experiment was created or run.

## 2. Environment

```text
Project root:
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

Virtual environment:
/Users/wwiras/Documents/src/AHBNProj/venv0.6

Python interpreter:
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
```

## 3. Repository Inspection

The project root was listed and a read-only broad ripgrep inspection was started for canonical and experimental fanout references. The visible portion confirmed canonical defaults including `min_fanout = 2` and `max_fanout = 4` in `ahbn/control.py`, runner fallbacks of 2 and 4, and the adaptive strategy's use of `node.control.fanout`. The inspection was not completed because the terminal capture was truncated.

## 4. Commands Executed

Commands are recorded exactly in execution order.

```sh
pwd
ls -la '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
rg -n --hidden --glob '!*.csv' --glob '!*.log' "min_fanout|max_fanout|fanout" '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
```

## 5. Terminal Output

The command completed with exit code 0, but the tool returned the following unexpected capture condition:

```text
Warning: truncated output (original token count: 18432)
Total output lines: 517
```

The visible output began:

```text
/Users/wwiras/Documents/ChatGPT/AHBN
total 152
drwxr-xr-x@ 13 wwiras  staff    416 Aug 21 08:09 .
drwxr-xr-x@ 16 wwiras  staff    512 Aug 21 08:09 ..
-rw-r--r--@  1 wwiras  staff  10244 Aug 21 08:51 .DS_Store
-rw-rw-r--@  1 wwiras  staff  21967 Aug 17 12:20 README.md
drwxr-xr-x@  6 wwiras  staff    192 Aug 19 22:21 __pycache__
drwxr-xr-x@ 18 wwiras  staff    576 Aug 19 11:39 ahbn
drwxr-xr-x@ 13 wwiras  staff    416 Aug 19 05:58 configs
drwxr-xr-x@ 12 wwiras  staff    384 Aug 21 08:36 docs
drwxr-xr-x@  8 wwiras  staff    256 Aug 21 08:49 outputs
-rw-rw-r--@  1 wwiras  staff    134 Aug 19 05:59 requirements.txt
-rw-r--r--@  1 wwiras  staff  28443 Aug 19 22:18 run_batch.py
-rw-r--r--@  1 wwiras  staff   5536 Aug 19 06:05 run_one.py
drwxr-xr-x@ 26 wwiras  staff    832 Aug 21 07:24 scripts
```

Relevant visible findings included:

```text
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/ahbn/control.py:84:    min_fanout: int = 2
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/ahbn/control.py:85:    max_fanout: int = 4
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/run_one.py:35:        min_fanout=ahbn_cfg.get("min_fanout", 2),
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/run_one.py:36:        max_fanout=ahbn_cfg.get("max_fanout", 4),
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/run_batch.py:40:        min_fanout=ahbn_cfg.get("min_fanout", 2),
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/run_batch.py:41:        max_fanout=ahbn_cfg.get("max_fanout", 4),
```

The full ripgrep output was not available because the command output was truncated. Per the mandatory stop rule, it was not silently reconstructed or replaced.

### Error / failed condition

The repository search generated more terminal output than the tool capture returned. This is unexpected output and prevents recording the complete relevant terminal output with confidence. Validation stopped immediately before later V1 inspection or any V2-V5 execution.

## 6. Files Created

- `docs/fanout6_result.md`

No validation scripts, configurations, raw CSVs, adaptive traces, summaries, or experiment logs were created.

## 7. Runs Performed

- Scenarios: none
- max_fanout values run: none
- Seeds: none
- Repetitions: none
- Expected number of runs: not yet determined because V1 did not complete
- Actual number of completed runs: 0

## 8. Results

No experimental results were produced.

## 9. Adaptive Behaviour

Not evaluated because validation stopped during V1.

## 10. Scientific Interpretation

No scientific interpretation is possible from an incomplete repository inspection and zero validation runs.

## 11. Final Decision

Not yet made. The earlier `RETAIN [2,4]` text was only a temporary disposition after an inspection-output truncation and is not scientific evidence for or against either bound. The corrected validation must complete before selecting `RETAIN [2,4]` or `AMEND TO [2,6]`.

## 12. Frozen Implementation Status

```text
Frozen canonical AHBN modified: No
Existing Stage 2 outputs modified: No
Existing Stage 4 outputs modified: No
Comparator implementations modified: No
```

## V1 Retry After Truncated Inspection

The initial broad repository search exceeded the terminal output limit. This was an inspection-command issue, not an AHBN or experimental failure. V1 was therefore repeated using targeted bounded inspection commands.

### CORRECTED TARGETED V1 INSPECTION

#### Commands Executed

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
sed -n '60,100p' ahbn/control.py
sed -n '235,325p' ahbn/control.py
sed -n '1,145p' ahbn/strategies/ahbn.py
sed -n '1,130p' ahbn/strategies/gossip.py
rg -n "dense|bottleneck|churn|ba_m|seeds|runs|repetitions" configs scripts --glob '*.yaml' --glob '*.py' | head -n 160
```

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
sed -n '1,165p' configs/stage2_parameter_sensitivity.yaml
sed -n '1,315p' scripts/run_stage2_sensitivity.py
sed -n '1,180p' ahbn/strategies/cluster.py
ls -la scripts/*fanout6* configs/*fanout6* outputs/*fanout6* 2>&1
```

#### Relevant Terminal Output

The first targeted command completed successfully. Relevant source output was:

```text
    # Canonical forwarding bounds
    min_fanout: int = 2
    max_fanout: int = 4

        fanout_span = p.max_fanout - p.min_fanout

        raw_fanout = (
            p.min_fanout
            + p.beta * state.weight * fanout_span
        )

        state.fanout = int(
            round(
                self.clamp(
                    raw_fanout,
                    p.min_fanout,
                    p.max_fanout,
                )
            )
        )
```

AHBN strategy consumption shown by the same command:

```text
        if self.adaptive_fanout:
            return max(
                1,
                int(node.control.fanout),
            )

        if mode == "gossip":
            self._gossip.fanout = fanout
            targets = self._gossip.select_targets(
                node,
                message,
                simulator,
            )

        elif mode == "cluster":
            self._cluster.fanout = fanout
            targets = self._cluster.select_targets(
                node,
                message,
                simulator,
            )
```

Gossip physical-neighbour bound shown by the same command:

```text
        candidates = [
            nbr_id
            for nbr_id in node.neighbors
            if nbr_id != node.node_id
            and nbr_id in simulator.nodes
            and simulator.nodes[nbr_id].is_active
        ]

        if not candidates:
            return []

        k = min(
            int(self.fanout),
            len(candidates),
        )

        return simulator.rng.sample(
            candidates,
            k,
        )
```

The targeted Stage 2 search identified:

```text
configs/stage2_parameter_sensitivity.yaml:18:runs_per_setting: 20
configs/stage2_parameter_sensitivity.yaml:69:  dense:
configs/stage2_parameter_sensitivity.yaml:92:  bottleneck:
configs/stage2_parameter_sensitivity.yaml:98:    ba_m: 3
configs/stage2_parameter_sensitivity.yaml:114:  churn:
scripts/run_stage2_sensitivity.py:157:    if runs_per_setting < 1:
scripts/run_stage2_sensitivity.py:168:    total_runs = total_settings * runs_per_setting
scripts/run_stage2_sensitivity.py:187:                for run_idx in range(runs_per_setting):
```

The second targeted command showed the Stage 2 configuration uses base seed 42 and 20 runs per setting, with existing `dense`, `bottleneck`, and `churn` scenarios. Its bottleneck BA scenario uses `ba_m: 3`. It also showed the Stage 2 runner derives seeds as `base_seed + run_idx`, invokes canonical AHBN through `run_single`, and already collects the required primary metrics plus trace-derived controller fanout statistics.

The bounded Structured implementation showed an AHBN forwarding budget and returns no more than that budget:

```text
        budget = max(
            1,
            int(self.fanout),
        )

        return selected[:budget]
```

The second command then failed with this exact terminal output:

```text
zsh:5: no matches found: scripts/*fanout6*
```

### Genuine Stop Condition

The shell rejected the unmatched validation-artifact glob before `ls` could execute. This was a command-execution failure. Per the failure / unexpected-result rule, the corrected inspection stopped immediately. No retry, speculative fix, V2 creation, or validation run was attempted.

### Files Created or Modified Before Stopping

- `docs/fanout6_result.md` updated with the corrected V1 audit trail and exact failure.

No validation script, validation configuration, raw results CSV, adaptive trace CSV, aggregate summary CSV, or validation log was created. Frozen production code and existing Stage 2/Stage 4 outputs remain unmodified.

## Resume After zsh Unmatched-Glob Stop

The previous stop was caused solely by zsh unmatched-glob behaviour. No implementation or experiment failure occurred. The failing wildcard command remains preserved above and was replaced by a safe bounded file-existence check. All valid V1 findings obtained before that stop remain valid.

### Commands Executed

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
find scripts configs outputs -maxdepth 2 -type f -name '*fanout6*' -print
sed -n '1,125p' run_batch.py
sed -n '330,445p' ahbn/simulator.py
sed -n '1,130p' ahbn/trace.py
sed -n '1,155p' ahbn/metrics.py
```

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
rg -n "class AdaptiveTraceRow|adaptive_trace_rows|return summary|simulator.run" ahbn run_batch.py
sed -n '125,235p' run_batch.py
sed -n '1,90p' ahbn/utils.py
sed -n '180,285p' ahbn/strategies/cluster.py
sed -n '445,635p' ahbn/simulator.py
```

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
sed -n '635,770p' ahbn/simulator.py
sed -n '85,175p' ahbn/utils.py
sed -n '1,120p' ahbn/node.py
sed -n '1,130p' ahbn/topology.py
```

```sh
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' scripts/validate_fanout6_amendment.py 2>&1 | tee outputs/fanout6_validation.log
```

After the failure, the still-running pipeline was interrupted with Ctrl-C. A bounded file check was then executed to identify files created before stopping:

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
find scripts configs outputs -maxdepth 2 -type f -name '*fanout6*' -print
```

### Relevant Terminal Output

The initial safe `find` returned no output, which was expected and confirmed that no pre-existing `fanout6` validation artifact existed.

One bounded inspection path did not exist:

```text
sed: ahbn/trace.py: No such file or directory
```

This was an inspection-path mistake, not missing required trace functionality. The same command already showed adaptive trace creation in `ahbn/simulator.py`; the next targeted command located `AdaptiveTraceRow` in `ahbn/utils.py` and confirmed that `run_single()` returns `sim.adaptive_trace_rows` when tracing is enabled.

The remaining bounded inspection confirmed that `run_single()` accepts an in-memory copied AHBN configuration, creates the canonical controller/strategy, runs the simulator, and returns the existing primary metrics and adaptive trace. No frozen source modification is required.

### V1 Disposition

PASS

Confirmed:

- frozen `min_fanout=2` and `max_fanout=4`;
- controller fanout is integer-rounded and clamped;
- changing only the copied validation configuration's `max_fanout` leaves EWMA, weights, kappa, beta, mode threshold/logic, comparator strategies, and topology generation unchanged;
- Gossip forwarding uses `min(fanout, len(active physical candidates))`;
- bounded Structured execution does not exceed its assigned AHBN forwarding budget;
- existing Stage 2 dense, bottleneck, and churn definitions, base seed 42, 20 repetitions, and bottleneck BA(m=3) can be reused.

### V2 Artifact Created

- `scripts/validate_fanout6_amendment.py`

The script schedules only 3 scenarios x 2 max-fanout bounds x 20 seeds = 120 AHBN runs, uses the existing Stage 2 definitions, changes only `max_fanout` in an in-memory copied configuration, records the existing adaptive trace, adds forwarding-bound diagnostics, and computes Student-t 95% confidence intervals.

### Genuine Execution Failure

The required validation command produced:

```text
tee: outputs/fanout6_validation.log: Operation not permitted
```

The pipeline was immediately interrupted and returned exit code 130. This is a genuine filesystem-permission failure affecting the required terminal log. Per the stop rule, the command was not retried and outputs were not redirected elsewhere.

The post-stop bounded check returned exactly:

```text
scripts/validate_fanout6_amendment.py
```

Therefore no raw results CSV, adaptive trace CSV, aggregate summary CSV, or terminal log was created, and no validation run was recorded as completed.

### Status at Stop

- V1: PASS
- Intended validation runs: 120
- Completed validation runs: 0 recorded
- Scientific decision: not made
- Frozen canonical AHBN modified: No
- Existing Stage 2 outputs modified: No
- Existing Stage 4 outputs modified: No
- Comparator implementations modified: No

## Diagnostic — Why Fanout 6 Was Not Reached

This was an inspection-only diagnostic. It used only the canonical controller source and the existing `outputs/csv/fanout6_validation_trace.csv`. No simulation was rerun, no result was modified, and no parameter or canonical implementation was changed.

### Commands Used

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
sed -n '78,90p' ahbn/control.py
sed -n '287,319p' ahbn/control.py
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' -c 'import pandas as pd; p="outputs/csv/fanout6_validation_trace.csv"; d=pd.read_csv(p); g=d.groupby(["validation_scenario","validation_max_fanout"]); print("rows",len(d)); print("groups");
for (s,m),x in g:
 w=x.weight.astype(float); raw=2.0+w*(float(m)-2.0); rounded=raw.clip(2.0,float(m)).round().astype(int); print(f"{s}, max={int(m)}, n={len(x)}, weight_min={w.min():.12f}, weight_mean={w.mean():.12f}, weight_max={w.max():.12f}, raw_min={raw.min():.12f}, raw_mean={raw.mean():.12f}, raw_max={raw.max():.12f}, rounded_min={rounded.min()}, rounded_max={rounded.max()}, trace_fanout_min={int(x.fanout.min())}, trace_fanout_max={int(x.fanout.max())}")'
```

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' -c 'print("Python round tie checks:", [(x, round(x)) for x in (2.5,3.5,4.5,5.5)]); print("max_fanout=4 (raw=2+2w): fanout 2: 0<=w<=0.25; fanout 3: 0.25<w<0.75; fanout 4: 0.75<=w<=1"); print("max_fanout=6 (raw=2+4w): fanout 2: 0<=w<=0.125; fanout 3: 0.125<w<0.375; fanout 4: 0.375<=w<=0.625; fanout 5: 0.625<w<0.875; fanout 6: 0.875<=w<=1")'
```

### Exact Fanout Equation

The canonical values are `min_fanout=2`, frozen `max_fanout=4`, and `beta=1.0`. Let controller weight be `w`, already clamped to `[0,1]`. The implementation is:

```text
span = max_fanout - min_fanout
raw_fanout = min_fanout + beta * w * span
fanout = int(round(clamp(raw_fanout, min_fanout, max_fanout)))
```

Thus, for this diagnostic:

```text
max_fanout=4: raw_fanout = 2 + 2w
max_fanout=6: raw_fanout = 2 + 4w
```

Python `round()` uses ties-to-even. The observed tie checks were `round(2.5)=2`, `round(3.5)=4`, `round(4.5)=4`, and `round(5.5)=6`.

### Observed Weight and Implied Fanout Ranges

The trace contained 585,577 controller updates.

| Scenario | max_fanout | Trace rows | Weight min | Weight mean | Weight max | Continuous fanout min | Continuous fanout mean | Continuous fanout max | Rounded range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bottleneck | 4 | 4,943 | 0.380086288317 | 0.489865750359 | 0.555582295127 | 2.760172576634 | 2.979731500717 | 3.111164590254 | 3–3 |
| bottleneck | 6 | 6,832 | 0.380572725663 | 0.482445167401 | 0.555573001420 | 3.522290902651 | 3.929780669603 | 4.222292005681 | 4–4 |
| churn | 4 | 279,630 | 0.467419003581 | 0.577011159527 | 0.612184455471 | 2.934838007163 | 3.154022319054 | 3.224368910942 | 3–3 |
| churn | 6 | 280,720 | 0.464833137892 | 0.571655303427 | 0.612187817217 | 3.859332551566 | 4.286621213710 | 4.448751268869 | 4–4 |
| dense | 4 | 5,609 | 0.390403058818 | 0.491819240270 | 0.539049492391 | 2.780806117637 | 2.983638480540 | 3.078098984781 | 3–3 |
| dense | 6 | 7,843 | 0.390128465309 | 0.482108145815 | 0.539049712033 | 3.560513861234 | 3.928432583258 | 4.156198848133 | 4–4 |

### Analytical Integer-Fanout Thresholds

For `max_fanout=4`, where raw fanout is `2+2w`:

| Integer fanout | Controller-weight interval |
|---:|---:|
| 2 | `0 <= w <= 0.25` |
| 3 | `0.25 < w < 0.75` |
| 4 | `0.75 <= w <= 1` |

For `max_fanout=6`, where raw fanout is `2+4w`:

| Integer fanout | Controller-weight interval |
|---:|---:|
| 2 | `0 <= w <= 0.125` |
| 3 | `0.125 < w < 0.375` |
| 4 | `0.375 <= w <= 0.625` |
| 5 | `0.625 < w < 0.875` |
| 6 | `0.875 <= w <= 1` |

Because of ties-to-even rounding, fanout 5 requires weight strictly greater than `0.625`; at exactly `0.625`, raw fanout is `4.5` and rounds to 4. Fanout 6 requires weight at least `0.875`; at exactly `0.875`, raw fanout is `5.5` and rounds to 6.

The maximum weight observed under `max_fanout=6` was `0.612187817217` in churn. It was below the fanout-5 threshold and far below the fanout-6 threshold. Therefore all `[2,6]` trace rows correctly rounded to fanout 4.

### Explanation of 3 Versus 4

The result is principally the same controller-weight operating region mapped across a wider output span. The two validations do not contain numerically identical weight sequences because their different forwarding fanouts alter later network observations, but both operate around a broadly similar middle-weight region.

At the representative midpoint `w=0.5`:

```text
[2,4]: 2 + 2(0.5) = 3 -> fanout 3
[2,6]: 2 + 4(0.5) = 4 -> fanout 4
```

All observed `[2,4]` weights lay inside its fanout-3 interval `(0.25,0.75)`. All observed `[2,6]` weights lay inside its fanout-4 interval `[0.375,0.625]`. The 3-versus-4 result is therefore explained by rescaling essentially the same normal controller-weight region across the wider `[2,6]` fanout interval, not by demonstrated excursions into a high-weight recovery region.

### Scientific Interpretation

1. `max_fanout=6` is technically reachable by the current mathematical controller: it is selected when `w >= 0.875` after the existing score-to-sigmoid computation.
2. It was not empirically reachable under the completed dense, bottleneck, and churn traces. Neither fanout 5 nor 6 occurred; the largest observed `[2,6]` weight was approximately `0.61219`.
3. In the existing evidence, changing `[2,4]` to `[2,6]` mainly rescaled the normal operating fanout from 3 to 4. It did not add demonstrated adaptive recovery behaviour above fanout 4.
4. Under these fixed scenarios and observed controller states, making fanout 5 or 6 more frequently reachable would require higher controller weights. Producing those higher weights would require changing another frozen controller parameter/equation or changing experimental conditions. Neither is authorized or scientifically tested here.
5. The existing evidence is insufficient to amend the canonical range to `[2,6]`: the required empirical reachability and decrease-after-use evidence was not demonstrated.

### Diagnostic Recommendation

RETAIN [2,4]

This recommendation is based on insufficient amendment evidence, not on optimization or a claim that six is mathematically impossible.

### Frozen Status

```text
Simulations rerun: No
Canonical AHBN modified: No
Validation results modified: No
Other frozen parameters modified: No
```

## Corrected Execution Using outputs/logs

The first validation execution did not start successfully because the terminal logging target `outputs/fanout6_validation.log` was not writable. No simulation result was recorded. The validation was therefore rerun using the established writable project log directory `outputs/logs/`.

### Directory and Writable-Path Checks

Exact initial command:

```sh
cd '/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6'
ls -ld outputs outputs/logs outputs/csv
probe_path='outputs/logs/.fanout6_write_probe'
: > "$probe_path"
ls -l "$probe_path"
rm "$probe_path"
```

Terminal output:

```text
drwxr-xr-x@  8 wwiras  staff  256 Aug 21 08:49 outputs
drwxr-xr-x@ 29 wwiras  staff  928 Aug 20 12:37 outputs/csv
drwxr-xr-x@ 11 wwiras  staff  352 Aug 21 08:35 outputs/logs
zsh:4: operation not permitted: outputs/logs/.fanout6_write_probe
ls: outputs/logs/.fanout6_write_probe: No such file or directory
rm: outputs/logs/.fanout6_write_probe: No such file or directory
```

This was workspace sandbox isolation, not a project permission or AHBN failure. The explicitly authorized probe was repeated through the approved project-path execution mechanism:

```sh
probe_path='outputs/logs/.fanout6_write_probe'
: > "$probe_path"
ls -l "$probe_path"
rm "$probe_path"
```

Terminal output:

```text
-rw-r--r--@ 1 wwiras  staff  0 Aug 21 10:18 outputs/logs/.fanout6_write_probe
```

The probe was removed by the same command. This confirmed `outputs/logs` was writable without permission or ownership changes.

### Corrected Validation Command

```sh
set -o pipefail
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' scripts/validate_fanout6_amendment.py 2>&1 | tee outputs/logs/fanout6_validation.log
```

Python/pipeline exit code:

```text
1
```

### Run Count and Output Files

```text
Expected runs: 120
Completed runs: 120
Scenarios: dense, bottleneck, churn
max_fanout settings: 4, 6
Seeds: 42 through 61
Repetitions per Scenario x max_fanout setting: 20
```

Files created by the corrected execution:

- `outputs/logs/fanout6_validation.log`
- `outputs/csv/fanout6_validation_raw.csv`
- `outputs/csv/fanout6_validation_trace.csv`
- `outputs/csv/fanout6_validation_summary.csv`

### Complete Relevant Terminal Summary and Failure

The complete per-run terminal output for runs `[001/120]` through `[120/120]` is retained verbatim in `outputs/logs/fanout6_validation.log`. Each setting completed 20 seeds. The final terminal output was:

```text
VALIDATION SUMMARY
Expected/completed runs: 120/120
Fanout >4 reached with max_fanout=6: False
Fanout subsequently decreased: False
Permanently at fanout 6: False
Effective forwarding respects physical degree: True
Effective forwarding respects requested budget: True
Raw results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/fanout6_validation_raw.csv
Adaptive trace: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/fanout6_validation_trace.csv
Aggregate summary: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/fanout6_validation_summary.csv
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_fanout6_amendment.py", line 242, in <module>
    main()
    ~~~~^^
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_fanout6_amendment.py", line 238, in main
    raise RuntimeError("Required max_fanout=6 adaptive behaviour was not demonstrated")
RuntimeError: Required max_fanout=6 adaptive behaviour was not demonstrated
```

Across the printed per-run diagnostics, the requested fanout range was `3-3` for every `max_fanout=4` run and `4-4` for every `max_fanout=6` run. Every run reported `physical_excess=0`.

### Genuine Scientific Stop Condition

The 120-run execution and CSV aggregation completed, but the required adaptive evidence did not. With `max_fanout=6`, fanout greater than four was never observed, and consequently a subsequent decrease from above four could not be demonstrated. This is a genuine scientifically unexpected validation result under the stated decision criteria. The validation therefore stopped without modifying or retrying the script, controller, scenarios, or algorithm.

### Status at This Stop

- V1: PASS
- Validation runs: 120/120 completed
- Fanout greater than 4 reached: No
- Fanout subsequently decreased from above 4: No
- Permanently at fanout 6: No
- Effective forwarding respected physical degree: Yes
- Effective forwarding respected requested budget: Yes
- Final scientific decision: not made at this stop, pending user inspection of the unexpected adaptive result
- Frozen canonical AHBN modified: No
- Existing Stage 2 outputs modified: No
- Existing Stage 4 outputs modified: No
- Comparator implementations modified: No
