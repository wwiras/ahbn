# Stage 1.5 — Adaptive Trace Integration for Exp07–Exp09

This change adds AHBN adaptive-trace collection to Exp07, Exp08, and Exp09 without changing the canonical controller.

## Changes

- `exp07`, `exp08`, and `exp09` now enable adaptive tracing only for the `ahbn` strategy.
- Each experiment returns both result rows and adaptive-trace rows, matching the pattern already used by Exp10–Exp12.
- `main()` saves timestamped adaptive-trace CSV files:
  - `exp07_adaptive_trace_<timestamp>.csv`
  - `exp08_adaptive_trace_<timestamp>.csv`
  - `exp09_adaptive_trace_<timestamp>.csv`
- `run_single()` accepts an optional `scenario_tag` used only as trace metadata.
- Scenario tags identify the swept setting:
  - Exp07: `fanout=<value>`
  - Exp08: `ch_overload_factor=<value>`
  - Exp09: `edge_prob=<value>`

## Controller integrity

No changes were made to:

- `ahbn/control.py`
- `ahbn/simulator.py` canonical update equations
- EWMA logic
- score equation
- sigmoid mapping
- mode decision
- fanout equation
- forwarding strategy execution

A direct tracing invariance check produced identical delivery ratio, propagation delay, duplicates, and total forwards with tracing disabled versus enabled for the same AHBN run.

## Important Exp07 note

The existing Exp07 configuration sweeps a configured `fanout`, but canonical AHBN uses adaptive fanout selected by `node.control.fanout`. The root YAML field `adaptive_fanout: false` is currently not consumed by `build_ahbn_strategy()`, which explicitly constructs AHBN with `adaptive_fanout=True`. This Stage 1.5 change intentionally does not alter that experimental behavior; Exp07's scientific interpretation should be resolved before final Stage 2/3 runs.
