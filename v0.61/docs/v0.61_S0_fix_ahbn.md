# S0 — AHBN Repair / Verify Protocol Semantics

## Scope

Targeted S0 audit of AHBN controller, sensing, mode/fanout decision, runtime
dispatch, sender exclusion, and comparator isolation. No S1 work and no Exp07,
Exp08, or Exp09 experiment was run.

## Reference

- Frozen AHBN reference: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
  (the repository's on-disk directory corresponding to v0.60; there is no
  directory literally named `v0.60`)
- Active implementation: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.61`
- Relevant v0.61 commit before repair:
  `860f0f1ac68f7b87b4caac42fdd09c34b0f48499`

## Environment

- Project root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.61`
- Interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Virtual environment: `/Users/wwiras/Documents/src/AHBNProj/venv0.6`
- Branch: `main`
- Initial worktree: clean

## Commands Executed

The audit used only `sed`, `rg`, `find`, `diff`, `shasum`, `git status`,
`git log`, `git diff`, import/compile checks, and small inline deterministic
Python probes. The significant commands and outputs are reproduced under
Terminal Output. No batch or final experiment command was executed.

## v0.60 → v0.61 Differential Audit

The frozen directory is named `v0.6`; it is treated as v0.60 throughout this
report.

| Area | Before repair classification | Evidence |
|---|---|---|
| `ahbn/control.py` | IDENTICAL | SHA-1 `9d5b972bbdc0694f62ea83efbddd6d822457e084` in both trees |
| `ahbn/strategies/ahbn.py` | IDENTICAL, then AHBN-RELEVANT REPAIR | SHA-1 `d6593e6cc4f2f9c96a0fd746493e3d0f14b5b713` before repair |
| `ahbn/simulator.py` | IDENTICAL, then AHBN-RELEVANT REPAIR | SHA-1 `8f5ef4d083c766d5de524898d45aa19e92a99986` before repair |
| `run_batch.py` AHBN builders (lines 1–59) | IDENTICAL | empty focused diff |
| v0.61 DC-SoC differences | EXPECTED CLEANUP / UNRELATED | confined to `strategies/dcsoc.py`; no AHBN import or dispatch |
| v0.61 Gossip constructor difference | EXPECTED CLEANUP / UNRELATED before repair | removal of standalone `fanout=None`; AHBN always supplies an integer |

### AHBN-relevant mismatch 1 — sender exclusion

- Files/functions: `Simulator.handle_receive`,
  `AHBNStrategy.select_targets`, `GossipStrategy.select_targets`, and
  `ClusterStrategy.select_targets`.
- Frozen ControlSim behavior: v0.6 and the initial v0.61 copy both retained
  `src_id` in receive events but did not propagate it into target selection.
- Earlier canonical evidence: v0.1 `Node.gossip_targets(sender)` and
  `Node.cluster_targets(sender, topology)` explicitly removed `sender` before
  selection.
- Scientific consequence: AHBN could spend part of its fanout immediately
  returning a message to its previous hop, increasing duplicates and reducing
  useful reach.
- Required repair: yes. The protocol explicitly requires sender exclusion
  before applying fanout.

### AHBN-relevant mismatch 2 — runtime bound enforcement

- File/function: `AHBNStrategy._get_effective_fanout`.
- Frozen and initial v0.61 behavior: controller output was clamped, but the
  execution layer accepted any integer in `node.control.fanout` and only
  enforced a minimum of one.
- Scientific consequence: malformed or overridden runtime state could forward
  outside canonical `[2,4]`.
- Required repair: yes. The protocol requires the controller and actual
  dissemination layer both to enforce `[2,4]`.

## Findings Before Repair

- Controller implementation, state initialization, four observations, EWMA
  equation, score equation, sigmoid, threshold, and fanout calculation were
  identical to the frozen v0.6 files.
- `alpha=0.30`; centers are all `0.50`; weights are `(-1,+1,-1,+1)`;
  `kappa=1`, `beta=1`, threshold `0.50`; controller bounds are `[2,4]`.
- `run_single`/`run_one` instantiate `AHBNStrategy` and `AHBNController` only
  for `strategy_name == "ahbn"`.
- Exp08 and Exp09 AHBN blocks contain the canonical parameter set. Exp07 has
  no AHBN override block and therefore uses canonical builder defaults.
- No AHBN behavior conditional on Exp07/08/09, experiment name, seed, density,
  or topology was found. Environment overload legitimately affects measured
  latency/utilization in the shared simulator.
- AHBN imports only its controller and the intentional Gossip/Structured
  execution primitives. It neither imports nor calls DC-SoC logic.
- Mandatory sender exclusion and execution-layer fanout enforcement were
  missing.

## Repairs Performed

All source changes are restoration/enforcement of frozen AHBN semantics:

1. `Simulator.handle_receive` passes the immediate `src_id` only on the
   controller-backed AHBN path.
2. `AHBNStrategy.select_targets` forwards that exclusion to its selected
   dissemination primitive.
3. Gossip and Structured primitives accept an optional excluded target and
   remove it while constructing candidates, before sampling/allocation.
   Standalone comparator calls omit the option and retain existing behavior.
4. `AHBNStrategy._get_effective_fanout` clamps both adaptive and fallback
   budgets to `[2,4]`.

No controller parameter, equation, observation, threshold, or comparator
policy was changed.

## Verification Evidence

- Controller probe: clamped observations updated from zero to
  `(0.30, 0.15, 0.00, 0.30)` under alpha `0.30`; computed score
  `0.15000000000000002`, weight `0.5374298453437496`, gossip, fanout 3.
- Gossip sender probe: eligible `{2,3,4}` produced three targets and excluded
  sender 0 before sampling.
- Structured sender probe: sender 0 was removed before budget two; result was
  `[4,2]`, preserving gateway-first allocation.
- Defensive bound probe: runtime values 99 and -99 became 4 and 2.
- Actual receive-path probe: `src_id=0`, receiver 1 sent to
  `[(1,3),(1,2),(1,4)]`; sender 0 was absent.
- Syntax/import check passed with bytecode cache redirected to `/tmp`.
- Exp07/08/09 configuration invariance probe passed.
- Exp08/Exp09 factory wiring probe passed without running simulations:
  `AHBNStrategy` + `AHBNController`, bounds `(2,4)`.
- `git diff --check` passed.

## AHBN Semantic Checklist

| ID | Result | Evidence |
|---|---|---|
| AHBN-01 | PASS | `run_one.py`/`run_batch.py` → controller + AHBN strategy → simulator |
| AHBN-02 | PASS | identical controller/simulator sensing code; deterministic EWMA probe |
| AHBN-03 | PASS | identical score/weight code and canonical parameters |
| AHBN-04 | PASS | weight `>=0.5` Gossip, otherwise Structured; bounded weight-driven fanout |
| AHBN-05 | PASS | `min_fanout=2` in defaults/configs |
| AHBN-06 | PASS | `max_fanout=4` in defaults/configs |
| AHBN-07 | PASS | node and strategy default fanout 3 |
| AHBN-08 | REPAIRED → PASS | controller clamp plus execution-layer `[2,4]` clamp |
| AHBN-09 | REPAIRED → PASS | immediate sender propagated and excluded before selection/allocation |
| AHBN-10 | PASS | no Exp07-specific AHBN tuning |
| AHBN-11 | PASS | canonical Exp08 AHBN block; no conditional tuning |
| AHBN-12 | PASS | canonical Exp09 AHBN block; no conditional tuning |
| AHBN-13 | PASS | no DC-SoC import/call in AHBN path |
| AHBN-14 | PASS | Gossip is an intentional execution primitive; no standalone comparator control logic leaks |
| AHBN-15 | PASS | Structured is an intentional execution primitive; no standalone comparator control logic leaks |
| AHBN-16 | PASS | actual factory and receive-path probes exercised controller-backed AHBN dispatch |

## Files Modified

Source files:

- `ahbn/simulator.py`
- `ahbn/strategies/ahbn.py`
- `ahbn/strategies/gossip.py`
- `ahbn/strategies/cluster.py`

Test/probe files: NONE (probes were inline and deterministic).

Documentation files:

- `docs/S0_fix_ahbn.md`

## Git Diff Summary

Before documentation was added:

```text
v0.61/ahbn/simulator.py          | 21 ++++++++++++++++-----
v0.61/ahbn/strategies/ahbn.py    | 15 +++++++++------
v0.61/ahbn/strategies/cluster.py | 12 +++++++++++-
v0.61/ahbn/strategies/gossip.py  |  6 ++++--
4 files changed, 40 insertions(+), 14 deletions(-)
```

## Final S0 AHBN Gate

All 16 mandatory evidence items pass. The only implementation changes restore
sender exclusion and enforce the already-canonical forwarding bounds at the
execution layer.

**S0 AHBN: REPAIRED → PASS**

## Terminal Output

Initial identity and clean-tree evidence:

```text
$ git status --short
(no output)
$ git rev-parse HEAD
860f0f1ac68f7b87b4caac42fdd09c34b0f48499
$ git log -1 --oneline
860f0f1 v0.61 fresh start with DC_SoC faithful last freeze
```

Reference path inspection:

```text
$ ls -la ..
...
drwxr-xr-x ... v0.6
drwxrwxr-x ... v0.61
$ rg --files ../v0.60
rg: ../v0.60: IO error ... No such file or directory
```

Pre-repair checksums:

```text
9d5b972bbdc0694f62ea83efbddd6d822457e084  ../v0.6/ahbn/control.py
9d5b972bbdc0694f62ea83efbddd6d822457e084  ahbn/control.py
d6593e6cc4f2f9c96a0fd746493e3d0f14b5b713  ../v0.6/ahbn/strategies/ahbn.py
d6593e6cc4f2f9c96a0fd746493e3d0f14b5b713  ahbn/strategies/ahbn.py
8f5ef4d083c766d5de524898d45aa19e92a99986  ../v0.6/ahbn/simulator.py
8f5ef4d083c766d5de524898d45aa19e92a99986  ahbn/simulator.py
```

The first compile attempt could not write bytecode because the active tree was
outside the tool's writable root. This was an environment failure, not a
syntax result, and is retained here as required:

```text
$ /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m compileall -q ahbn run_one.py run_batch.py
*** Error compiling 'ahbn/control.py'...
PermissionError: [Errno 1] Operation not permitted: 'ahbn/__pycache__'
... same PermissionError for the remaining requested modules ...
```

Corrected non-writing syntax check:

```text
$ PYTHONPYCACHEPREFIX=/tmp/ahbn_s0_pycache .../venv0.6/bin/python -m compileall -q ahbn run_one.py run_batch.py
syntax_import_check: PASS
```

Targeted controller and selection probes:

```text
controller_probe: PASS {'d_hat': 0.3, 'l_hat': 0.15, 'u_hat': 0.0,
'c_hat': 0.3, 'score': 0.15000000000000002,
'weight': 0.5374298453437496, 'mode': 'gossip', 'fanout': 3}
gossip_sender_and_bounds_probe: PASS [3, 2, 4]
structured_sender_before_budget_probe: PASS [4, 2]
S0_TARGETED_PROBES: PASS
```

The first runtime probe omitted metric registration in its test fixture and
therefore stopped before selection. The source was not changed in response:

```text
KeyError: 'runtime'
```

After registering the synthetic message, the same path passed:

```text
runtime_sender_propagation_probe: PASS [(1, 3), (1, 2), (1, 4)]
exp07_fanout.yaml: canonical AHBN parameters PASS
exp08_ch_bottleneck.yaml: canonical AHBN parameters PASS
exp09_dense_topology.yaml: canonical AHBN parameters PASS
configuration_invariance_probe: PASS
```

An attempted helper import used a nonexistent function name and made no source
change:

```text
ImportError: cannot import name 'build_simulation' from 'run_batch'
```

The corrected non-running factory probe used `build_simulation_from_config`:

```text
configs/exp08_ch_bottleneck.yaml AHBN runtime wiring: PASS
configs/exp09_dense_topology.yaml AHBN runtime wiring: PASS
runtime_wiring_probe: PASS
```

Isolation searches:

```text
$ rg -n '^from ahbn\.strategies|^import ahbn\.strategies' ...
ahbn/strategies/ahbn.py:7:from ahbn.strategies.base import ForwardingStrategy
ahbn/strategies/ahbn.py:8:from ahbn.strategies.cluster import ClusterStrategy
ahbn/strategies/ahbn.py:9:from ahbn.strategies.gossip import GossipStrategy
ahbn/simulator.py:13:from ahbn.strategies.base import ForwardingStrategy

$ rg -n -i 'if .*exp0[789]|if .*experiment|if .*seed|if .*density|if .*overload|if .*topology' ahbn/strategies/ahbn.py ahbn/control.py ahbn/simulator.py
ahbn/simulator.py:141:        if dst.is_overloaded:
ahbn/simulator.py:333:        if node.is_overloaded:
```

The two overload matches are environment observation handling, not
experiment-specific controller tuning.

Final compact terminal summary:

```text
============================================================
S0 — AHBN PROTOCOL SEMANTICS
============================================================

Reference:
  v0.60 frozen AHBN (on disk: v0.6)

Active:
  v0.61

Controller semantics:             PASS
EWMA equation:                    PASS
Controller equation:              PASS
Mode-selection rule:              PASS
Fanout bounds [2,4]:              PASS
Sender exclusion:                 PASS
Experiment-specific tuning:       NONE
DC-SoC leakage into AHBN:         NONE
Gossip leakage into AHBN:         NONE
Structured leakage into AHBN:     NONE

Source repairs performed:         YES
Files modified:                   ahbn/simulator.py,
                                  ahbn/strategies/ahbn.py,
                                  ahbn/strategies/gossip.py,
                                  ahbn/strategies/cluster.py,
                                  docs/S0_fix_ahbn.md

Documentation:
  docs/S0_fix_ahbn.md

FINAL:
  S0 AHBN REPAIRED → PASS
============================================================
```
