# Stage 4 — Exp08 CH Overload

## Experiment checklist

- [x] **E0 — Configuration inspection/freeze:** PASS after the documented E0.1 minimal configuration reconciliation
- [x] **E1 — Validate CH-overload injection:** PASS
- [x] **E2 — Run Gossip:** PASS
- [x] **E3 — Run Structured:** PASS
- [x] **E4 — Run DC-SoC:** PASS; the pre-execution diagnostic deviation is documented and excluded from the official dataset
- [x] **E5 — Run AHBN:** PASS
- [x] **E6 — Validate AHBN adaptive traces:** PASS
- [x] **E7 — Aggregate 20 runs (mean and 95% CI):** PASS
- [x] **E8 — Plot four algorithms:** PASS
- [x] **E9 — Scientific interpretation:** PASS

**Exp08 final status: COMPLETE — E0 through E9 PASS.** Exp09 was not started.

## E0 — Configuration Inspection / Freeze

### Purpose

Inspect the existing Exp08 contract, trace its effective runtime configuration, verify frozen comparator parameters, and stop before E1 if any scientific ambiguity remains. No algorithm or experiment behavior was changed.

### Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python \
  scripts/inspect_exp08_e0.py
```

Exact interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`

### Repository state inspected

- Root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
- Exp08 configuration: `configs/exp08_ch_bottleneck.yaml`
- Frozen DC-SoC reference: `configs/stage3_dcsoc.yaml`
- Construction/runner paths: `run_one.py`, `run_batch.py`
- Runtime: `ahbn/config.py`, `ahbn/topology.py`, `ahbn/simulator.py`, `ahbn/node.py`, `ahbn/control.py`, `ahbn/metrics.py`
- Strategies: `ahbn/strategies/gossip.py`, `cluster.py`, `dcsoc.py`, `ahbn.py`
- Existing overload/failure support and prior Stage 3.5 freeze validation were also inspected.

### Experimental contract

Exp08 currently defines BA topology with 100 nodes, `m=3`, four static clusters, topology caching, base seed 42, 20 runs per setting (seeds 42–61), one message `m1` from fixed source node 0 at time 0, and event-queue exhaustion as termination. The timing convention is seconds: base delay 1.0 and per-send uniform jitter `[0.0, 0.2]`; default node processing delay is 0.0.

The independent-variable sweep is `ch_overload_factor = [1.0, 1.5, 2.0, 3.0]`. Other production-run arguments remain invariant within the Exp08 loops.

### Overload mechanism

`configs/exp08_ch_bottleneck.yaml:ch_overload_factor` is loaded by `ahbn.config.load_yaml_config`, iterated by `run_batch.exp08`, passed through `run_batch.run_single` into `Simulator.ch_overload_factor`, and consumed by `Simulator.send_message`. For a destination with `is_cluster_head=true`, it adds `base_delay * max(0, factor - 1)` to arrival delay. Thus the effective base component is multiplied by the factor. It does not change capacity, queue pressure, drops, availability, or `Node.is_overloaded`.

Targets are algorithm-specific. At seed 42: Gossip `[]`, Structured `[0,1,2,3]`, DC-SoC `[4]`, AHBN `[0,1,2,3]`. The physical overloaded set is not identical across algorithms.

### Comparator freeze

- Gossip is fixed fanout 3, non-adaptive, and controller-independent, but is absent from Exp08's configured strategy list.
- Structured uses round-robin static clusters, lowest-ID CHs, member-to-CH forwarding, and CH-to-members plus adjacent-CH gateways.
- DC-SoC uses frozen `eps=2.0`, `min_samples=3`, `fanout=3`, and `inter_fanout=1`; however, Exp08 does not state these values and reaches them through runner fallbacks.
- AHBN matches the frozen values (`alpha=.30`, centers `.50`, weights `-1,+1,-1,+1`, `kappa=1`, `beta=1`, threshold `.50`, fanout 2–4, default 3), with EWMA, adaptive score, and adaptive fanout active. No emergency failure/bottleneck override exists in the canonical controller/strategy; adding one was not attempted.

### Configuration-consumption checks

Overload, timing, workload, seed, canonical AHBN, and frozen DC-SoC effective values are consumed. The gate fails because the required four-comparator set is not configured, target fairness is not satisfied, DC-SoC values are implicit, and the requested emergency-override expectation differs from the frozen implementation.

### Controlled-variable checks

Topology is fixed per seed, configured strategies share workload/timing/seed pairing, parameters are frozen, and only the overload factor externally varies. No Exp08-specific tuning was found. The target semantics and comparator omission remain scientifically meaningful ambiguities.

### Terminal output

```text
========================================================================
STAGE 4 — FINAL COMPARATIVE EVALUATION
Exp08 — CH Overload
E0 — Configuration Inspection / Freeze
========================================================================

Repository:
  root                  : /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

Python:
  interpreter           : /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python

-----------------------------------------------------------------------
TOPOLOGY
-----------------------------------------------------------------------
Topology type           : BA
Node count              : 100
BA m / equivalent       : 3
Topology fixed params   : num_clusters=4; use_topology_cache=true
Seed policy             : base seed + run index; same seed reused across configured algorithms
Runs per setting        : 20
Exact seeds             : [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

-----------------------------------------------------------------------
WORKLOAD
-----------------------------------------------------------------------
Messages/run            : 1 (message_id=m1)
Source selection        : fixed node 0
Message timing          : injected at simulation clock 0.0
Termination             : event queue exhaustion; no duration/time limit
Same workload           : YES for every configured algorithm

-----------------------------------------------------------------------
TIMING
-----------------------------------------------------------------------
Base delay              : 1.0 seconds
Jitter                  : uniform [0.0, 0.2] seconds per send
Processing delay        : 0.0 seconds for default medium nodes
Units                   : seconds (current simulator convention)
Other timing fields     : CH extra = base_delay * max(0, factor - 1)

-----------------------------------------------------------------------
CH OVERLOAD
-----------------------------------------------------------------------
Config source           : /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/configs/exp08_ch_bottleneck.yaml
Configuration key       : ch_overload_factor
Overload levels         : [1.0, 1.5, 2.0, 3.0]

Physical meaning:
  Multiplicative arrival-delay factor for sends whose destination node is
  marked is_cluster_head; effective one-hop base component becomes
  base_delay * factor, before jitter and resource delays. It does not alter
  service capacity, queues, drops, availability, or Node.is_overloaded.

Runtime trace:
  configs/exp08_ch_bottleneck.yaml: ch_overload_factor
    -> ahbn.config.load_yaml_config
    -> run_batch.exp08 overload loop
    -> run_batch.run_single(ch_overload_factor=overload)
    -> Simulator.ch_overload_factor
    -> Simulator.send_message: if dst.is_cluster_head
    -> extra += base_delay * max(0, factor - 1)
Target-selection rule   : each algorithm's constructed nodes with is_cluster_head=true
Target node IDs (seed 42, representative first paired run):
  gossip                : []
  cluster               : [0, 1, 2, 3]
  dcsoc                 : [4]
  ahbn                  : [0, 1, 2, 3]

Same physical targets across algorithms:
  gossip                : []
  cluster               : [0, 1, 2, 3]
  dcsoc                 : [4]
  ahbn                  : [0, 1, 2, 3]
Overloaded physical node set is identical across algorithms: NO

-----------------------------------------------------------------------
COMPARATOR FREEZE
-----------------------------------------------------------------------

Gossip:
  fanout                : 3 (runner default; Exp08 has no fanout key)
  adaptive              : NO
  AHBN controller used  : NO
  present in Exp08      : NO
  status                : FAIL

Structured:
  cluster rule          : sorted node IDs assigned round-robin modulo 4
  CH rule               : lowest node ID in each cluster
  forwarding            : member->CH; CH->all local members + adjacent CH gateways
  status                : PASS

DC-SoC:
  DBSCAN eps            : 2.0 (fallback; section absent in Exp08)
  DBSCAN min_samples    : 3 (fallback; section absent in Exp08)
  CH rule               : highest physical degree; tie -> lowest node ID
  forwarding            : intra-cluster physical-neighbour fanout 3; CH gateway reserve 1
  runtime AHBN control  : NO
  Exp08-specific tuning : NO
  status                : PASS

AHBN:
  alpha                 : 0.30
  d0/l0/u0/c0           : 0.50/0.50/0.50/0.50
  w_d/w_l/w_u/w_c       : -1.0/+1.0/-1.0/+1.0
  kappa                 : 1.0
  beta                  : 1.0
  tau_mode              : 0.50 (config key: mode_threshold)
  fanout bounds         : [2, 4]
  default fanout        : 3
  EWMA                  : YES
  adaptive score/fanout : YES/YES
  emergency override    : NO (none present in canonical controller/strategy)
  Exp08-specific tuning : NO
  status                : PASS

-----------------------------------------------------------------------
CONFIGURATION CONSUMPTION
-----------------------------------------------------------------------
Overload config consumed          : PASS
Timing config consumed            : PASS
Workload config consumed          : PASS
Seed config consumed              : PASS
AHBN frozen config consumed       : PASS
DC-SoC frozen config consumed     : PASS

Unused / overridden fields:
  Exp08 has no dcsoc section; frozen values are reached only through hard-coded
  run_batch fallbacks. Gossip is absent from strategies. ch_overload_factor
  is consumed, but has no effect for Gossip because no CH is constructed.

-----------------------------------------------------------------------
CONTROLLED VARIABLES
-----------------------------------------------------------------------
Topology fixed per seed           : PASS
Same workload                     : PASS (for configured strategies)
Same timing model                 : PASS
Same seed pairing                 : PASS (for configured strategies)
Algorithm parameters frozen       : PASS
Only overload level externally varies: PASS

-----------------------------------------------------------------------
SCIENTIFIC FREEZE CHECK
-----------------------------------------------------------------------
Algorithm-specific tuning detected:
  NO
Unresolved configuration ambiguity:
  YES
Scientific-design modification required:
  YES

Discrepancies:
  1. Comparator set required=[gossip, cluster, dcsoc, ahbn], configured=['cluster', 'dcsoc', 'ahbn'].
  2. Overload targets are algorithm-specific: Gossip none; Structured/AHBN
     static heads; DC-SoC density-cluster heads.
  3. Frozen DC-SoC parameters are not explicit in Exp08 and rely on fallbacks.
  4. The requested emergency failure/bottleneck override is not present in
     the canonical AHBN controller/strategy; adding one would change design.

Smallest likely cause:
  Exp08 predates the four-comparator Stage 4 freeze and retained its original
  algorithm-specific CH-overload semantics and implicit DC-SoC defaults.

Issue classification:
  stale configuration; overload-target fairness; scientific-design ambiguity

========================================================================
E0 RESULT: FAIL
========================================================================

STOPPED BEFORE E1.

Please review the terminal output before any correction is applied.
```

### Result

E0 RESULT: FAIL

STOPPED BEFORE E1.

### Notes

No correction was applied. The inspection script is non-experimental: it constructs effective strategy state through the production `run_single` path but suppresses event-queue execution, so it does not generate comparative paper results. The frozen AHBN controller and all baseline implementations/configurations remain unchanged.

## E0.1 — Minimal Configuration Reconciliation

### Purpose

Reconcile the stale Exp08 configuration with the already-frozen Stage 4 comparator set, without changing the existing experimental mechanisms or any algorithm implementation.

### Previous E0 findings

- Gossip was absent from the configured comparator list.
- Frozen DC-SoC values were implicit runner fallbacks.
- The inspection incorrectly treated an emergency override as expected, although it is not part of the frozen controller.
- Architecture-specific CH-overload semantics required an explicit scientific interpretation.

### Changes applied

`configs/exp08_ch_bottleneck.yaml`:

- Added `gossip` to the comparator list.
- Added explicit frozen DC-SoC values: `eps=2.0`, `min_samples=3`, `fanout=3`, `inter_fanout=1`.

`scripts/inspect_exp08_e0.py`:

- Replaced identical-node targeting with the frozen architecture-specific CH-role semantics.
- Made explicit DC-SoC values and all four comparators required for PASS.
- Reports the emergency override as not part of the frozen controller and does not require it.

`docs/stage4_exp08.md`:

- Preserved the failed E0 record and appended this reconciliation record.

No frozen algorithm implementation files changed.

### Scientific design preserved

Unchanged: overload levels and mechanism; topology; workload; seed set and 20-run policy; timing and seconds convention; Gossip fanout/behavior; Structured behavior; DC-SoC effective values, clustering, CH selection, and forwarding; AHBN behavior and all frozen parameters. No tuning was performed.

The frozen interpretation is: Exp08 evaluates sensitivity to cluster-head bottleneck latency according to each architecture's own structural dependency. Structured, DC-SoC, and AHBN receive extra delay at their respective cluster-head nodes. Gossip has no cluster-head role and is the CH-independent dissemination reference.

### Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python \
  scripts/inspect_exp08_e0.py
```

### Terminal output

```text
========================================================================
STAGE 4 — FINAL COMPARATIVE EVALUATION
Exp08 — CH Overload
E0 — Configuration Inspection / Freeze
========================================================================

Repository:
  root                  : /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

Python:
  interpreter           : /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python

-----------------------------------------------------------------------
TOPOLOGY
-----------------------------------------------------------------------
Topology type           : BA
Node count              : 100
BA m / equivalent       : 3
Topology fixed params   : num_clusters=4; use_topology_cache=true
Seed policy             : base seed + run index; same seed reused across configured algorithms
Runs per setting        : 20
Exact seeds             : [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

-----------------------------------------------------------------------
COMPARATOR SET
-----------------------------------------------------------------------
Configured              : ['gossip', 'cluster', 'dcsoc', 'ahbn']
Required Stage 4 set    : ['gossip', 'cluster', 'dcsoc', 'ahbn']
Result                  : PASS

-----------------------------------------------------------------------
WORKLOAD
-----------------------------------------------------------------------
Messages/run            : 1 (message_id=m1)
Source selection        : fixed node 0
Message timing          : injected at simulation clock 0.0
Termination             : event queue exhaustion; no duration/time limit
Same workload           : YES for every configured algorithm

-----------------------------------------------------------------------
TIMING
-----------------------------------------------------------------------
Base delay              : 1.0 seconds
Jitter                  : uniform [0.0, 0.2] seconds per send
Processing delay        : 0.0 seconds for default medium nodes
Units                   : seconds (current simulator convention)
Other timing fields     : CH extra = base_delay * max(0, factor - 1)

-----------------------------------------------------------------------
CH OVERLOAD
-----------------------------------------------------------------------
Config source           : /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/configs/exp08_ch_bottleneck.yaml
Configuration key       : ch_overload_factor
Overload levels         : [1.0, 1.5, 2.0, 3.0]

Physical meaning:
  Multiplicative arrival-delay factor for sends whose destination node is
  marked is_cluster_head; effective one-hop base component becomes
  base_delay * factor, before jitter and resource delays. It does not alter
  service capacity, queues, drops, availability, or Node.is_overloaded.

Runtime trace:
  configs/exp08_ch_bottleneck.yaml: ch_overload_factor
    -> ahbn.config.load_yaml_config
    -> run_batch.exp08 overload loop
    -> run_batch.run_single(ch_overload_factor=overload)
    -> Simulator.ch_overload_factor
    -> Simulator.send_message: if dst.is_cluster_head
    -> extra += base_delay * max(0, factor - 1)
Target-selection rule   : algorithm-specific CH role
Target node IDs (seed 42, representative first paired run):
  gossip                : []
  cluster               : [0, 1, 2, 3]
  dcsoc                 : [4]
  ahbn                  : [0, 1, 2, 3]

CH OVERLOAD SEMANTICS:
  gossip                : []
  cluster               : [0, 1, 2, 3]
  dcsoc                 : [4]
  ahbn                  : [0, 1, 2, 3]
Target semantics:
  Gossip                : no CH role; no CH-specific target
  Structured            : own static cluster heads
  DC-SoC                : own DBSCAN-derived cluster heads
  AHBN                   : own static cluster heads
Identical physical targets required: NO
Reason                  : CH-role sensitivity experiment
Result                  : PASS

-----------------------------------------------------------------------
COMPARATOR FREEZE
-----------------------------------------------------------------------

Gossip:
  fanout                : 3 (runner default; Exp08 has no fanout key)
  adaptive              : NO
  AHBN controller used  : NO
  present in Exp08      : YES
  status                : PASS

Structured:
  cluster rule          : sorted node IDs assigned round-robin modulo 4
  CH rule               : lowest node ID in each cluster
  forwarding            : member->CH; CH->all local members + adjacent CH gateways
  status                : PASS

DC-SoC:
  DBSCAN eps            : 2.0
  DBSCAN min_samples    : 3
  CH rule               : highest physical degree; tie -> lowest node ID
  forwarding            : intra-cluster physical-neighbour fanout 3; CH gateway reserve 1
  runtime AHBN control  : NO
  Exp08-specific tuning : NO
  explicit in Exp08     : YES
  matches Stage 3.5     : YES
  status                : PASS

AHBN:
  alpha                 : 0.30
  d0/l0/u0/c0           : 0.50/0.50/0.50/0.50
  w_d/w_l/w_u/w_c       : -1.0/+1.0/-1.0/+1.0
  kappa                 : 1.0
  beta                  : 1.0
  tau_mode              : 0.50 (config key: mode_threshold)
  fanout bounds         : [2, 4]
  default fanout        : 3
  EWMA                  : YES
  adaptive score/fanout : YES/YES
  emergency override    : NOT PART OF FROZEN CONTROLLER
  Exp08-specific tuning : NO
  status                : PASS

-----------------------------------------------------------------------
CONFIGURATION CONSUMPTION
-----------------------------------------------------------------------
Overload config consumed          : PASS
Timing config consumed            : PASS
Workload config consumed          : PASS
Seed config consumed              : PASS
AHBN frozen config consumed       : PASS
DC-SoC frozen config consumed     : PASS

Unused / overridden fields:
  None identified. Gossip has no CH role, so the consumed CH-overload
  mechanism intentionally has no directly targeted Gossip nodes.

-----------------------------------------------------------------------
CONTROLLED VARIABLES
-----------------------------------------------------------------------
Topology fixed per seed           : PASS
Same workload                     : PASS (for configured strategies)
Same timing model                 : PASS
Same seed pairing                 : PASS (for configured strategies)
Algorithm parameters frozen       : PASS
Only overload level externally varies: PASS

-----------------------------------------------------------------------
SCIENTIFIC FREEZE CHECK
-----------------------------------------------------------------------
Algorithm-specific tuning detected:
  NO
Unresolved configuration ambiguity:
  NO
Scientific-design modification required:
  NO

========================================================================
E0 RESULT: PASS
========================================================================

E0.1 reconciliation complete.

Stage 4 Exp08 comparator configuration is now explicit and frozen.

Comparator set:
  Gossip / Structured / DC-SoC / AHBN

No algorithm-specific tuning detected.
No frozen controller/baseline implementation changed.
CH-overload semantics explicitly documented.

READY FOR:
E1 — Validate CH-overload injection
```

### Result

E0 RESULT: PASS

E0.1 reconciliation complete. Ready for E1; E1 was not executed.
## E1 — Validate CH-Overload Injection

### Purpose

Independently validate that the frozen Exp08 `ch_overload_factor` condition applies its added one-hop arrival latency only to each comparator's architecture-specific cluster-head/bottleneck role, activates from simulation start, preserves non-target nodes, and does not directly mutate forwarding or controller policy. This validation does not run the final comparative experiment.

### Files inspected

- `configs/exp08_ch_bottleneck.yaml`
- `docs/stage4_exp08.md`
- `scripts/inspect_exp08_e0.py`
- `run_batch.py`
- `ahbn/simulator.py`
- `ahbn/node.py`
- `ahbn/topology.py`
- `ahbn/cluster.py`
- `ahbn/control.py`
- `ahbn/strategies/gossip.py`
- `ahbn/strategies/cluster.py`
- `ahbn/strategies/dcsoc.py`
- `ahbn/strategies/ahbn.py`

### Validation script

Created `scripts/validate_exp08_e1.py`. It constructs paired normal (`ch_overload_factor=1.0`) and overloaded (`ch_overload_factor=3.0`) simulators through the production Exp08 construction path without executing the event queue or producing comparative measurements. It independently calculates expected Structured/AHBN heads from the round-robin/lowest-ID rule and DC-SoC heads from runtime DBSCAN membership plus the independently applied highest-physical-degree/tie-lowest-ID rule. Deterministic paired probes then validate the exact delay delta for every node and snapshot policy state before and after injection.

Exp08 has no later overload event trigger: `ch_overload_factor` is active from simulator construction/run start (`t=0`). The runtime parameter changes CH-destination arrival delay; it does not change `Node.processing_delay`, queue service capacity, `Node.is_overloaded`, or forwarding policy.

### Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_e1.py
```

Exact interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`

### Complete terminal output

```text
========================================================================
STAGE 4 — EXP08
E1 — Validate CH-Overload Injection
========================================================================

Configuration:
  Config              : configs/exp08_ch_bottleneck.yaml
  Python              : /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
  Comparators         : gossip, cluster, dcsoc, ahbn
  Seed                : 42
  Activation          : simulator construction / run start (t=0)
  Normal value        : ch_overload_factor=1.0
  Overloaded value    : ch_overload_factor=3.0

------------------------------------------------------------------------
GOSSIP
------------------------------------------------------------------------
Relevant bottleneck role : CH-independent static Gossip reference (no CH target)
Expected target           : []
Runtime target            : []
Target-selection basis    : no native or mapped CH; expected empty target set
Target resolution         : PASS

Before overload (t=0, factor=1.0):
  Target                  : none by architecture
  Node 0   one-hop delay : 1.127885359692

After overload (t=0, configured run factor):
  Target                  : none; no delay injected (expected)
  Node 0   one-hop delay : 1.127885359692
  Non-target added delay  : 0.000000000000
  processing_delay        : 0.0 before / 0.0 after (unchanged)
Injection observed        : NO (expected: no CH target)
Non-target unexpectedly overloaded: NO
Injection activation      : PASS
Target isolation          : PASS
Forwarding-policy intact  : PASS
Comparator isolation      : PASS
Deterministic replay      : PASS
Result                    : PASS

------------------------------------------------------------------------
CLUSTER
------------------------------------------------------------------------
Relevant bottleneck role : static Structured cluster heads
Expected target           : [0, 1, 2, 3]
Runtime target            : [0, 1, 2, 3]
Target-selection basis    : round-robin clusters; lowest node ID per cluster
Target resolution         : PASS

Before overload (t=0, factor=1.0):
  Node 0   one-hop delay : 1.127885359692
  Node 4   one-hop delay : 1.147294242833

After overload (t=0, configured run factor):
  Node 0   one-hop delay : 3.127885359692
  Observed added delay    : 2.000000000000
  Node 4   one-hop delay : 1.147294242833
  Non-target added delay  : 0.000000000000
  processing_delay        : 0.0 before / 0.0 after (unchanged)
Injection observed        : YES
Non-target unexpectedly overloaded: NO
Injection activation      : PASS
Target isolation          : PASS
Forwarding-policy intact  : PASS
Comparator isolation      : PASS
Deterministic replay      : PASS
Result                    : PASS

------------------------------------------------------------------------
DC-SOC
------------------------------------------------------------------------
Relevant bottleneck role : DBSCAN-derived density-cluster heads
Expected target           : [4]
Runtime target            : [4]
Target-selection basis    : DBSCAN membership; highest physical degree, tie -> lowest ID
Target resolution         : PASS

Before overload (t=0, factor=1.0):
  Node 4   one-hop delay : 1.147294242833
  Node 0   one-hop delay : 1.127885359692

After overload (t=0, configured run factor):
  Node 4   one-hop delay : 3.147294242833
  Observed added delay    : 2.000000000000
  Node 0   one-hop delay : 1.127885359692
  Non-target added delay  : 0.000000000000
  processing_delay        : 0.0 before / 0.0 after (unchanged)
Injection observed        : YES
Non-target unexpectedly overloaded: NO
Injection activation      : PASS
Target isolation          : PASS
Forwarding-policy intact  : PASS
Comparator isolation      : PASS
Deterministic replay      : PASS
Result                    : PASS

------------------------------------------------------------------------
AHBN
------------------------------------------------------------------------
Relevant bottleneck role : static cluster heads used by canonical AHBN
Expected target           : [0, 1, 2, 3]
Runtime target            : [0, 1, 2, 3]
Target-selection basis    : round-robin clusters; lowest node ID per cluster
Target resolution         : PASS

Before overload (t=0, factor=1.0):
  Node 0   one-hop delay : 1.127885359692
  Node 4   one-hop delay : 1.147294242833

After overload (t=0, configured run factor):
  Node 0   one-hop delay : 3.127885359692
  Observed added delay    : 2.000000000000
  Node 4   one-hop delay : 1.147294242833
  Non-target added delay  : 0.000000000000
  processing_delay        : 0.0 before / 0.0 after (unchanged)
Injection observed        : YES
Non-target unexpectedly overloaded: NO
Injection activation      : PASS
Target isolation          : PASS
Forwarding-policy intact  : PASS
Comparator isolation      : PASS
Deterministic replay      : PASS
Result                    : PASS

========================================================================
E1 CHECKS
========================================================================
Required interpreter                  : PASS
Frozen four-comparator set            : PASS
Target resolution                     : PASS
Configured overload activates         : PASS
Correct target node(s) affected       : PASS
Non-target nodes remain normal        : PASS
No direct forwarding-policy mutation  : PASS
Comparator isolation                  : PASS
Deterministic behaviour               : PASS

========================================================================
E1 RESULT: PASS
========================================================================
```

### Result

E1 RESULT: PASS

### Scientific interpretation

E1 confirmed that the Exp08 overload condition is injected into the architecture-specific CH/bottleneck role for Gossip, Structured, DC-SoC, and AHBN; activates at the configured point; affects the intended target rather than unrelated nodes; and does not directly alter the frozen dissemination or controller mechanisms. Gossip remains the intentionally CH-independent reference and therefore has no directly overloaded node.

## E2 — Run Gossip

### Execution context

- Date: 2026-08-20 (Asia/Kuala_Lumpur)
- Working directory: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
- Python interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Frozen configuration: `configs/exp08_ch_bottleneck.yaml`
- Selected strategy: `gossip` only
- Topology: BA, 100 nodes, `m=3`, topology cache enabled
- Seeds: 42–61
- Intended runs: 80 (20 seeds × 4 overload factors)
- Gossip fanout: 3 (frozen runner default)
- `ch_overload_factor`: `[1.0, 1.5, 2.0, 3.0]`
- Activation: simulator construction/run start (`t=0`)
- Resolved bottleneck semantics: Gossip is the CH-independent dissemination reference; it assigns no cluster-head role and therefore has no direct CH-overload target.
- Expected output: timestamped raw/per-run CSV under `outputs/csv/` and terminal log under `outputs/logs/`.

`run_batch.py` exposes only `--config`; it has no single-strategy CLI option. The execution therefore used a minimal inline orchestration command: it loaded the frozen YAML, shallow-copied the loaded mapping, restricted only the `exp08()` strategy iteration to `gossip`, then invoked the existing production `exp08()` runner and `save_results_csv()` output path. It did not implement or modify experimental behavior, forwarding, randomization, configuration values, overload semantics, or metric collection.

Only Gossip was executed.

### Exact command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -c 'from ahbn.config import load_yaml_config; from ahbn.utils import save_results_csv; from run_batch import exp08; cfg = load_yaml_config("configs/exp08_ch_bottleneck.yaml"); run_cfg = dict(cfg); run_cfg["strategies"] = ["gossip"]; print("E2 Exp08 Gossip-only execution"); print("Config: configs/exp08_ch_bottleneck.yaml"); print("Strategy: gossip"); print("Seeds: 42-61"); print("Overload factors: [1.0, 1.5, 2.0, 3.0]"); print("Intended runs: 80"); rows, trace_rows = exp08(run_cfg); path = save_results_csv(rows, "outputs/csv/exp08_gossip_results.csv"); print(f"Completed runs: {len(rows)}"); print(f"Adaptive trace rows: {len(trace_rows)}"); print(f"Saved {path}")' 2>&1 | tee outputs/logs/exp08_e2_gossip.log
```

### Complete terminal output

```text
E2 Exp08 Gossip-only execution
Config: configs/exp08_ch_bottleneck.yaml
Strategy: gossip
Seeds: 42-61
Overload factors: [1.0, 1.5, 2.0, 3.0]
Intended runs: 80
Completed runs: 80
Adaptive trace rows: 0
Saved outputs/csv/exp08_gossip_results_20260820_111017.csv
```

Warnings/errors: none.

### Output and sanity checks

- Raw/per-run results: `outputs/csv/exp08_gossip_results_20260820_111017.csv`
- Terminal log: `outputs/logs/exp08_e2_gossip.log`
- Completed rows: 80
- Strategies present: `gossip` only
- Unique seeds: 20; minimum 42, maximum 61
- Rows per overload factor: 20 for each of 1.0, 1.5, 2.0, and 3.0
- Delivery ratio range: 0.76–0.92
- Propagation delay range: 8.6104107855–11.8554085614 seconds
- Duplicates range: 153–185
- Total forwards range: 228–276
- Results SHA-256: `9f164955e1cd1bae972af43b24480e996abf3b41fc5a8314efe5500f140f97c5`
- Log SHA-256: `c6603701160ff8331212dbc4d5ef65faa7a8e1c186c4b2c5ea421b18b9fa276f`

### Result

E2 RESULT: PASS

All 80 frozen Exp08 Gossip runs completed and remain available as individual raw rows. No other comparator ran. E3 was not started.

## E4 — Pre-execution diagnostic deviation note

One unsaved DC-SoC diagnostic simulation was accidentally executed while checking the production construction path.

It invoked `run_single()` and therefore completed a simulation, but its result was neither saved nor incorporated into the E4 dataset. It is classified as a **non-experimental diagnostic run** and excluded from all analysis.

No parameters, algorithms, seeds, overload factors, or experimental settings were changed as a result. The official E4 dataset remains the predefined 80 runs: 20 seeds × 4 overload factors.

### Exact diagnostic command

```bash
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -c 'from ahbn.config import load_yaml_config; from run_batch import run_single; cfg=load_yaml_config("configs/exp08_ch_bottleneck.yaml"); d=cfg["dcsoc"]; sim=run_single(cfg=cfg,strategy_name="dcsoc",seed=42,topology_type=cfg["topology_type"],num_nodes=cfg["num_nodes"],use_topology_cache=cfg["use_topology_cache"],base_delay=cfg["base_delay"],jitter=cfg["jitter"],message_source=cfg["message_source"],num_clusters=cfg["num_clusters"],ch_overload_factor=1.0,ba_m=cfg["ba_m"],enable_adaptive_trace=False,scenario_tag="preflight") ; print("Preflight production run summary keys:", sorted(sim.keys()))'
```

### Complete diagnostic terminal output

```text
Preflight production run summary keys: ['adaptation_event_count', 'adaptation_rate', 'churn_event_count', 'churn_feedback_update_count', 'churn_join_count', 'churn_leave_count', 'cluster_repair_count', 'delivery_ratio', 'duplicates', 'failed_node_id', 'failure_mode', 'fanout_change_count', 'load_balance_cv', 'max_normalized_load', 'medium_forward_share', 'message_id', 'mode_switch_count', 'propagation_delay', 'recovery_time', 'strong_forward_share', 'total_forwards', 'weak_forward_share']
```

Diagnostic simulations before official batch: 1, excluded.

The diagnostic result was not saved, substituted for an official observation, used for tuning, or included in any aggregate statistic, confidence interval, table, or figure.

## E4 — Run DC-SoC

### Execution context

- Date: 2026-08-20 (Asia/Kuala_Lumpur)
- Working directory: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
- Python interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Frozen configuration: `configs/exp08_ch_bottleneck.yaml`
- Selected strategy: `dcsoc` only
- Seeds: 42–61 inclusive (20 unique seeds)
- Overload factors: `[1.0, 1.5, 2.0, 3.0]`
- Runs per overload factor: 20
- Official E4 experimental runs: 80
- Diagnostic simulations before official batch: 1, excluded
- Recorded E4 dataset rows / runs used for analysis: 80
- Frozen DC-SoC parameters: `eps=2.0`, `min_samples=3`, `fanout=3`, `inter_fanout=1`
- Overload targeting: E1-validated architecture-specific DBSCAN-derived DC-SoC cluster heads, resolved by the production construction path and runtime `is_cluster_head` state.

Only DC-SoC was executed in the official E4 batch. The earlier unsaved diagnostic result was not reused or substituted.

### Exact official execution command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -c 'from ahbn.config import load_yaml_config; from ahbn.utils import save_results_csv; from run_batch import exp08; cfg = load_yaml_config("configs/exp08_ch_bottleneck.yaml"); run_cfg = dict(cfg); run_cfg["strategies"] = ["dcsoc"]; d = run_cfg["dcsoc"]; print("E4 Exp08 DC-SoC-only execution"); print("Config: configs/exp08_ch_bottleneck.yaml"); print("Strategy: dcsoc"); print("Seeds: 42-61"); print("Unique seeds expected: 20"); print("Overload factors: [1.0, 1.5, 2.0, 3.0]"); print("Runs per overload factor: 20"); print("Intended runs: 80"); print(f"Frozen DC-SoC parameters: eps={d[chr(101)+chr(112)+chr(115)]}, min_samples={d[chr(109)+chr(105)+chr(110)+chr(95)+chr(115)+chr(97)+chr(109)+chr(112)+chr(108)+chr(101)+chr(115)]}, fanout={d[chr(102)+chr(97)+chr(110)+chr(111)+chr(117)+chr(116)]}, inter_fanout={d[chr(105)+chr(110)+chr(116)+chr(101)+chr(114)+chr(95)+chr(102)+chr(97)+chr(110)+chr(111)+chr(117)+chr(116)]}"); print("Overload targeting: DC-SoC architecture-specific DBSCAN-derived cluster heads via runtime is_cluster_head"); rows, trace_rows = exp08(run_cfg); path = save_results_csv(rows, "outputs/csv/exp08_dcsoc_results.csv"); print(f"Completed runs: {len(rows)}"); print(f"Adaptive trace rows: {len(trace_rows)}"); print(f"Saved {path}")' 2>&1 | tee outputs/logs/exp08_e4_dcsoc.log
```

### Complete official terminal output

```text
E4 Exp08 DC-SoC-only execution
Config: configs/exp08_ch_bottleneck.yaml
Strategy: dcsoc
Seeds: 42-61
Unique seeds expected: 20
Overload factors: [1.0, 1.5, 2.0, 3.0]
Runs per overload factor: 20
Intended runs: 80
Frozen DC-SoC parameters: eps=2.0, min_samples=3, fanout=3, inter_fanout=1
Overload targeting: DC-SoC architecture-specific DBSCAN-derived cluster heads via runtime is_cluster_head
Completed runs: 80
Adaptive trace rows: 0
Saved outputs/csv/exp08_dcsoc_results_20260820_114555.csv
```

Warnings/errors: none.

### Output and post-run validation

- Raw/per-run results: `outputs/csv/exp08_dcsoc_results_20260820_114555.csv`
- Terminal log: `outputs/logs/exp08_e4_dcsoc.log`
- CSV rows: 80
- Strategies present: `dcsoc` only; no Gossip, Structured/cluster, or AHBN rows
- Unique seeds: 20; exact set 42–61; minimum 42, maximum 61
- Overload factors: exactly 1.0, 1.5, 2.0, and 3.0
- Rows per overload factor: 20 each
- Expected Exp08 schema: present
- Adaptive trace rows: 0 (expected for the fixed DC-SoC comparator)
- Delivery ratio range: 0.69–0.94
- Propagation delay range: 6.8187090117–15.2569962441 seconds
- Duplicates range: 139–189
- Total forwards range: 207–282
- Frozen parameters after run: `eps=2.0`, `min_samples=3`, `fanout=3`, `inter_fanout=1`
- Architecture-specific overload targeting: preserved
- E2 Gossip SHA-256 unchanged: `9f164955e1cd1bae972af43b24480e996abf3b41fc5a8314efe5500f140f97c5`
- E3 Structured SHA-256 unchanged: `0ee7e3e5e49d0cc4e101427cf02e9e9c29a05b7399c916f81f61b7d3b60d1c59`
- E4 CSV SHA-256: `1e4ede149ddcd5ebae4664ea7a29ba85c87af5d9053488693eb921b4d46831b1`
- E4 log SHA-256: `f99af22b0303d9e12980fafaa6edab787655009ef2f0be9e61d1cd34ac0c0029`
- Scientific implementation/configuration changes during E4: none
- Post-run validation: PASS

### Result

E4 RESULT: PASS

The official E4 dataset contains exactly the predefined 80 recorded observations. The one earlier diagnostic simulation remains transparently documented and excluded from the experimental dataset and all analysis. No AHBN implementation was modified. No DC-SoC tuning was performed. Frozen Stage 3.5 DC-SoC parameters and the E1-validated architecture-specific DC-SoC overload targeting were preserved. Existing E2/E3 outputs were not modified. E5 was not started.

## E5 — Run AHBN

### Purpose and execution context

Execute the frozen AHBN-only Exp08 production matrix and preserve its naturally generated adaptive traces for E6. This was execution and basic integrity validation only; no behavioural analysis, tuning, repair, or scientific reinterpretation was performed.

- Date: 2026-08-20 (Asia/Kuala_Lumpur)
- Working directory: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
- Python interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Authoritative configuration: `configs/exp08_ch_bottleneck.yaml`
- Strategy: `ahbn` only
- Seeds: 42–61 inclusive (20 unique seeds)
- Overload factors: `[1.0, 1.5, 2.0, 3.0]`
- Runs per overload factor: 20
- Intended and actual completed runs: 80
- Frozen parameters consumed: `alpha=0.3`, centers `(0.5,0.5,0.5,0.5)`, weights `(-1,+1,-1,+1)`, `kappa=1`, `beta=1`, fanout bounds `[2,4]`, mode threshold `0.5`, default fanout `3`
- Adaptive tracing: enabled by the existing production `exp08()` path for AHBN
- Other comparators executed during E5: none
- Extra AHBN simulations before or after the batch: none

`run_batch.py` has no single-strategy CLI option. As in E2–E4, the production execution loaded the frozen YAML, shallow-copied the loaded mapping, restricted only the strategy iteration to `ahbn`, and invoked the existing `exp08()` production function. Both output writers are the existing canonical utilities.

### Exact production command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -c 'from ahbn.config import load_yaml_config; from ahbn.utils import save_results_csv, save_adaptive_trace_csv; from run_batch import exp08; cfg = load_yaml_config("configs/exp08_ch_bottleneck.yaml"); run_cfg = dict(cfg); run_cfg["strategies"] = ["ahbn"]; a = run_cfg["ahbn"]; print("E5 Exp08 AHBN-only execution"); print("Config: configs/exp08_ch_bottleneck.yaml"); print("Strategy: ahbn"); print("Seeds: 42-61"); print("Unique seeds expected: 20"); print("Overload factors: [1.0, 1.5, 2.0, 3.0]"); print("Runs per overload factor: 20"); print("Intended runs: 80"); print(f"Frozen AHBN parameters: alpha={a[chr(97)+chr(108)+chr(112)+chr(104)+chr(97)]}, centers=({a[chr(100)+chr(48)]},{a[chr(108)+chr(48)]},{a[chr(117)+chr(48)]},{a[chr(99)+chr(48)]}), weights=({a[chr(119)+chr(95)+chr(100)]},{a[chr(119)+chr(95)+chr(108)]},{a[chr(119)+chr(95)+chr(117)]},{a[chr(119)+chr(95)+chr(99)]}), kappa={a[chr(107)+chr(97)+chr(112)+chr(112)+chr(97)]}, beta={a[chr(98)+chr(101)+chr(116)+chr(97)]}, fanout=[{a[chr(109)+chr(105)+chr(110)+chr(95)+chr(102)+chr(97)+chr(110)+chr(111)+chr(117)+chr(116)]},{a[chr(109)+chr(97)+chr(120)+chr(95)+chr(102)+chr(97)+chr(110)+chr(111)+chr(117)+chr(116)]}], threshold={a[chr(109)+chr(111)+chr(100)+chr(101)+chr(95)+chr(116)+chr(104)+chr(114)+chr(101)+chr(115)+chr(104)+chr(111)+chr(108)+chr(100)]}, default_fanout={a[chr(100)+chr(101)+chr(102)+chr(97)+chr(117)+chr(108)+chr(116)+chr(95)+chr(102)+chr(97)+chr(110)+chr(111)+chr(117)+chr(116)]}"); rows, trace_rows = exp08(run_cfg); path = save_results_csv(rows, "outputs/csv/exp08_ahbn_results.csv"); trace_path = save_adaptive_trace_csv(trace_rows, "outputs/csv/exp08_ahbn_adaptive_trace.csv", add_timestamp=True); print(f"Completed runs: {len(rows)}"); print(f"Adaptive trace rows: {len(trace_rows)}"); print(f"Saved {path}"); print(f"Saved {trace_path}")' 2>&1 | tee outputs/logs/exp08_e5_ahbn.log
```

### Complete terminal output

```text
E5 Exp08 AHBN-only execution
Config: configs/exp08_ch_bottleneck.yaml
Strategy: ahbn
Seeds: 42-61
Unique seeds expected: 20
Overload factors: [1.0, 1.5, 2.0, 3.0]
Runs per overload factor: 20
Intended runs: 80
Frozen AHBN parameters: alpha=0.3, centers=(0.5,0.5,0.5,0.5), weights=(-1.0,1.0,-1.0,1.0), kappa=1.0, beta=1.0, fanout=[2,4], threshold=0.5, default_fanout=3
Completed runs: 80
Adaptive trace rows: 19985
Saved outputs/csv/exp08_ahbn_results_20260820_115817.csv
Saved outputs/csv/exp08_ahbn_adaptive_trace_20260820_115817.csv
```

Warnings/errors: none.

### Outputs and sanity checks

- Raw/per-run results: `outputs/csv/exp08_ahbn_results_20260820_115817.csv`
- Adaptive trace: `outputs/csv/exp08_ahbn_adaptive_trace_20260820_115817.csv`
- Terminal log: `outputs/logs/exp08_e5_ahbn.log`
- Result rows: exactly 80; no duplicate strategy/seed/overload combinations
- Strategy isolation: `ahbn` only
- Seeds: exact complete set 42–61; 20 unique; minimum 42; maximum 61
- Overload factors: exactly 1.0, 1.5, 2.0, and 3.0; 20 rows each
- Required fields: `delivery_ratio`, `propagation_delay`, `duplicates`, and `total_forwards` present with no missing, NaN, or infinite values
- Delivery ratio range: 0.70–0.95, entirely within `[0,1]`
- Propagation delay range: 7.6439283606–16.7392140870 seconds
- Duplicates range: 141–191; no negative values
- Total forwards range: 210–285; no negative values
- Adaptive trace rows: 19,985; all tagged `ahbn`; all 20 seeds and all four overload scenario tags present; no missing experiment/strategy/seed/scenario context
- Frozen AHBN source/config changes during E5: none
- E2 Gossip SHA-256 unchanged: `9f164955e1cd1bae972af43b24480e996abf3b41fc5a8314efe5500f140f97c5`
- E3 Structured SHA-256 unchanged: `0ee7e3e5e49d0cc4e101427cf02e9e9c29a05b7399c916f81f61b7d3b60d1c59`
- E4 DC-SoC SHA-256 unchanged: `1e4ede149ddcd5ebae4664ea7a29ba85c87af5d9053488693eb921b4d46831b1`
- E5 results SHA-256: `f0011b3ca87c6794832ead9793010d3aa27d5811ead72c980ac6b587674f60ac`
- E5 adaptive trace SHA-256: `2e7ab084cf1abe8bcdde28dfe4806940055146eb54011709535a4e972ccb3362`
- E5 log SHA-256: `19757a2cb1aaa7e9e0374e41b7953c3e110e765b2ad44a76ce7c9e79a534c5de`

### Result

E5 RESULT: PASS

Exactly 80 intended AHBN production executions completed, with no additional AHBN simulations. The result matrix, basic numeric integrity, strategy isolation, frozen implementation protection, prior-output preservation, and adaptive trace existence checks all passed. Behavioural interpretation is deferred to E6.


## E7 — Aggregate 20 Runs (Mean and 95% CI)

### Scope and inputs

E7 validated and aggregated only the four frozen per-run result files:

- `outputs/csv/exp08_gossip_results_20260820_111017.csv`
- `outputs/csv/exp08_structured_results_20260820_112714.csv`
- `outputs/csv/exp08_dcsoc_results_20260820_114555.csv`
- `outputs/csv/exp08_ahbn_results_20260820_115817.csv`

The Structured comparator retains its canonical stored strategy value, `cluster`. The AHBN adaptive trace was not used or treated as a set of independent statistical samples.

### Method

For every strategy × `ch_overload_factor` condition, each metric was calculated from the 20 independent seed-level observations. The aggregated metrics were `delivery_ratio`, `propagation_delay`, `duplicates`, and `total_forwards`.

The script records `n`, the arithmetic mean, sample standard deviation (`ddof=1`), standard error, and the two-sided Student-t 95% confidence interval:

```text
SEM = sample_std / sqrt(n)
95% CI = mean ± t_(0.975, n-1) × SEM
n = 20; df = 19
```

No confidence interval was clipped to a metric's natural bounds.

### Exact command

Executed from `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`:

```bash
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/aggregate_exp08_e7.py 2>&1 | tee outputs/logs/exp08_e7_aggregation.log
```

### Complete terminal output

```text
E7 Exp08 aggregation
Input datasets:
  gossip: outputs/csv/exp08_gossip_results_20260820_111017.csv
  cluster: outputs/csv/exp08_structured_results_20260820_112714.csv
  dcsoc: outputs/csv/exp08_dcsoc_results_20260820_114555.csv
  ahbn: outputs/csv/exp08_ahbn_results_20260820_115817.csv

Compact aggregation table (mean [95% CI]):
ahbn     overload=1   n=20  delivery_ratio=0.8305 [0.809379, 0.851621]  propagation_delay=10.015 [9.55615, 10.4739]  duplicates=167.1 [162.876, 171.324]  total_forwards=249.15 [242.814, 255.486]
ahbn     overload=1.5 n=20  delivery_ratio=0.8295 [0.806002, 0.852998]  propagation_delay=9.67072 [9.04412, 10.2973]  duplicates=166.9 [162.2, 171.6]  total_forwards=248.85 [241.801, 255.899]
ahbn     overload=2   n=20  delivery_ratio=0.837 [0.813917, 0.860083]  propagation_delay=10.5334 [9.48505, 11.5817]  duplicates=168.4 [163.783, 173.017]  total_forwards=251.1 [244.175, 258.025]
ahbn     overload=3   n=20  delivery_ratio=0.8205 [0.79623, 0.84477]  propagation_delay=9.92877 [9.3517, 10.5058]  duplicates=165.1 [160.246, 169.954]  total_forwards=246.15 [238.869, 253.431]
cluster  overload=1   n=20  delivery_ratio=1 [1, 1]  propagation_delay=4.49796 [4.45441, 4.54152]  duplicates=99 [99, 99]  total_forwards=198 [198, 198]
cluster  overload=1.5 n=20  delivery_ratio=1 [1, 1]  propagation_delay=6.02282 [5.98005, 6.0656]  duplicates=99 [99, 99]  total_forwards=198 [198, 198]
cluster  overload=2   n=20  delivery_ratio=1 [1, 1]  propagation_delay=7.52282 [7.48005, 7.5656]  duplicates=99 [99, 99]  total_forwards=198 [198, 198]
cluster  overload=3   n=20  delivery_ratio=1 [1, 1]  propagation_delay=10.5228 [10.48, 10.5656]  duplicates=99 [99, 99]  total_forwards=198 [198, 198]
dcsoc    overload=1   n=20  delivery_ratio=0.827 [0.808772, 0.845228]  propagation_delay=10.1437 [9.46217, 10.8251]  duplicates=166.4 [162.754, 170.046]  total_forwards=248.1 [242.632, 253.568]
dcsoc    overload=1.5 n=20  delivery_ratio=0.8375 [0.818482, 0.856518]  propagation_delay=10.2838 [9.32741, 11.2401]  duplicates=168.5 [164.696, 172.304]  total_forwards=251.25 [245.544, 256.956]
dcsoc    overload=2   n=20  delivery_ratio=0.8515 [0.830173, 0.872827]  propagation_delay=10.1711 [9.40893, 10.9333]  duplicates=171.3 [167.035, 175.565]  total_forwards=255.45 [249.052, 261.848]
dcsoc    overload=3   n=20  delivery_ratio=0.8215 [0.794926, 0.848074]  propagation_delay=10.0069 [9.16294, 10.8508]  duplicates=165.3 [159.985, 170.615]  total_forwards=246.45 [238.478, 254.422]
gossip   overload=1   n=20  delivery_ratio=0.8305 [0.809379, 0.851621]  propagation_delay=10.015 [9.55615, 10.4739]  duplicates=167.1 [162.876, 171.324]  total_forwards=249.15 [242.814, 255.486]
gossip   overload=1.5 n=20  delivery_ratio=0.8305 [0.809379, 0.851621]  propagation_delay=10.015 [9.55615, 10.4739]  duplicates=167.1 [162.876, 171.324]  total_forwards=249.15 [242.814, 255.486]
gossip   overload=2   n=20  delivery_ratio=0.8305 [0.809379, 0.851621]  propagation_delay=10.015 [9.55615, 10.4739]  duplicates=167.1 [162.876, 171.324]  total_forwards=249.15 [242.814, 255.486]
gossip   overload=3   n=20  delivery_ratio=0.8305 [0.809379, 0.851621]  propagation_delay=10.015 [9.55615, 10.4739]  duplicates=167.1 [162.876, 171.324]  total_forwards=249.15 [242.814, 255.486]

Strategies: 4
Seeds per condition: 20
Overload factors: [1.0, 1.5, 2.0, 3.0]
Expected raw rows: 320
Validated raw rows: 320
Expected aggregate conditions: 16
Generated aggregate conditions: 16
95% CI method: Student t
Degrees of freedom per condition: 19
Metrics aggregated: ['delivery_ratio', 'propagation_delay', 'duplicates', 'total_forwards']
Saved: outputs/csv/exp08_summary_20260820_123734.csv
Overall E7: PASS
```

### Validation and outputs

- Input files existed and each contained exactly 80 result rows.
- Each input contained only its expected canonical strategy value (`gossip`, `cluster`, `dcsoc`, or `ahbn`).
- Each strategy contained exactly the seed set 42–61 and exactly the overload-factor set `[1.0, 1.5, 2.0, 3.0]`.
- Every overload factor had exactly 20 observations, with no duplicate `(seed, ch_overload_factor)` pairs.
- All four metric columns existed and all aggregated values were numeric and finite.
- All four strategies shared the same complete 20 × 4 experimental grid.
- The summary contained exactly 16 expected conditions and every row had `n=20`.
- All means, sample standard deviations, SEMs, and CI bounds were finite; standard deviations and SEMs were non-negative; every CI contained its mean.
- No expected condition was missing and no extra condition was present.
- Frozen input hashes remained unchanged after aggregation.

Output summary: `outputs/csv/exp08_summary_20260820_123734.csv`
Terminal log: `outputs/logs/exp08_e7_aggregation.log`

### Result

E7 RESULT: PASS

No simulations were run or rerun. The AHBN adaptive trace was not treated as independent statistical samples. Frozen AHBN and comparator implementations, Exp08 configuration, and existing raw result CSVs were unchanged. E8 plotting was not started, and E9 scientific interpretation was not started.

## E8 — Plot Four Algorithms

### Purpose and provenance

E8 created the final descriptive 2 × 2 Exp08 comparison figure for Gossip, Structured, DC-SoC, and AHBN. It plotted the frozen E7 means and already-computed Student-t 95% confidence intervals against CH overload factor. No scientific interpretation was performed.

- E7 status: PASS
- E7 raw rows previously validated: 320
- E7 conditions aggregated: 16
- E7 runs per condition: 20
- E7 CI method: Student t, df=19
- Frozen E7 summary CSV used: `outputs/csv/exp08_summary_20260820_123734.csv`
- E7 terminal log: `outputs/logs/exp08_e7_aggregation.log`
- E7 aggregation script: `scripts/aggregate_exp08_e7.py`
- E8 plotting script created: `scripts/plot_exp08_e8.py`
- Exact Python interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Exact execution command: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/plot_exp08_e8.py`

### Validation and outputs

- Summary conditions validated: 16
- Strategies found: `ahbn`, `cluster`, `dcsoc`, `gossip`; displayed as AHBN, Structured, DC-SoC, and Gossip, respectively
- Overload factors found: `1.0`, `1.5`, `2.0`, `3.0`
- Runs per condition found: 20
- Metrics plotted: `delivery_ratio`, `propagation_delay`, `duplicates`, `total_forwards`
- Numerical integrity: each metric plotted exactly 16 means and 16 corresponding 95% CIs, each taken directly from one frozen E7 summary row
- PNG: `outputs/figures/exp08_four_algorithms_e8.png` (300 dpi)
- PDF: `outputs/figures/exp08_four_algorithms_e8.pdf`
- Complete terminal log: `outputs/logs/exp08_e8_plotting.log`
- Existing designated E8 figures replaced: NO; neither output existed before execution

### Result

E8 RESULT: PASS

The frozen E7 summary CSV was the sole numerical plotting input. No raw result CSV or AHBN adaptive trace was used as numerical plotting input. No simulations were rerun, no aggregation was recomputed, and no E7 artifact was modified. No frozen algorithm implementation, AHBN controller logic, DC-SoC implementation, or Exp08 configuration was modified. No scientific interpretation was performed. E9 was not started.

## E9 — Scientific Interpretation

This section incorporates the complete E9 scientific interpretation and is the single canonical Exp08 record. No separate E9 interpretation artifact is required.

### 1. Overall Finding

Within the frozen Exp08 setting, increasing CH overload primarily exposed a latency cost in the strongly CH-dependent Structured strategy rather than a delivery failure. Structured maintained a delivery-ratio mean of 1.000 at every overload factor, with 99 duplicate messages and 198 total forwards, while its mean propagation delay rose monotonically from 4.498 s to 10.523 s. Gossip was unchanged across overload factors because its architecture assigns no CH role and therefore had no direct CH-overload target. DC-SoC and AHBN showed non-monotonic, comparatively small changes in all four aggregate metrics; their means remained close to the Gossip operating point. Thus, Exp08 reveals a reliability-latency-overhead trade-off, not universal superiority by one adaptive method.

### 2. Delivery Ratio

The E7 mean sequences at overload factors 1.0, 1.5, 2.0, and 3.0 were: Gossip 0.8305, 0.8305, 0.8305, 0.8305 (95% CI [0.8094, 0.8516] throughout); Structured 1.000 at every factor (95% CI [1.000, 1.000]); DC-SoC 0.8270, 0.8375, 0.8515, 0.8215 (CIs [0.8088, 0.8452], [0.8185, 0.8565], [0.8302, 0.8728], [0.7949, 0.8481]); and AHBN 0.8305, 0.8295, 0.8370, 0.8205 (CIs [0.8094, 0.8516], [0.8060, 0.8530], [0.8139, 0.8601], [0.7962, 0.8448]).

Structured maintained delivery best and exhibited no degradation. Gossip was stable at a lower mean. DC-SoC and AHBN did not degrade monotonically: both increased modestly through factor 2.0 and then fell at factor 3.0. From factor 1.0 to 3.0, the simple descriptive changes were -0.0055 for DC-SoC and -0.0100 for AHBN. DC-SoC exceeded AHBN at factor 2.0 (0.8515 versus 0.8370); at factor 3.0 they were nearly equal (0.8215 and 0.8205), and both were below Gossip (0.8305). The evidence does not support claiming that AHBN delivered better than every comparator.

### 3. Propagation Delay

Mean delay sequences were 10.015 s throughout for Gossip; 4.498, 6.023, 7.523, and 10.523 s for Structured; 10.144, 10.284, 10.171, and 10.007 s for DC-SoC; and 10.015, 9.671, 10.533, and 9.929 s for AHBN. Structured showed the clearest CH-bottleneck sensitivity, increasing by 6.025 s from factor 1.0 to 3.0 while preserving perfect delivery. Its 95% CIs were [4.454, 4.542] s at factor 1.0 and [10.480, 10.566] s at factor 3.0.

Gossip was invariant because it had no CH target. DC-SoC remained close to 10 s, and AHBN fluctuated around the same level, peaking at factor 2.0 (10.533 s; CI [9.485, 11.582]) rather than factor 3.0. At factor 3.0, Structured had the largest mean delay (10.523 s), followed by Gossip (10.015 s), DC-SoC (10.007 s), and AHBN (9.929 s). These mean rankings are descriptive, not formal pairwise significance results. Structured's full delivery therefore carried an increasing latency cost under CH overload.

### 4. Duplicate Messages

Structured produced 99 duplicates at every factor, substantially fewer than the other strategies. Gossip produced 167.1 throughout. DC-SoC means were 166.4, 168.5, 171.3, and 165.3; AHBN means were 167.1, 166.9, 168.4, and 165.1. The corresponding 95% CIs for DC-SoC ranged from [162.8, 170.0] at factor 1.0 to [160.0, 170.6] at factor 3.0; AHBN ranged from [162.9, 171.3] to [160.2, 170.0]. Increased overload did not produce a monotonic redundancy increase: DC-SoC and AHBN peaked at factor 2.0 and declined at factor 3.0.

Fewer duplicates did not imply a reliability penalty here: Structured combined the lowest duplicate count with perfect delivery. Its observed cost instead appeared in delay. Conversely, the additional redundancy of Gossip, DC-SoC, and AHBN did not yield Structured-level delivery.

### 5. Total Forwards

Structured used 198 forwards at every factor, versus 249.15 for Gossip throughout. DC-SoC means were 248.10, 251.25, 255.45, and 246.45; AHBN means were 249.15, 248.85, 251.10, and 246.15. At factor 3.0, the 95% CIs were [238.48, 254.42] for DC-SoC and [238.87, 253.43] for AHBN. DC-SoC and AHBN rose to factor 2.0 and then declined rather than increasing monotonically.

Structured was most communication-efficient by both overhead metrics and delivered every node. Gossip, DC-SoC, and AHBN used approximately 48–57 more forwards per run without achieving higher delivery. High forwarding cost therefore did not purchase the highest reliability in this setting. Structured's efficiency was accompanied by increasing latency sensitivity rather than delivery degradation.

### 6. Cross-Algorithm Trade-Off

Numerically, Structured occupied the strongest reliability/overhead point—delivery 1.000, 99 duplicates, and 198 forwards—but its delay increased from 4.498 s to 10.523 s. This is consistent with, but does not prove, accumulation of injected latency along mandatory structured/CH paths. Gossip's constant metrics are consistent with decentralized static dissemination and the absence of a CH-overload target; this is architecture-specific immunity to this injection, not independence from every overload form.

DC-SoC's fixed DBSCAN clustering, fixed CH selection, fanout 3, inter-fanout 1, and structurally determined forwarding produced stable delay but no clear reliability or overhead advantage over Gossip and AHBN. DC-SoC is not adaptive. AHBN remained near the Gossip/DC-SoC performance band. Its aggregate outcomes do not show a decisive external advantage, although E6 demonstrates internal adaptive activity.

### 7. AHBN Adaptive Behaviour

E6 validated 19,985 trace rows across all 80 AHBN runs. Runtime latency and utilization observations were finite, non-constant, and varied across overload conditions. The controller made 4,831 dissemination-mode transitions; all 80/80 runs contained transitions; and there were zero controller-decision consistency mismatches.

Runtime fanout remained exactly 3 in every trace row, with zero fanout transitions, within the frozen [2,4] bounds. Exp08 therefore demonstrates mode adaptation, not fanout adaptation. AHBN's external behaviour may cautiously be related to switching dissemination modes in response to local observations, but E7 shows only that AHBN remained broadly comparable to Gossip and DC-SoC. It does not establish causation or universal superiority.

### 8. Limitations and Caution

Exp08 tests one CH-overload scenario, topology/workload configuration, and set of frozen implementations and parameters. Overload targets are architecture-specific: Structured, DC-SoC, and AHBN receive added delay at their respective CH roles, whereas Gossip constructs no CH and has no direct target. Each condition contains 20 seed-level runs. Student-t 95% confidence intervals quantify uncertainty around sample means but are not formal pairwise significance tests. The experiment does not isolate the causal contribution of AHBN mode switching, and conclusions apply only to the frozen implementations and parameterization used here.

### 9. Final Exp08 Takeaway

Exp08 indicates that CH overload need not reduce delivery when a dissemination structure preserves reachability, but it can shift cost into propagation delay. Structured maintained complete delivery with the lowest duplicate and forwarding counts, while its mean delay rose from 4.498 s to 10.523 s. Gossip was invariant because it had no CH-overload target, and fixed DC-SoC and adaptive AHBN remained near the same lower-delivery, higher-overhead operating band. AHBN demonstrably adapted through dissemination-mode switching in every run, with fanout fixed at 3, but Exp08 does not show universal performance superiority; it supports a conservative claim of internally responsive mode adaptation with external performance broadly comparable to Gossip and DC-SoC under the frozen conditions.

### Result

E9 RESULT: PASS

No simulations were rerun, no aggregation or confidence intervals were recomputed, and no raw result, E7 summary, E8 numerical result, frozen implementation, controller logic, or Exp08 configuration was modified. The E9 interpretation was consolidated into this document without changing its scientific conclusions. Exp09 was not started.
