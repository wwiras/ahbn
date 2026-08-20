# Stage 4 — Exp08 CH Overload

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
