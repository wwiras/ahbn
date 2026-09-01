# S1 — DC-SoC paper-to-simulator fidelity audit

## 1. Purpose and decision boundary

This is an audit-only comparison of the frozen DC-SoC simulator against Dong et al., “DC-SoC: Optimizing a Blockchain Data Dissemination Model Based on Density Clustering and Social Mechanisms”, *Applied Sciences* 14(21), 10058 (2024), DOI: 10.3390/app142110058. No simulator, configuration, validation output, raw CSV, Gossip, Structured, AHBN, or Stage 4 result was changed. No Stage 4 batch was run and S2 was not started.

Authoritative paper: https://doi.org/10.3390/app142110058

## 2. Files inspected

Executable code was the authority for current behavior. The following were read:

- all files under `configs/`, specifically including `stage3_dcsoc.yaml`, `exp08_ch_bottleneck.yaml`, `exp09_dense_topology.yaml`, and `exp10_failure.yaml`;
- `run_batch.py`, `run_one.py`;
- `ahbn/simulator.py`, `ahbn/node.py`, `ahbn/topology.py`, `ahbn/cluster.py`, `ahbn/control.py`;
- `ahbn/failure_injector.py`, `ahbn/churn_manager.py`, `ahbn/metrics.py`;
- `ahbn/strategies/dcsoc.py`, `gossip.py`, `cluster.py`, and `ahbn.py`;
- all `scripts/validate_dcsoc_s1.py` through `scripts/validate_dcsoc_s11.py`, plus `scripts/validate_dcsoc_s35_freeze.py`;
- `docs/stage3_dcsoc_sanity_validation.md`, including the recorded Stage 3, 3.4, and 3.5 evidence;
- the complete publisher HTML paper, especially Sections 3.1–3.3.4, 4, and 5 and Tables 2–3.

## 3. Commands executed and complete relevant output

The commands below are the complete audit command record. Long source listings are represented by the exact line ranges inspected; the fidelity matrix cites the relevant line ranges. No command invoked `run_batch.py` or a Stage 4 batch.

```text
$ pwd
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

$ git rev-parse --show-toplevel
/Users/wwiras/Documents/src/AHBNProj/ahbn

$ git status --short
[no output]

$ find /Users/wwiras/Documents/src/AHBNProj -maxdepth 5 -type f \( -iname '*.pdf' -o -iname '*dcsoc*' -o -iname '*dc-soc*' \) -print
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/docs/stage3_dcsoc_sanity_validation.md
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/configs/stage3_dcsoc.yaml
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s1.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s7.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s6.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s2.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s9.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s11.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s10.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s8.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s5.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s1.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s35_freeze.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_dcsoc_s4.py
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/ahbn/strategies/dcsoc.py
[No DC-SoC paper PDF was present locally; the publisher paper was used.]

$ nl -ba configs/stage3_dcsoc.yaml
60  # DBSCAN hop-distance radius.
61  eps: 2.0
63  # Minimum DBSCAN neighborhood density.
64  min_samples: 3
66  # Maximum total forwarding budget per node.
67  fanout: 3
69  # Maximum portion ... reserved for structured inter-cluster forwarding.
71  inter_fanout: 1

$ nl -ba ahbn/topology.py | sed -n '289,520p'
289 def assign_dcsoc_clusters(...)
297 DBSCAN is applied to the all-pairs shortest-path hop-distance matrix...
370 labels = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed")...
415 # Attach DBSCAN noise nodes (-1) to their nearest cluster.
460 cluster_mgr = ClusterManager(head_selection="highest_physical_degree")
475 # Choose one head per cluster: highest physical degree, tie -> lowest node ID.
486 head_id = max(... key=lambda nid: (len(nodes[nid].original_neighbors), -nid))
498 # Create logical inter-cluster head chain...
517 nodes[left].gateway_neighbors.append(right)
518 nodes[right].gateway_neighbors.append(left)

$ nl -ba ahbn/strategies/dcsoc.py | sed -n '100,254p'
100 def select_targets(...)
124 local_candidates = [active physical neighbours in the same cluster]
151 if not node.is_cluster_head:
153     return self._sample(... fanout ...)
172 gateway_candidates = [active gateway neighbours]
196 gateway_budget = min(self.inter_fanout, self.fanout, len(gateway_candidates))
217 local_budget = max(0, self.fanout - len(selected_gateways))
243 return de-duplicated selected_gateways + selected_local

$ nl -ba run_one.py | sed -n '115,170p'; nl -ba run_batch.py | sed -n '133,218p'
run_one.py:122 cluster_manager = assign_dcsoc_clusters(... eps default 2.0, min_samples default 3)
run_one.py:138 strategy = DCSOCStrategy(... fanout default 3, inter_fanout default 1)
run_one.py:168 controller=controller [remains None on the dcsoc branch]
run_batch.py:140 cluster_manager = assign_dcsoc_clusters(...)
run_batch.py:156 strategy = DCSOCStrategy(...)
run_batch.py:201 controller=controller [remains None on the dcsoc branch]
run_batch.py:217 resource_aware_heads=False

$ nl -ba ahbn/topology.py | sed -n '244,287p'; nl -ba ahbn/simulator.py | sed -n '734,788p'
244 def refresh_cluster_overlay(...)
256 cluster_mgr.cluster_to_members = {}
257 cluster_mgr.cluster_to_head = {}
259 [reuse each active node's existing cluster_id]
269 head_id = _select_cluster_head(...)
273 [rebuild a linear chain of cluster heads]
281 def repair_topology_after_churn(...)
286 refresh_active_neighbors(nodes)
287 refresh_cluster_overlay(nodes, cluster_mgr,...)
734 def handle_churn_leave(...)
747 node.leave_network()
749 repair_topology_after_churn(...)
755 record churn event; 756 record cluster repair
762 def handle_churn_join(...)
775 node.rejoin_network()
777 repair_topology_after_churn(...)
783 record churn event; 784 record cluster repair

$ nl -ba ahbn/failure_injector.py | sed -n '55,99p'
55 def apply(self, simulator):
58 if mode == "node_failure": ... simulator.nodes[node_id].fail()
65 elif mode == "ch_failure": ... simulator.nodes[node_id].fail()
72 elif mode == "overload": ... node.set_overload(extra_delay)
86 self.applied = True
[No topology repair, core replacement, recovery, or reclustering call exists here.]

$ PYTHONDONTWRITEBYTECODE=1 /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_s35_freeze
STAGE 3.5 — DC-SoC MINIMAL PARAMETER SANITY / FREEZE
Topology type: BA
Supported experiment sizes: 30 / 50 / 100
DBSCAN eps: 2.0
DBSCAN min_samples: 3
Cluster-head rule: highest physical degree
Tie-break: lowest node ID
Forwarding: intra-cluster neighbour forwarding + CH gateway forwarding
Fanout: fixed (total=3, CH gateway reserve=1)
Adaptive control: disabled
AHBN parameters consumed: NO
DC-SoC baseline: FROZEN
AHBN: FROZEN_SEPARATELY
Comparison ready: YES
STAGE 3.5 RESULT: PASS

$ for n in 1 2 3 4 5 6 7 8 9 10 11; do PYTHONDONTWRITEBYTECODE=1 /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_s${n} | tail -n 8; done
S1 PASS — Every node has exactly one valid and internally consistent DC-SoC cluster assignment.
S2 RESULT: PASS — Cluster-head identification is correct. Tie-breaking NOT OBSERVED in natural topology.
S3 RESULT: PASS — Intra-cluster forwarding observed.
S4 RESULT: PASS — Cluster-head relay behavior passes; inter-cluster dissemination NOT EXERCISED.
S5 duplicate behavior plausible: PASS
S6 RESULT: PASS — churn structural update and reconstructed active memberships/heads match validator.
S7 RESULT: PASS — DC-SoC structure-adaptive, forwarding-fixed; no AHBN runtime controller.
S8 RESULT: PASS — forwarding remains structurally determined.
S9 RESULT: PASS — AHBN independence.
S10 RESULT: PASS — deterministic replay.
S11 RESULT: PASS — AHBN-controller isolation.
```

The paper was inspected via the publisher HTML and DOI. Relevant paper evidence is paraphrased rather than reproduced: Sections 3.1–3.2 define latency-distance density clustering and DBSCAN++; 3.3.1 defines a DAG/broadcast multi-way tree and core-driven push with confirmations; 3.3.2 defines heartbeat deactivation, grace/expulsion, core replacement and relationship transfer; 3.3.3 defines on-demand missing-block recovery; 3.3.4 defines periodic `du` re-clustering after ongoing propagation completes, with temporary block caching. Table 2 reports the paper experiment's `eps=3`, `minPts=10`; these values are not directly portable because this simulator uses hop distance rather than the paper's latency vector.

## 4. Frozen DC-SoC configuration and behavior found

- `eps=2.0`; `min_samples=3`; standard sklearn DBSCAN over all-pairs physical hop distance, not paper DBSCAN++ over propagation latency.
- Every noise node is attached to the nearest established cluster; if no cluster exists, all nodes enter cluster 0.
- Exactly one simulator cluster head per cluster: highest original physical degree; tie: lowest node ID.
- Fixed total fanout 3; a head reserves at most `inter_fanout=1` from that same budget for a gateway.
- Ordinary nodes randomly push to active same-cluster physical neighbours. Heads randomly push to the linear head-chain gateway plus local neighbours.
- Clustering runs at construction only. Churn repair does **not** call DBSCAN and preserves existing `cluster_id` values.
- Generic churn leave/join rebuilds active neighbours, cluster membership, heads, and the head chain immediately. It is a simplified structural rebuild, not the paper's grace-period/core-relationship transfer protocol.
- Exp10 failure injection calls `Node.fail()` directly and performs no structural repair or replacement. Also, current `configs/exp10_failure.yaml` does not list `dcsoc` among its strategies.
- Recovery only marks a node active and restores overlay membership through generic churn repair. There is no ledger/missing-block state or recovery traffic.
- The AHBN controller is `None` for DC-SoC. No EWMA, mode switch, adaptive fanout, or other runtime forwarding adaptation is used.

## 5. Paper → simulator fidelity matrix

| ID | DC-SoC mechanism | Paper behaviour | Current simulator behaviour | Fidelity status | Dissemination-relevant? | Recommended S2 action | Future validation | Measurable cost / metric | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| A1 | Density distance | Cluster per organization using peer transmission delay as Euclidean distance | One global clustering over physical hop-count shortest paths | SIMPLIFIED | Yes, Exp09 | SIMPLIFY WITH JUSTIFICATION: retain simulator-native delay/hop model only with explicit mapping | Crafted latency-vs-hop topology changes assignments as specified | `initial_clustering_time`, distance computations | Paper 3.2; `topology.py:297-374` |
| A2 | DBSCAN++ | Uses DBSCAN++/K-center sampling; paper states O(mn) | sklearn standard DBSCAN; all-pairs paths | SIMPLIFIED | Yes, scaling/Exp09 | SIMPLIFY WITH JUSTIFICATION unless algorithmic clustering cost is a study target | Golden assignments plus complexity/counter test | clustering operations and `initial_clustering_time`; never Python wall clock as simulated delay | Paper 3.2; `topology.py:333-374` |
| A3 | Paper parameters | Evaluation Table 2 uses `eps=3`, `minPts=10` with latency 1–30 ms | `eps=2.0`, `min_samples=3` with hop distance | SIMPLIFIED | Yes | KEEP CURRENT only as a documented model calibration; do not claim numerical parameter fidelity | sensitivity/cluster-count sanity without retuning Stage 4 in S1 | cluster count, outlier count | Paper Table 2; config lines 60–64 |
| A4 | Noise/outliers retained | Outliers retained and attached near a core to preserve fairness/coverage | Noise attached to nearest cluster member; all-one fallback | FUNCTIONALLY EQUIVALENT | Yes | KEEP CURRENT | all nodes assigned exactly once; nearest-cluster oracle | attachment count, coverage | Paper 3.2; `topology.py:389-454`; validator S1 |
| A5 | Core/leaf semantics | DBSCAN core objects head neighborhood adjacency; non-core nodes are leaves | One elected head per entire DBSCAN cluster; all others ordinary members | MISSING | Yes, Exp08/09/10 | IMPLEMENT paper role/topology semantics or explicitly narrow comparator claim | crafted multi-core cluster verifies role set and parent/child relations | core count, leaf count, edges | Paper 3.2–3.3.1; `topology.py:475-496` |
| A6 | Core selection | Density creates core objects; replacement election uses vote or maximum social relation to failed core | Initial head is maximum physical degree, lowest-ID tie | SIMPLIFIED | Yes, Exp08/10 | SIMPLIFY WITH JUSTIFICATION for initial representative; replacement needs separate faithful rule | tied degree and replacement-oracle tests | election comparisons | Paper 3.2, 3.3.2; `topology.py:475-492` |
| A7 | Propagation structure | Adjacency lists form a directed acyclic graph/broadcast multi-way tree with superior cores, parents, children, neighborhoods | Undirected physical intra-cluster graph plus bidirectional linear chain of one head per cluster | MISSING | Yes, all three | IMPLEMENT minimal explicit dissemination DAG/relationships | acyclicity, reachability, parent-child and neighborhood invariants | `topology_edges_changed`, diameter, edge count | Paper 3.3.1/Fig.3; `topology.py:498-518` |
| B1 | Buffered/timed master push | Master buffers blocks and pushes when full or interval expires | A message is injected immediately | MISSING | Not for single-block propagation comparison unless batching is introduced | INTENTIONALLY OMIT | Scope assertion: experiments inject one block and exclude batching | none in current scope | Paper 3.3.1; `simulator.py:163-183` |
| B2 | Core-driven push | Master sends to neighborhood/routing nodes; core objects continue propagation; non-core nodes only append block | Every first-receiving ordinary node forwards randomly; head also forwards | MISSING | Yes, all three | IMPLEMENT | trace proves leaves never forward and cores cover tree | transmissions, per-role forward load, propagation delay | Paper 3.3.1; `dcsoc.py:124-243`; `simulator.py:699-727` |
| B3 | Inter-cluster routing | Routing/core hierarchy in broadcast tree | Random selection from adjacent head(s) in linear head chain, reserve 1 | SIMPLIFIED | Yes, all three | IMPLEMENT as part of explicit DAG; keep bounded fanout only if comparison policy is documented | deterministic multi-cluster delivery/path trace | cross-cluster transmissions, hop/path length | Paper 3.3.1; `dcsoc.py:172-210`; validator S4 says not exercised in frozen sanity topology |
| B4 | Completion confirmation | Cores confirm to superior core; master decides push complete after sufficient confirmations | No confirmation messages/state | MISSING | Yes for completion/control overhead; not needed to forward payload | IMPLEMENT lightweight confirmation events/counters if paper-complete dissemination is claimed | completion occurs only after required confirmations | confirmation messages, completion delay | Paper 3.3.1; no corresponding code |
| B5 | Duplicate handling | Ledger/hash check prevents accepting an already held block | `seen_messages` suppresses re-forward after duplicate; duplicates counted | FUNCTIONALLY EQUIVALENT | Yes | KEEP CURRENT | independent duplicate oracle (existing S5) | duplicates, redundant transmissions | Paper 3.3.1; `node.py:149-166`; `simulator.py:646-666` |
| C1 | Heartbeat/inactive grace | Heartbeat marks peer deactivated, temporarily removes it; reinstate if back in fixed period, else expel | Scheduled churn immediately removes; no heartbeat, grace timer, or formal expulsion | MISSING | Yes, Exp10 | IMPLEMENT lifecycle states/timers at abstraction level | deactivate→reinstate and deactivate→expel timelines | inactive duration, detection/control events | Paper 3.3.2; `churn_manager.py`; `node.py:172-193` |
| C2 | Inactive ordinary leaf | Temporarily removed; relationship can be reinstated; new peer randomly enters a neighborhood until update | Churn rebuild excludes inactive node; join restores old `cluster_id`; no new-node model | SIMPLIFIED | Yes, Exp10 | IMPLEMENT minimum leave/rejoin/new-node distinctions | old node restores relation; new node gets temporary neighborhood | repair count, edges changed | Paper 3.3.2; `simulator.py:734-788`; `topology.py:244-287` |
| C3 | Core replacement | Failed core transfers role; vote or automatic social-related successor inside same neighborhood | Generic churn rebuild elects by generic head rule; ordinary Exp10 core failure does no repair | MISSING | Yes, strongly Exp10; Exp08 only if overload is treated as inactivity (paper does not require that) | IMPLEMENT | fail core during propagation; correct successor and uninterrupted reachability | `core_replacement_count`, replacement delay | Paper 3.3.2; `failure_injector.py:55-91`; churn path cited above |
| C4 | Relationship transfer/local repair | Transfer failed core's neighborhood affiliation and notify all parent peers; local operation | Recomputes all active cluster member lists and entire linear head chain; no parents/neighborhood transfer/notifications | SIMPLIFIED | Yes, Exp10 | IMPLEMENT local structural repair after explicit relationships exist | unaffected clusters/edges unchanged; affected parent and neighborhood updated | `structural_repair_count`, `topology_edges_changed`, repair control messages | Paper 3.3.2; `topology.py:244-287` |
| C5 | Returning former core | Returns within window as ordinary leaf in corresponding neighborhood | Join may immediately regain head under deterministic re-election | MISSING | Yes, Exp10/churn | IMPLEMENT | former core rejoins and is not core until a later legitimate update/election | role transitions, recovery count | Paper 3.3.2; `simulator.py:762-784` |
| D1 | Ledger recovery | New/reactivated peer requests batches of missing blocks from multiple high-social-value peers in own/other organizations; on demand; no periodic pull | No per-node ledger gaps, request peers, recovery messages, or recovery delay | MISSING | Yes, Exp10 if nodes return/join; not relevant to permanent one-way failures | IMPLEMENT simplified on-demand recovery with defensible peer selection absent social model | returning node misses blocks, requests and catches up before normal status | `recovery_count`, recovery messages/bytes, `T_recovery` | Paper 3.3.3; no corresponding code |
| D2 | Periodic pull removed | Paper deliberately retains recovery but removes frequent pull | No pull | EXACT | No separate action | KEEP CURRENT | assert no periodic pull events | zero pull traffic | Paper 3.3.3; no pull code |
| E1 | Periodic `du` update | After each lifecycle, online peers are re-clustered and propagation structure regenerated | DBSCAN only at construction; churn repair preserves cluster IDs | MISSING | Exp09 only if simulation spans `du` or state changes; strongly Exp10 dynamic; not Exp08 solely to add cost | IMPLEMENT with an explicit lifecycle trigger only in applicable experiments | topology/state change before `du`, assignments unchanged before and rebuilt after | `recluster_count`, `recluster_time`, `topology_edges_changed` | Paper 3.1, 3.3.4; `run_*` construction; `topology.py:259-287` |
| E2 | Update barrier/cache | Wait for current propagation tasks; cache newly arrived blocks; rebuild, then push cache | No update barrier or cache | MISSING | Yes if E1 and multi-block overlap are modeled | IMPLEMENT only with periodic update/multi-block overlap; otherwise omit by scope | update event during propagation waits; cached ordering preserved | wait/cache count, `T_topology_update` | Paper 3.3.4; no corresponding code |
| F1 | Social trust acceptance | Every propagation computes social credibility; acceptance and successor/recovery peer choices use trust | No trust state or acceptance gate | INTENTIONALLY OMITTED | Security-relevant, but outside benign Stage 4 dissemination comparison; replacement/recovery need a non-social surrogate disclosed | INTENTIONALLY OMIT; define deterministic dissemination-only surrogate in S2 | malicious/trust code absent; surrogate deterministic | none unless security experiments added | Paper 3.3.1, Sec.4; no code |
| F2 | Economic incentives | Rewards/penalties contribute to social credibility and malicious resistance | Not modeled | INTENTIONALLY OMITTED | No for benign dissemination comparison | INTENTIONALLY OMIT | scope test/documentation | none | Paper Sec.4; no code |
| F3 | Malicious/Sybil/Byzantine defense | Trust/economic mechanisms filter distorted data and reduce attacker influence | Not modeled | INTENTIONALLY OMITTED | No for Exp08/09/10 as currently benign failure/topology experiments | INTENTIONALLY OMIT | scope assertion; do not claim security fidelity | none | Paper Secs.4–5; no code |
| F4 | Fabric/crypto/organizations | CA/MSP, ledgers, hashes, encryption, channels, anchors, cross-organization requests | Abstract event simulator with no Fabric stack or organizations | INTENTIONALLY OMITTED | Only ledger-gap/recovery abstraction affects dissemination | INTENTIONALLY OMIT implementation details; model only recovery traffic/state | abstraction contract | simulated messages/bytes only | Paper 2–4; simulator architecture |

Counts: 25 mechanisms; EXACT/FUNCTIONALLY EQUIVALENT 3; SIMPLIFIED 7; MISSING dissemination-relevant 10 (A5, A7, B2, B4, C1, C3, C5, D1, E1, and conditionally E2); intentionally omitted 4; one missing but out-of-scope single-block batching mechanism (B1).

## 6. Structural-adaptation findings

### Core replacement

- Paper: **YES**. An inactive core's neighborhood affiliation is transferred; parents are notified; vote or automatic election chooses a replacement.
- Current: **NO faithful implementation**. Generic churn rebuild can choose another head, but it rebuilds whole active cluster/head-chain state with a different rule. Exp10's `ch_failure` calls `fail()` and does not invoke repair at all.
- Experiment relevance: high for Exp10; not triggered in Exp09 static density sweeps; Exp08 overload is not automatically a paper-defined inactive-core event.
- State S2 would change: role, core/head map, parent/child/neighborhood links, gateway/routing links, active/inactive lifecycle, notification/replacement events.
- Measure: `core_replacement_count`, replacement/control messages, `topology_edges_changed`, and a separately modeled `T_topology_update`.

### Local structural repair

- Paper: **YES**. It transfers only the affected core neighborhood relationship and notifies parent cores.
- Current: **SIMPLIFIED and non-local** on churn, absent on Exp10 failure. It regenerates active membership/head maps and the complete linear head chain without explicit parent/child state.
- Experiment relevance: high for Exp10; relevant to Exp09 only if dynamic state changes; not inherently triggered by Exp08 overload.
- State S2 would change: explicit dissemination DAG, neighborhood ownership, parent links, pending notifications, active membership.
- Measure: `structural_repair_count`, affected/unaffected edge invariants, repair messages, `topology_edges_changed`, `T_topology_update`.

### Periodic re-clustering/structure regeneration

- Paper: **YES**, every `du`, after current propagation completes; new blocks are cached while waiting.
- Current: **NO**. DBSCAN is called only during simulator construction. Churn repair retains `cluster_id`.
- Experiment relevance: Exp09 only when a run spans a justified `du` or topology/state changes; high for dynamic Exp10; must not be added to Exp08 merely to impose overhead.
- State S2 would change: cluster assignments, core/leaf roles, DAG edges, update lifecycle/barrier/cache.
- Measure: `recluster_count`, clustering operation/time model, `topology_edges_changed`, queued blocks, `T_topology_update`.

## 7. Exp08 / Exp09 / Exp10 relevance

| Missing/relevant mechanism | Exp08 CH overload | Exp09 density/topology | Exp10 dynamic/failure |
|---|---|---|---|
| Faithful core/leaf DAG and core-only push | High: determines actual CH load | High: density changes roles/paths | High: defines repair target/state |
| Core replacement | Only if a core becomes inactive; overload alone is insufficient | Low in static runs | Critical for `ch_failure` |
| Local relationship repair | Not for overload alone | Only if dynamic topology changes | Critical |
| Return-as-leaf and grace/expulsion lifecycle | No | No in static sweep | Critical when return/churn is modeled |
| Ledger recovery | No | No in static sweep | High for join/reactivation; N/A to permanent failure without return |
| Periodic `du` re-clustering | Only if lifecycle naturally expires; never to add artificial cost | High if run/state model spans update | High in dynamic lifecycle |
| Update barrier/cache | No in current single-message runs | Conditional on `du` overlap | Conditional on `du` overlap |

Important configuration finding: `configs/exp10_failure.yaml:18-21` currently excludes `dcsoc`. S1 does not redesign or rerun Exp10, but S2/experiment planning must resolve this before claiming a DC-SoC Exp10 comparison.

## 8. Measurable-cost requirements

S2 must preserve three separate concepts:

- `T_propagation`: payload dissemination and, if modeled, confirmation traffic;
- `T_topology_update`: core election, relationship notifications/changes, and periodic re-clustering/regeneration;
- `T_recovery`: on-demand missing-block request and transfer after join/reactivation.

Required counters/metrics are `initial_clustering_time` (using a defensible simulation-time/operation model), `recluster_count`, `recluster_time`, `core_replacement_count`, `structural_repair_count`, `recovery_count`, and `topology_edges_changed`, plus control/recovery message counts where applicable. Python wall-clock time must never be injected as network or simulation delay. S2 must define a defensible simulation-time cost model before any structural timing is applied; S1 supplies no invented numerical costs.

## 9. Intentional omissions

Trust/social scoring, economic incentives, malicious detection, Sybil/Byzantine defense, Fabric MSP/CA/channel machinery, cryptography, and exact Fabric storage are omitted because Stage 4 compares benign dissemination behavior, not security or platform integration. This omission has a boundary: where the paper uses social value to choose a replacement or recovery source, S2 needs an explicit deterministic dissemination-only surrogate and must not claim full social-mechanism fidelity. Ledger recovery must still be abstracted because its traffic and delay directly affect dissemination after return.

## 10. Frozen-code proof

Pre-audit SHA-256 values:

```text
999740f7262d9d918c16fe701e0c9da024be90e5a2e9ad95f41bc84a993d784a  ahbn/strategies/dcsoc.py
50ed8c10408bb5601ccd6f441b2aed834a3a427b00d434aea10a4222b72441db  ahbn/strategies/ahbn.py
9a19ae2c9766ea36fe873d4d643cf51d9e8df555b42d2de11d946d30fb60f75f  ahbn/control.py
3cff1c3ead4ef3dbec8c1f67dc30a18cc8bb8ef5eb15ad55d591f94e558b2d53  ahbn/simulator.py
916eaa0e21cec4d3982876a983b858e2216a67bf146faa65c3a46fa349892419  ahbn/topology.py
c9adcbdb20e6d8ae052b6de712fb11dea2e3364bdc50d33211c0e2d79f06c853  run_batch.py
0da5a733a01909e67591773ffdee1939ed7d79c833fab6afa57a31527502ab35  run_one.py
```

Post-audit verification command and complete output:

```text
$ shasum -a 256 ahbn/strategies/dcsoc.py ahbn/strategies/ahbn.py ahbn/control.py ahbn/simulator.py ahbn/topology.py run_batch.py run_one.py; git status --short; git diff -- ahbn/strategies/dcsoc.py ahbn/strategies/ahbn.py ahbn/control.py ahbn/simulator.py ahbn/topology.py run_batch.py run_one.py; git diff --stat
999740f7262d9d918c16fe701e0c9da024be90e5a2e9ad95f41bc84a993d784a  ahbn/strategies/dcsoc.py
50ed8c10408bb5601ccd6f441b2aed834a3a427b00d434aea10a4222b72441db  ahbn/strategies/ahbn.py
9a19ae2c9766ea36fe873d4d643cf51d9e8df555b42d2de11d946d30fb60f75f  ahbn/control.py
3cff1c3ead4ef3dbec8c1f67dc30a18cc8bb8ef5eb15ad55d591f94e558b2d53  ahbn/simulator.py
916eaa0e21cec4d3982876a983b858e2216a67bf146faa65c3a46fa349892419  ahbn/topology.py
c9adcbdb20e6d8ae052b6de712fb11dea2e3364bdc50d33211c0e2d79f06c853  run_batch.py
0da5a733a01909e67591773ffdee1939ed7d79c833fab6afa57a31527502ab35  run_one.py
?? docs/S1_faithful_dcsoc.md
[No frozen-file diff and no diff-stat output: only the untracked audit document exists.]
```

The hashes match exactly. Frozen DC-SoC modified: **NO**. Frozen AHBN modified: **NO**. The only filesystem change is this audit document.

## 11. Final decision

**S1 PASS.** The paper and executable-code evidence are sufficient to define a bounded S2 revision scope. “PASS” does not mean the current comparator is paper-faithful; it means the fidelity gaps are now explicit enough to scope S2 without guessing.

### Proposed S2 candidate scope (not implemented)

**MUST IMPLEMENT**

1. Explicit core/leaf neighborhood and parent/child dissemination structure with core-driven push and inter-cluster routing.
2. Inactive-core replacement with local relationship transfer/notification, including Exp10 failure-path integration.
3. Returning former core as an ordinary leaf; explicit inactive grace/reinstate/expel lifecycle at simulator abstraction level.
4. On-demand missing-block recovery abstraction for new/reactivated peers, with separate `T_recovery` and traffic metrics.
5. Periodic `du` re-clustering/regeneration with barrier/cache only where experiment lifecycle and multi-block execution make it applicable.
6. Structural/cost counters: initial/recluster operations, replacements, repairs, recoveries, and changed edges; define the simulation-time model before applying delay.

**KEEP CURRENT**

1. Seed discipline and duplicate suppression/accounting.
2. Noise-node retention/nearest-cluster attachment.
3. No AHBN controller or runtime adaptive forwarding in DC-SoC.
4. Fixed bounded forwarding budget only if reconciled with core-only tree push and documented as the comparison resource policy.
5. Current `eps=2`, `min_samples=3` only as simulator calibration, not as paper parameter fidelity.

**INTENTIONALLY OMIT**

1. Social trust calculations and economic incentives.
2. Malicious-node/Sybil/Byzantine detection experiments.
3. Fabric-specific CA/MSP/channel, cryptographic, and storage implementation details.
4. Master buffering/push intervals unless Stage 4 expands beyond single-message dissemination.

No S2 code is included in this document.

## 12. Required terminal summary

```text
S1 — DC-SoC fidelity audit

Status: PASS

Paper mechanisms audited: 25
Exact/equivalent: 3
Simplified: 7
Missing dissemination-relevant: 10
Intentionally omitted: 4

Core replacement:
  Paper: YES
  Current implementation: NO (generic churn re-election is not faithful; Exp10 failure has none)
  S2 action: IMPLEMENT

Structural repair:
  Paper: YES
  Current implementation: SIMPLIFIED on churn, absent on Exp10 failure
  S2 action: IMPLEMENT local relationship transfer/notification

Periodic re-clustering:
  Paper: YES
  Current implementation: NO
  S2 action: IMPLEMENT only at justified lifecycle boundaries

Frozen DC-SoC modified: NO
Frozen AHBN modified: NO
Stage 4 rerun: NO
S2 started: NO

Documentation:
docs/S1_faithful_dcsoc.md
```
