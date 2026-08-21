# S2 — revised faithful dissemination-focused DC-SoC

## 1. Objective and scope

S2 revises only the DC-SoC comparator to preserve density-derived cluster membership, an explicit core-rooted propagation structure, core-driven push, local core replacement/repair, return-as-leaf, on-demand recovery, and explicit periodic `du` regeneration. It does not reproduce social trust, incentives, malicious-node defence, Fabric, cryptography, or batching. AHBN, Gossip, Structured, controller bounds, Stage 4 configurations, and Stage 4 results remain frozen.

The resulting architectural path is `network state → cluster/propagation structure → repair/replacement/re-clustering → dissemination`. DC-SoC never constructs or consults an AHBN controller and has no adaptive fanout.

## 2. S1 requirements used

The complete `docs/S1_faithful_dcsoc.md` was read before changes. S2 implements S1 items A5/A7, B2/B3, C3/C4/C5, D1, and E1 at dissemination abstraction level. Duplicate suppression, noise attachment, `eps=2.0`, `min_samples=3`, fixed `fanout=3`, and `inter_fanout=1` remain. Batching and the update cache are not active because Stage 4 is single-message-per-run; a `du` event is therefore an explicit barrier event and cannot corrupt an overlapping payload in the current execution model.

## 3. Baseline integrity

Pre-change `git status --short`, `git diff --stat`, and `git diff` were empty. Baseline SHA-256:

```text
999740f7262d9d918c16fe701e0c9da024be90e5a2e9ad95f41bc84a993d784a  ahbn/strategies/dcsoc.py
50ed8c10408bb5601ccd6f441b2aed834a3a427b00d434aea10a4222b72441db  ahbn/strategies/ahbn.py
9a19ae2c9766ea36fe873d4d643cf51d9e8df555b42d2de11d946d30fb60f75f  ahbn/control.py
3cff1c3ead4ef3dbec8c1f67dc30a18cc8bb8ef5eb15ad55d591f94e558b2d53  ahbn/simulator.py
916eaa0e21cec4d3982876a983b858e2216a67bf146faa65c3a46fa349892419  ahbn/topology.py
cd781c729b6b54138c2303caf8858b4d4eb8bb21bb4ab55c4b4645144ae8ab52  ahbn/node.py
bbf93b32a2cc8f2327f140fbc2eeeebf72fcb93729847993a714b98d2175b56d  ahbn/failure_injector.py
625faaf2945761c1699c662cbec31b70ac23a97abbb596c873d761a767eba891  ahbn/churn_manager.py
c9adcbdb20e6d8ae052b6de712fb11dea2e3364bdc50d33211c0e2d79f06c853  run_batch.py
0da5a733a01909e67591773ffdee1939ed7d79c833fab6afa57a31527502ab35  run_one.py
```

All legacy S1–S11 validators and `validate_dcsoc_s35_freeze` passed before implementation. The focused pre-change structure validator failed because all explicit role/parent/child/core-neighbour/edge/generation fields were absent. The focused push validator failed with leaf targets `[0, 5, 3]`, proving independent leaf gossip.

## 4. Minimal structural model and deterministic surrogates

Each node has a cluster, `core` or `leaf` role, optional parent, children, core neighbours, active state, and lifecycle state. `ClusterManager.structural_edges` is a deterministic directed acyclic propagation overlay with a structural generation. One elected core represents each DBSCAN cluster; cores form a directed inter-cluster routing chain and ordinary members attach as leaves. The election rule is highest original physical degree, tie lowest node ID. This replaces the paper's social-value election and hierarchy with a deterministic dissemination-only surrogate; it is not claimed to reproduce social election.

Leaves may send only to their assigned parent. Cores drive downstream propagation. The existing total fixed budget remains three targets per forwarding call; inter-cluster relationships are structural children within that same budget. Duplicate suppression is unchanged.

Core failure transfers the affected parent/child relationships to the deterministic eligible replacement in the same cluster. It does not re-cluster or rebuild unrelated regions. A returning former core is reinstated as a leaf; only a later explicit `du` regeneration can elect it again.

Recovery selects the lowest-ID active node that holds every missing item. This deterministic surrogate replaces social-value recovery source selection. One simulator event models the request/transfer using `base_delay + seeded jitter`; no Python wall-clock duration is used.

Re-clustering runs only on an explicit `dcsoc_recluster` (`du`) event. It re-evaluates online nodes with the retained DBSCAN calibration and increments the generation. Changed state may, but need not, change assignments/edges.

## 5. Cost and time accounting

Counters are `initial_clustering_count`, `recluster_count`, `core_replacement_count`, `structural_repair_count`, `recovery_count`, `topology_edges_changed`, `repair_control_events`, `recovery_request_count`, and `recovery_transfer_count`.

`T_propagation` remains the message metric's simulated first-to-last delivery interval. `T_recovery` is the accumulated simulator-event delay of recovery transfers. Local repair and periodic DBSCAN work expose counts and changed edges; `T_topology_update` remains zero because the simulator has no defensible native numerical computation-time model. No arbitrary seconds and no Python `time.time()`/`perf_counter()` enter propagation.

## 6. Files changed

`ahbn/node.py`, `ahbn/cluster.py`, `ahbn/topology.py`, `ahbn/strategies/dcsoc.py`, `ahbn/failure_injector.py`, and generic event plumbing in `ahbn/simulator.py`; four focused validators were added under `scripts/`. No other strategy, controller, runner, config, or Stage 4 output changed.

## 7. Validation and legacy reinterpretation

Commands used the required interpreter and `PYTHONDONTWRITEBYTECODE=1`:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_faithful_structure
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_core_driven_push
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_lifecycle_post_s2
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_reclustering
```

All focused validators pass. They cover structure, reciprocal references, acyclicity, leaf behaviour, replacement, local repair, unaffected-edge preservation, return-as-leaf, simulator-time recovery, both equivalent and changed re-clustering cases, deterministic replay, cost counters, and controller absence.

Post-change legacy S1/S2/S3/S7/S9/S10/S11 pass. S4/S5/S6/S8 are explicitly superseded: S4/S5 classify only physical edges plus the old bidirectional head chain as valid, S6 requires the old global churn reconstruction, and S8 requires leaf target sensitivity to physical-neighbour sampling. Those are precisely the simplified behaviours S2 removes. Their failing post-change outputs were retained during validation and the scripts were not deleted.

AHBN isolation hashes remain exactly baseline: `ahbn/strategies/ahbn.py` is `50ed8c...41db`, `ahbn/control.py` is `9a19ae...60f75f`, and both runners retain their baseline hashes. Gossip and Structured files are unchanged. `configs/exp10_failure.yaml` still excludes DC-SoC, as required for the later post-freeze experiment step.

## 8. Decision

S2 implementation and focused validation: **PASS**. Stage 4 rerun: **NO**. S3/S4 started: **NO**. AHBN modified: **NO**.
