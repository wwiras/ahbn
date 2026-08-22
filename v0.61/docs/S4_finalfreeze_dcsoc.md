# S4 — final freeze with faithful DC-SoC

## 1. Purpose and decision

S4 is a freeze-and-verify stage, not an experiment. It fixes the exact Gossip,
Structured, revised faithful DC-SoC, and canonical AHBN implementations and
parameters for the final Stage 4 comparison. No Stage 4 batch was run and no
raw result was created or overwritten.

**Decision: PASS.** The four comparator implementations are internally valid
and isolated. One non-tuning configuration defect in Exp09 was corrected by
making the already-frozen S2 DC-SoC parameters explicit. Exp10, Exp11, and
Exp12 still require a comparator-list applicability decision before rerun; that
is a configuration-only gate and does not reopen this algorithm freeze.

- S1 fidelity audit: **PASS**
- S2 faithful dissemination-focused revision: **PASS**
- S3 max-fanout sensitivity: **COMPLETE; canonical max remains 4**
- AHBN: **FROZEN**
- DC-SoC: **FROZEN**
- Gossip: **FROZEN**
- Structured: **FROZEN**

## 2. Authoritative records and fanout evidence

Read completely before any change:

- `docs/S1_faithful_dcsoc.md`
- `docs/S2_faithful_dcsoc.md`
- `docs/fanout6_result.md`

S1 established the fidelity gaps. S2 implemented the dissemination-focused
core/leaf structure, core push, lifecycle repair/replacement/recovery, and
explicit `du` regeneration while retaining simulator-calibrated `eps=2.0`,
`min_samples=3`, fixed `fanout=3`, and `inter_fanout=1`.

The max-fanout record identifies `[2,6]` as an amendment/sensitivity check,
not a canonical setting. In 120 validation runs, fanout 5 and 6 were never
reached; widening the bound mainly rescaled the normal operating fanout from 3
to 4. The record explicitly concludes that the evidence is insufficient to
amend the canonical range. Occurrences of 5/6 in Exp07 and
`validate_fanout6_amendment.py` are sensitivity-only and do not enter the
canonical AHBN execution path.

## 3. Files inspected

All files requested by S4 were inspected:

```text
ahbn/strategies/gossip.py
ahbn/strategies/cluster.py
ahbn/strategies/dcsoc.py
ahbn/strategies/ahbn.py
ahbn/control.py
ahbn/node.py
ahbn/cluster.py
ahbn/topology.py
ahbn/simulator.py
ahbn/failure_injector.py
ahbn/churn_manager.py
run_one.py
run_batch.py
configs/exp07_fanout.yaml
configs/exp08_ch_bottleneck.yaml
configs/exp09_dense_topology.yaml
configs/exp10_failure.yaml
configs/exp11_churn.yaml
configs/exp12_mixed_resources.yaml
configs/sanity_neutral.yaml
configs/sanity_overload.yaml
configs/sanity_churn.yaml
configs/stage2_parameter_sensitivity.yaml
configs/stage3_dcsoc.yaml
scripts/validate_dcsoc_*.py
scripts/validate_stage4_exp07_execution.py
scripts/validate_fanout6_amendment.py
```

## 4. Pre-freeze repository snapshot

Commands and complete relevant output:

```text
$ pwd
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

$ git rev-parse --show-toplevel
/Users/wwiras/Documents/src/AHBNProj/ahbn

$ git status --short
M  ahbn/cluster.py
M  ahbn/failure_injector.py
M  ahbn/node.py
M  ahbn/simulator.py
M  ahbn/strategies/dcsoc.py
M  ahbn/topology.py
A  docs/S2_faithful_dcsoc.md
A  scripts/validate_dcsoc_core_driven_push.py
A  scripts/validate_dcsoc_faithful_structure.py
A  scripts/validate_dcsoc_lifecycle_post_s2.py
A  scripts/validate_dcsoc_reclustering.py

$ git diff --stat
[no unstaged output; the S2 changes above were staged]

$ git diff
[no unstaged output; the S2 changes above were staged]
```

The dirty state is the validated S2 implementation and its records/validators;
it was preserved. No unrelated user file was cleaned or discarded.

## 5. AHBN freeze verification

Executable defaults and final configuration agree:

```text
alpha=0.30
d0=l0=u0=c0=0.50
w_d=-1.0, w_l=+1.0, w_u=-1.0, w_c=+1.0
kappa=1.0, beta=1.0, tau/mode_threshold=0.50
min_fanout=2, max_fanout=4, default_fanout=3
```

`AHBNController` retains the EWMA `alpha*new+(1-alpha)*old`, centered score,
stable logistic sigmoid, threshold mode selection, and rounded/clamped fanout
rule. `AHBNStrategy` alone consumes `node.control.mode/fanout`; it delegates to
bounded Gossip or Structured forwarding. The S2 hashes recorded in
`docs/S2_faithful_dcsoc.md` exactly match current `ahbn.py`, `control.py`, and
both runners, proving S2 did not alter canonical AHBN or its construction.
Emergency/failure sensing remains the existing generic simulator behavior; no
DC-SoC repair or regeneration is invoked by AHBN.

Validation output:

```text
$ /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_stage4_exp07_execution
Gossip fixed-fanout sweep: [2, 3, 4, 5, 6]
Gossip scheduled runs    : 100
AHBN scheduled runs      : 20
AHBN receives sweep value: NO
AHBN min_fanout          : 2
AHBN max_fanout          : 4
AHBN default_fanout      : 3
Expected total runs      : 120
EXP07 EXECUTION VALIDATION: PASS

$ /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -c '<canonical parameter and neutral-equation assertions>'
AHBN canonical parameter/equation/controller assertions: PASS
```

**AHBN: FROZEN**

## 6. Revised faithful DC-SoC freeze verification

Frozen parameters are `eps=2.0`, `min_samples=3`, `fanout=3`, and
`inter_fanout=1`. They are simulator calibrations; paper `eps=3.0` is not
substituted.

The S2 implementation contains explicit deterministic core/leaf relationships
and a directed acyclic structural overlay. Leaves only uplink to their parent;
cores drive downstream and inter-cluster forwarding within the fixed budget.
It provides local core replacement, local relationship repair, former-core
return as leaf, simulator-event recovery, and explicit `dcsoc_recluster`/`du`
regeneration. It records clustering, replacement, repair, recovery, changed
edge, control-event, request, and transfer counters. Recovery delay uses the
simulation clock (`base_delay + seeded jitter`); there is no arbitrary topology
delay and no Python wall-clock timing. The DC-SoC construction path passes
`controller=None`; it has no EWMA, score, mode controller, or adaptive fanout.

Focused S2 validator output:

```text
$ .../venv0.6/bin/python -m scripts.validate_dcsoc_faithful_structure
PASS — all active nodes covered
PASS — acyclic
PASS — no self loops
PASS — no duplicate edges
PASS — reciprocal parent/children
PASS — at least two cores
PASS — ordinary leaves present
PASS — explicit DC-SoC propagation structure is valid

$ .../venv0.6/bin/python -m scripts.validate_dcsoc_core_driven_push
leaf=1 parent=0 targets=[0]
PASS — ordinary leaf does not independently fan out

$ .../venv0.6/bin/python -m scripts.validate_dcsoc_lifecycle_post_s2
PASS — core replaced
PASS — local repair counted
PASS — unaffected relationships retained
PASS — former core returns as leaf
PASS — simulator-time recovery
PASS — explicit du regeneration
PASS — AHBN controller calls
failed=3 replacement=0 repair=1 changed_edges=20 recovery_time=1.0 generation=2

$ .../venv0.6/bin/python -m scripts.validate_dcsoc_reclustering
PASS — equivalent update stays equivalent
PASS — changed inputs regenerate structure
PASS — fixed-input determinism
recluster_count=2 generation=3 topology_edges_changed=8
```

Applicable legacy validators produced:

```text
validate_dcsoc_s1:  S1 PASS
validate_dcsoc_s2:  S2 RESULT: PASS
validate_dcsoc_s3:  S3 RESULT: PASS
validate_dcsoc_s7:  S7 RESULT: PASS; controller calls 0
validate_dcsoc_s9:  S9 RESULT: PASS; deterministic and AHBN-independent
validate_dcsoc_s10: Overall S10 result: PASS
validate_dcsoc_s11: S11 RESULT: PASS; simulator.controller=None
validate_dcsoc_s35_freeze: STAGE 3.5 RESULT: PASS
```

Legacy S4/S5/S6/S8 remain superseded exactly as S2 documents; they encode the
removed simplified leaf-gossip/global-rebuild behavior and were not used as
freeze gates.

**DC-SoC: FROZEN**

## 7. Gossip freeze verification

Standalone Gossip is unstructured uniform sampling of up to fixed `fanout`
active physical neighbors with the seeded simulator RNG. Its parameter source
is the experiment's global/static fanout (default 3). It has no controller,
cluster structure, structural repair, or runtime adaptation. AHBN owns a
separate internal `GossipStrategy` instance and may set that instance's fanout;
this does not mutate the standalone comparator.

**Gossip: FROZEN**

## 8. Structured freeze verification

`assign_static_clusters` supplies deterministic static cluster membership and
heads. A member forwards only to its head. A head forwards to all active local
members plus gateway heads. The standalone `ClusterStrategy(fanout=None)` is
unbounded by a runtime controller; the optional budget is used only by AHBN's
private Structured delegate. No DC-SoC roles, replacement, recovery, or `du`
regeneration are consulted.

**Structured: FROZEN**

## 9. Cross-algorithm isolation matrix

| Comparator | AHBN controller | DC-SoC repair/regeneration | Structured-only cluster behavior | Runtime adaptive fanout | Independent leaf gossip |
|---|---:|---:|---:|---:|---:|
| Gossip | No | No | No | No | N/A |
| Structured | No | No | Yes | No | No |
| DC-SoC | No | Yes | No | No | No |
| AHBN | Yes | No | Selected only in AHBN cluster mode | Yes, `[2,4]` | N/A |

Shared `Node`, `Simulator`, failure, churn, and cluster fields are generic
plumbing. DC-SoC lifecycle events are guarded by the DC-SoC strategy/structure;
controller updates return inertly when `controller=None`. Runtime sentinels in
S7/S11 observed zero AHBN controller calls for DC-SoC.

## 10. Parameter and behavior freeze matrix

| Comparator | Structure | Runtime adaptation | Fanout | Key parameters | Controller | Structural regeneration |
|---|---|---|---|---|---|---|
| Gossip | Unstructured physical neighbors | None | Static experiment fanout; default 3 | Seeded uniform selection | None | None |
| Structured | Static clusters, member→head, head→members/gateways | None | Standalone returns all valid structured targets | Static cluster count/topology | None | None |
| DC-SoC | Density-derived explicit core/leaf DAG | Repair and explicit `du` only | Fixed 3 total; inter portion 1 | `eps=2.0`, `min_samples=3` | None | Yes, explicit `du` |
| AHBN | Static underlying clusters/topology | Mode and forwarding budget | `[2,4]`, default 3 | canonical alpha/centres/weights/kappa/beta/threshold | Canonical AHBN | None |

## 11. Final Stage 4 configuration inspection

Resolved inspection output after the only S4 config reconciliation:

```text
exp07: strategies=[gossip, ahbn], fanout sweep=[2,3,4,5,6]; AHBN does not receive sweep
exp08: strategies=[gossip, cluster, dcsoc, ahbn]; dcsoc={2.0,3,3,1}; AHBN=[2,4]
exp09: strategies=[gossip, cluster, dcsoc, ahbn]; global fanout=4;
       dcsoc={eps:2.0,min_samples:3,fanout:3,inter_fanout:1}; AHBN=[2,4]
exp10: strategies=[gossip, cluster, ahbn]; global fanout=3; AHBN=[2,4]
exp11: strategies=[gossip, cluster, ahbn]; global fanout=3; AHBN=[2,4]
exp12: strategies=[gossip, cluster, ahbn]; global fanout=3; AHBN=[2,4]
```

Before reconciliation, Exp09's global `fanout: 4` flowed into DC-SoC through
the runner fallback. S4 added only an explicit `dcsoc` block with the already
frozen S2 values. The global 4 remains the scientifically defined Gossip dense
condition. No algorithm file or parameter value was tuned.

Exp10 still excludes DC-SoC, as S1/S2 warned. Exp11 and Exp12 also exclude it.
Before final reruns, decide and document scientific applicability for each; if
all four are applicable, add `dcsoc` to the comparator list and copy the exact
frozen S2 block. This action is mandatory before rerun but is not authorization
to alter any comparator implementation or parameter.

## 12. No-post-hoc-tuning classification

- AHBN values in Exp08–Exp12 and sanity configs: **CANONICAL**.
- Exp07 Gossip fanout `[2,3,4,5,6]`: **EXPERIMENTAL CONDITION**; the validator
  proves these values do not enter AHBN.
- `validate_fanout6_amendment.py` in-memory max 6: **SENSITIVITY-ONLY**.
- Stage 2 OFAT sweeps: **SENSITIVITY-ONLY**.
- Exp09 global Gossip fanout 4: **EXPERIMENTAL CONDITION**.
- Exp09 DC-SoC fallback to 4: **STALE/INVALID**, corrected config-only to 3.
- DC-SoC eps/min-samples/fanout/inter-fanout on final paths: **CANONICAL**.
- No remaining max 5/6 occurs on a final canonical AHBN path.
- No comparator-specific runtime tuning hook was found.

## 13. Regression and smoke results

All Python commands used exactly the required interpreter and set
`PYTHONDONTWRITEBYTECODE=1`. In addition to the validators above, four
fixed-seed `run_one.py` constructions used `configs/sanity_neutral.yaml`.
They were explicitly labeled **SMOKE TEST ONLY — NOT EXPERIMENTAL RESULT** and
did not save output:

```text
gossip:  constructed and completed; delivery_ratio=0.9411764705882353
cluster: constructed and completed; delivery_ratio=1.0
dcsoc:   constructed and completed; delivery_ratio=0.23529411764705882
ahbn:    constructed and completed; delivery_ratio=0.9411764705882353
```

The DC-SoC smoke delivery is not an experimental result or performance gate;
the focused validators prove the intended bounded core-driven structure.

## 14. Final SHA-256 freeze manifest

```text
114a6311ef17b10f2fda12ac696402f2b383c849d663c3a66861a26bb6fb1664  ahbn/strategies/gossip.py
0c5a421434806099e43fbc3f118f6737920fc155e750c2100492e6dbe93c027c  ahbn/strategies/cluster.py
e0c1b109427cdd5e6a0055da350a2b9a7f2003af2b9f3da58e56dff73a3ac58e  ahbn/strategies/dcsoc.py
50ed8c10408bb5601ccd6f441b2aed834a3a427b00d434aea10a4222b72441db  ahbn/strategies/ahbn.py
9a19ae2c9766ea36fe873d4d643cf51d9e8df555b42d2de11d946d30fb60f75f  ahbn/control.py
4968331fb013ac47b1513c7da5c3555b1c0edbfe3c4ee4690f0a604732df2f3a  ahbn/node.py
afe97a053e2d0e3414906c29c079a7e6b5582eccf8a79e540bc2b88a36e5f50d  ahbn/cluster.py
ddaddc522af2f5f24f4b2109e778f59637bbc3a9183d4c26ce4ce7248527d7fc  ahbn/topology.py
46009c14765667fa147909abdc6f8d3b4226b23933e4615a1e47f0248aa31c53  ahbn/simulator.py
556327f690ff20e50821aa08eb87a0a870ff198db5012c2039c2030fd9839830  ahbn/failure_injector.py
625faaf2945761c1699c662cbec31b70ac23a97abbb596c873d761a767eba891  ahbn/churn_manager.py
0da5a733a01909e67591773ffdee1939ed7d79c833fab6afa57a31527502ab35  run_one.py
c9adcbdb20e6d8ae052b6de712fb11dea2e3364bdc50d33211c0e2d79f06c853  run_batch.py
922c79af378f679a26e6efc23a234f37c6d09e76bc281c3e36b6b62a93b788a2  configs/exp07_fanout.yaml
4d18537eb8c02b5b1208c0554b8544d88465367827dac2dde32c23fc6479966f  configs/exp08_ch_bottleneck.yaml
e50b78929ff7079e6cb494a5b71792564f7e2922ec25527ea95db0f6bfe3da61  configs/exp09_dense_topology.yaml
deef57bdfae2c0799e70fa0cc91b6a2b1544e5eb348e4a83700b6c33da2809e5  configs/exp10_failure.yaml
05a32cf9189919b8b1b439ac69968245774962ba234803e16cb732efb9e95135  configs/exp11_churn.yaml
763faf75a63baa52054f88b60056f0eec2e15adddff8ef06277cfb1955a03e9e  configs/exp12_mixed_resources.yaml
```

These algorithm, shared execution, runner, and configuration files are frozen.
The only permitted pre-rerun edits are documented comparator-list inclusion
and the exact copied S2 DC-SoC block where scientifically applicable. Such
edits require an updated config hash appended to this record before execution.

## 15. Final freeze declaration

**AHBN: FROZEN**  
**DC-SoC: FROZEN**  
**Gossip: FROZEN**  
**Structured: FROZEN**

> From S4 PASS onward, no algorithmic implementation or comparator parameter
> may be changed in response to observed Stage 4 comparative results. Any
> necessary algorithmic correction requires reopening the freeze and rerunning
> all affected final experiments.

S4 final decision: **PASS**, subject only to the documented pre-rerun
comparator-list applicability reconciliation for Exp10–Exp12. The final Stage 4
rerun has **not** started.
