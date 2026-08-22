#!/usr/bin/env python3
"""Focused pre-execution regression gate for the v0.61 Exp08 workflow."""
from __future__ import annotations
import hashlib, inspect, sys
from dataclasses import fields
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahbn.config import load_yaml_config
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.utils import Exp08ExecutionEvidence
from run_batch import build_ahbn_params, run_single
CFG = load_yaml_config(ROOT / "configs/exp08_ch_bottleneck.yaml")
POLICY_FILES = [ROOT / x for x in (
    "ahbn/strategies/gossip.py", "ahbn/strategies/cluster.py",
    "ahbn/strategies/dcsoc.py", "ahbn/strategies/ahbn.py", "ahbn/control.py")]

def digest() -> str:
    h = hashlib.sha256()
    for path in POLICY_FILES: h.update(path.read_bytes())
    return h.hexdigest()

def run(name: str) -> dict:
    return run_single(cfg=CFG, strategy_name=name, seed=42,
        topology_type=CFG["topology_type"], num_nodes=CFG["num_nodes"],
        use_topology_cache=CFG["use_topology_cache"], base_delay=CFG["base_delay"],
        jitter=CFG["jitter"], message_source=CFG["message_source"],
        num_clusters=CFG["num_clusters"], ch_overload_factor=3.0,
        ba_m=CFG["ba_m"], enable_adaptive_trace=(name == "ahbn"),
        scenario_tag="workflow-regression")

def main() -> int:
    policy_before = digest()
    runs = {name: run(name) for name in CFG["strategies"]}
    replay = run("dcsoc")
    d = runs["dcsoc"]
    evidence_fields = {f.name for f in fields(Exp08ExecutionEvidence)}
    required = {"strategy", "seed", "overload_factor", "topology_seed", "topology_identity",
        "effective_message_source", "dcsoc_master", "dcsoc_eligible_overload_nodes",
        "dcsoc_selected_overload_node", "dcsoc_selected_overload_role"}
    checks = {
        "EXP08-01 v0.61 paths only": ROOT.name == "v0.61",
        "EXP08-02 four comparators configured": CFG["strategies"] == ["gossip", "cluster", "dcsoc", "ahbn"],
        "EXP08-03 factors exact": [float(x) for x in CFG["ch_overload_factor"]] == [1.0, 1.5, 2.0, 3.0],
        "EXP08-04 seeds exact": [CFG["seed"] + i for i in range(CFG["runs_per_setting"])] == list(range(42, 62)),
        "EXP08-05 Gossip sweep absent": GossipStrategy().fanout is None and "fanouts" not in CFG,
        "EXP08-06 Structured uncapped": ClusterStrategy().fanout is None,
        "EXP08-07 DC-SoC uncapped": "fanout" not in inspect.signature(DCSOCStrategy).parameters and d["max_structural_obligations"] > 3,
        "EXP08-08 DC-SoC source Master": d["effective_source_id"] == d["dcsoc_master_id"],
        "EXP08-09 DC-SoC target Master/Core": d["dcsoc_selected_overload_role"] in {"Master", "Core"},
        "EXP08-10 Tail never eligible": d["dcsoc_overload_target_id"] in d["dcsoc_eligible_overload_nodes"],
        "EXP08-11 deterministic target": d["dcsoc_overload_target_id"] == replay["dcsoc_overload_target_id"],
        "EXP08-12 AHBN frozen [2,4]": (build_ahbn_params(CFG).min_fanout, build_ahbn_params(CFG).max_fanout) == (2, 4),
        "EXP08-13 topology pairing": len({x["topology_identity"] for x in runs.values()}) == 1,
        "EXP08-14 evidence fields": required <= evidence_fields,
        "EXP08-15 evidence is observational": policy_before == digest(),
    }
    print("Exp08 S3 workflow regression gate")
    for name, passed in checks.items(): print(f"{'PASS' if passed else 'FAIL'}  {name}")
    overall = all(checks.values())
    print(f"WORKFLOW RESULT: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1
if __name__ == "__main__": raise SystemExit(main())
