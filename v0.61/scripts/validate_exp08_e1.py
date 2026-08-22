#!/usr/bin/env python3
"""Non-authoritative deterministic bottleneck preflight for v0.61 Exp08."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahbn.config import load_yaml_config
from run_batch import build_ahbn_params, run_single
CFG = load_yaml_config(ROOT / "configs/exp08_ch_bottleneck.yaml")

def run(strategy: str, factor: float) -> dict:
    return run_single(cfg=CFG, strategy_name=strategy, seed=42,
        topology_type=CFG["topology_type"], num_nodes=CFG["num_nodes"],
        use_topology_cache=CFG["use_topology_cache"], base_delay=CFG["base_delay"],
        jitter=CFG["jitter"], message_source=CFG["message_source"],
        num_clusters=CFG["num_clusters"], ch_overload_factor=factor,
        ba_m=CFG["ba_m"], enable_adaptive_trace=(strategy == "ahbn"),
        scenario_tag=f"e1-factor={factor}")

def main() -> int:
    before_params = deepcopy(vars(build_ahbn_params(CFG)))
    pairs = {name: (run(name, 1.0), run(name, 3.0)) for name in CFG["strategies"]}
    replay = run("dcsoc", 3.0)
    d0, d3 = pairs["dcsoc"]
    checks = {
        "Gossip runs without synthetic CH": pairs["gossip"][0]["delivery_ratio"] >= 0 and pairs["gossip"][0]["dcsoc_master_id"] is None,
        "Structured pressure increases delay": pairs["cluster"][1]["propagation_delay"] > pairs["cluster"][0]["propagation_delay"],
        "DC-SoC structure precedes eligible selection": bool(d3["dcsoc_eligible_overload_nodes"]),
        "DC-SoC source remains Master": d3["effective_source_id"] == d3["dcsoc_master_id"],
        "DC-SoC target eligible": d3["dcsoc_overload_target_id"] in d3["dcsoc_eligible_overload_nodes"],
        "DC-SoC target not Tail": d3["dcsoc_selected_overload_role"] in {"Master", "Core"},
        "DC-SoC deterministic replay": d3["dcsoc_overload_target_id"] == replay["dcsoc_overload_target_id"],
        "DC-SoC pressure increases delay": d3["propagation_delay"] > d0["propagation_delay"],
        "DC-SoC structural coverage retained": d3["total_forwards"] == d0["total_forwards"],
        "DC-SoC obligations uncapped": d3["max_structural_obligations"] > 3,
        "AHBN parameters unchanged": before_params == vars(build_ahbn_params(CFG)),
        "paired physical topology": len({pair[0]["topology_identity"] for pair in pairs.values()}) == 1,
    }
    print("E1 — v0.61 Exp08 bottleneck preflight (non-authoritative)")
    for name, passed in checks.items(): print(f"{'PASS' if passed else 'FAIL'}  {name}")
    overall = all(checks.values())
    print(f"E1 RESULT: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1
if __name__ == "__main__": raise SystemExit(main())
