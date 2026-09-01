"""Stage 3.5: record and verify the frozen DC-SoC comparison baseline."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Iterable

from ahbn.config import load_yaml_config
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage3_dcsoc.yaml"
CONSTRUCTION_PATHS = (PROJECT_ROOT / "run_one.py", PROJECT_ROOT / "run_batch.py")

# Stage 3 comparison design recorded here as part of the freeze. These are
# experiment dimensions, not tunable DC-SoC algorithm parameters.
TOPOLOGY_TYPE = "BA"
SUPPORTED_EXPERIMENT_SIZES = (30, 50, 100)


def _dcsoc_branch_nodes(path: Path) -> list[ast.AST]:
    """Return only strategy-name branches whose condition selects DC-SoC."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    branches: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        constants = {
            value.value
            for value in ast.walk(node.test)
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        }
        if "dcsoc" in constants:
            branches.extend(node.body)
    return branches


def _symbols(nodes: Iterable[ast.AST]) -> set[str]:
    symbols: set[str] = set()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                symbols.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                symbols.add(node.attr.lower())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                symbols.add(node.value.lower())
    return symbols


def _source_tree(obj) -> ast.Module:
    return ast.parse(inspect.getsource(obj))


def _contains_call(tree: ast.AST, function_name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name) and node.func.id == function_name
            or isinstance(node.func, ast.Attribute) and node.func.attr == function_name
        )
        for node in ast.walk(tree)
    )


def _strategy_defaults() -> dict[str, int]:
    signature = inspect.signature(DCSOCStrategy.__init__)
    return {
        "fanout": signature.parameters["fanout"].default,
        "inter_fanout": signature.parameters["inter_fanout"].default,
    }


def _parameter_consumption(symbols: set[str]) -> dict[str, bool]:
    groups = {
        "alpha": {"alpha"},
        "beta": {"beta"},
        "gamma": {"gamma"},
        "EWMA": {"ewma", "d_hat", "l_hat", "u_hat", "c_hat", "update_metrics"},
        "thresholds": {
            "threshold", "thresholds", "mode_threshold",
            "d0", "l0", "u0", "c0",
        },
        "mode score": {"mode_score", "score", "compute_score"},
        "adaptive fanout": {
            "adaptive_fanout", "min_fanout", "max_fanout",
            "decide_mode_and_fanout",
        },
        "AHBN node control state": {"control", "nodecontrolstate"},
    }
    return {label: bool(symbols & terms) for label, terms in groups.items()}


def _print_field(label: str, value) -> None:
    print(f"{label}:")
    print(value)


def main() -> None:
    errors: list[str] = []
    cfg = load_yaml_config(str(CONFIG_PATH))
    dcsoc_cfg = cfg.get("dcsoc")
    if not isinstance(dcsoc_cfg, dict):
        dcsoc_cfg = {}
        errors.append("required dcsoc configuration section is missing")

    required = ("eps", "min_samples", "fanout", "inter_fanout")
    missing = [name for name in required if name not in dcsoc_cfg]
    if missing:
        errors.append("required baseline parameters not identified: " + ", ".join(missing))

    defaults = _strategy_defaults()
    if not missing:
        if int(dcsoc_cfg["fanout"]) != defaults["fanout"]:
            errors.append("configured fanout does not match DCSOCStrategy default")
        if int(dcsoc_cfg["inter_fanout"]) != defaults["inter_fanout"]:
            errors.append("configured inter_fanout does not match DCSOCStrategy default")

    strategy_tree = _source_tree(DCSOCStrategy)
    cluster_tree = _source_tree(assign_dcsoc_clusters)
    construction_nodes = [
        node
        for path in CONSTRUCTION_PATHS
        for node in _dcsoc_branch_nodes(path)
    ]
    if not construction_nodes:
        errors.append("DC-SoC construction path could not be identified")

    inspected_nodes: list[ast.AST] = [strategy_tree, cluster_tree, *construction_nodes]
    symbols = _symbols(inspected_nodes)
    consumption = _parameter_consumption(symbols)

    if "ahbncontroller" in symbols or "ahbnstrategy" in symbols:
        errors.append("DC-SoC imports or constructs AHBN controller state")
    if consumption["adaptive fanout"]:
        errors.append("DC-SoC reads adaptive fanout")
    if consumption["EWMA"]:
        errors.append("DC-SoC reads EWMA metrics")
    if consumption["AHBN node control state"]:
        errors.append("DC-SoC reads AHBN node control state")

    topology_source = inspect.getsource(assign_dcsoc_clusters)
    head_rule_ok = (
        "highest_physical_degree" in topology_source
        and "original_neighbors" in topology_source
        and "-nid" in topology_source
    )
    forwarding_ok = all(
        token in _symbols([strategy_tree])
        for token in ("neighbors", "cluster_id", "gateway_neighbors", "fanout", "inter_fanout")
    )
    construction_ok = (
        _contains_call(ast.Module(body=construction_nodes, type_ignores=[]), "assign_dcsoc_clusters")
        and _contains_call(ast.Module(body=construction_nodes, type_ignores=[]), "DCSOCStrategy")
    )
    if not head_rule_ok:
        errors.append("cluster-head rule could not be identified")
    if not forwarding_ok:
        errors.append("fixed DC-SoC forwarding path could not be identified")
    if not construction_ok:
        errors.append("DC-SoC construction parameters could not be traced")

    overall_consumed = any(consumption.values())
    passed = not errors and not overall_consumed

    print("=" * 72)
    print("STAGE 3.5 — DC-SoC MINIMAL PARAMETER SANITY / FREEZE")
    print("=" * 72)
    print("\nDC-SoC baseline parameter snapshot:\n")
    _print_field("Topology type", TOPOLOGY_TYPE)
    _print_field("Supported experiment sizes", " / ".join(map(str, SUPPORTED_EXPERIMENT_SIZES)))
    _print_field("DBSCAN eps", dcsoc_cfg.get("eps", "UNIDENTIFIED"))
    _print_field("DBSCAN min_samples", dcsoc_cfg.get("min_samples", "UNIDENTIFIED"))
    _print_field("Cluster-head rule", "highest physical degree" if head_rule_ok else "UNIDENTIFIED")
    _print_field("Tie-break", "lowest node ID" if head_rule_ok else "UNIDENTIFIED")
    _print_field("Forwarding", "intra-cluster neighbour forwarding + CH gateway forwarding" if forwarding_ok else "UNIDENTIFIED")
    _print_field("Fanout", f"fixed (total={dcsoc_cfg.get('fanout', 'UNIDENTIFIED')}, CH gateway reserve={dcsoc_cfg.get('inter_fanout', 'UNIDENTIFIED')})")
    _print_field("Adaptive control", "disabled" if not overall_consumed else "DETECTED")

    print("\nAHBN parameter consumption:\n")
    for label in ("alpha", "beta", "gamma", "EWMA", "thresholds", "mode score", "adaptive fanout"):
        _print_field(label, "YES" if consumption[label] else "NO")
    print("\nOverall:\n")
    _print_field("AHBN parameters consumed", "YES" if overall_consumed else "NO")

    print("\nFreeze status:\n")
    _print_field("DC-SoC baseline", "FROZEN" if passed else "NOT_FROZEN")
    _print_field("AHBN", "FROZEN_SEPARATELY")
    _print_field("Comparison ready", "YES" if passed else "NO")

    result = {
        "stage": "3.5",
        "status": "PASS" if passed else "FAIL",
        "baseline": {
            "topology_type": TOPOLOGY_TYPE,
            "supported_experiment_sizes": list(SUPPORTED_EXPERIMENT_SIZES),
            "dbscan": {
                "eps": dcsoc_cfg.get("eps"),
                "min_samples": dcsoc_cfg.get("min_samples"),
            },
            "cluster_head_rule": "highest_physical_degree" if head_rule_ok else None,
            "tie_break": "lowest_node_id" if head_rule_ok else None,
            "forwarding": "intra_cluster_neighbours_and_ch_gateways" if forwarding_ok else None,
            "fanout": dcsoc_cfg.get("fanout"),
            "inter_fanout": dcsoc_cfg.get("inter_fanout"),
            "adaptive_control": False if not overall_consumed else None,
        },
        "ahbn_parameter_consumption": {
            label.lower().replace(" ", "_"): value
            for label, value in consumption.items()
        },
        "ahbn_parameters_consumed": overall_consumed,
        "comparison_ready": passed,
        "errors": errors,
    }
    print("\nMachine-readable result:\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\n" + "=" * 72)
    print(f"STAGE 3.5 RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 72)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
