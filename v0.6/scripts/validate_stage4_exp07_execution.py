from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.config import load_yaml_config
from run_batch import build_ahbn_params, build_ahbn_strategy, exp07


def _fake_run_single(**kwargs) -> dict:
    summary = {
        "delivery_ratio": 1.0,
        "propagation_delay": 0.0,
        "duplicates": 0,
        "total_forwards": 0,
    }
    if kwargs["enable_adaptive_trace"]:
        summary["adaptive_trace_rows"] = []
    return summary


def main() -> None:
    cfg = load_yaml_config("configs/exp07_fanout.yaml")

    with patch("run_batch.run_single", side_effect=_fake_run_single) as mocked:
        rows, _ = exp07(cfg)

    calls = [call.kwargs for call in mocked.call_args_list]
    gossip_calls = [call for call in calls if call["strategy_name"] == "gossip"]
    ahbn_calls = [call for call in calls if call["strategy_name"] == "ahbn"]
    gossip_rows = [row for row in rows if row.strategy == "gossip"]
    ahbn_rows = [row for row in rows if row.strategy == "ahbn"]

    fanouts = list(cfg["fanouts"])
    runs = int(cfg["runs_per_setting"])
    params = build_ahbn_params(cfg)
    strategy = build_ahbn_strategy(cfg, fanout=None)

    checks = {
        "gossip_sweep": sorted({call["fanout"] for call in gossip_calls}) == fanouts,
        "gossip_runs": len(gossip_calls) == len(fanouts) * runs,
        "ahbn_runs": len(ahbn_calls) == runs,
        "ahbn_no_sweep": all(call["fanout"] is None for call in ahbn_calls),
        "ahbn_bounds": (params.min_fanout, params.max_fanout) == (2, 4),
        "ahbn_default": strategy.default_fanout == 3,
        "ahbn_adaptive": strategy.adaptive_fanout is True,
        "ahbn_label": all(row.fanout is None for row in ahbn_rows),
        "gossip_labels": sorted({row.fanout for row in gossip_rows}) == fanouts,
        "total_runs": len(rows) == 120,
    }

    print("=" * 72)
    print("STAGE 4 — EXP07 EXECUTION VALIDATION")
    print("=" * 72)
    print(f"Gossip fixed-fanout sweep: {fanouts}")
    print(f"Gossip scheduled runs    : {len(gossip_calls)}")
    print(f"AHBN scheduled runs      : {len(ahbn_calls)}")
    print(f"AHBN receives sweep value: {'NO' if checks['ahbn_no_sweep'] else 'YES'}")
    print(f"AHBN min_fanout          : {params.min_fanout}")
    print(f"AHBN max_fanout          : {params.max_fanout}")
    print(f"AHBN default_fanout      : {strategy.default_fanout}")
    print(f"AHBN result fanout       : {ahbn_rows[0].fanout if ahbn_rows else 'missing'}")
    print(f"Expected total runs      : {len(rows)}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"Failed checks            : {', '.join(failed)}")
        print("=" * 72)
        print("EXP07 EXECUTION VALIDATION: FAIL")
        print("=" * 72)
        raise SystemExit(1)

    print("=" * 72)
    print("EXP07 EXECUTION VALIDATION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
