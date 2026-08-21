"""Configuration-only gate for the final Stage 4 Exp08--Exp12 rerun."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "exp08_ch_bottleneck.yaml",
    "exp09_dense_topology.yaml",
    "exp10_failure.yaml",
    "exp11_churn.yaml",
    "exp12_mixed_resources.yaml",
)
EXPECTED_STRATEGIES = ["gossip", "cluster", "dcsoc", "ahbn"]
EXPECTED_DCSOC = {
    "eps": 2.0,
    "min_samples": 3,
    "fanout": 3,
    "inter_fanout": 1,
}


def main() -> int:
    runner_text = (ROOT / "run_batch.py").read_text(encoding="utf-8")
    failures: list[str] = []

    for strategy in EXPECTED_STRATEGIES:
        if f'elif strategy_name == "{strategy}"' not in runner_text and not (
            strategy == "gossip" and 'if strategy_name == "gossip"' in runner_text
        ):
            failures.append(f"run_batch.py does not support strategy {strategy!r}")

    for filename in CONFIGS:
        path = ROOT / "configs" / filename
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        label = str(cfg.get("experiment", filename)).upper()
        strategies = cfg.get("strategies")
        dcsoc = cfg.get("dcsoc")
        ahbn = cfg.get("ahbn")

        if strategies != EXPECTED_STRATEGIES:
            failures.append(f"{label}: strategies={strategies!r}")
        if strategies and len(strategies) != len(set(strategies)):
            failures.append(f"{label}: duplicate comparator entry")
        if dcsoc != EXPECTED_DCSOC:
            failures.append(f"{label}: explicit dcsoc block={dcsoc!r}")
        if not isinstance(ahbn, dict):
            failures.append(f"{label}: missing ahbn block")
            continue
        for key, expected in {
            "alpha": 0.3,
            "kappa": 1.0,
            "beta": 1.0,
            "mode_threshold": 0.5,
            "min_fanout": 2,
            "max_fanout": 4,
            "default_fanout": 3,
        }.items():
            if ahbn.get(key) != expected:
                failures.append(f"{label}: ahbn.{key}={ahbn.get(key)!r}")

        # Mirrors both runner fallbacks. An explicit complete block must win over
        # any experiment-level fanout (notably Exp09's Gossip condition of 4).
        resolved = {
            "eps": dcsoc.get("eps", cfg.get("dcsoc_eps", 2.0)) if dcsoc else None,
            "min_samples": dcsoc.get(
                "min_samples", cfg.get("dcsoc_min_samples", 3)
            ) if dcsoc else None,
            "fanout": dcsoc.get("fanout", cfg.get("fanout", 3)) if dcsoc else None,
            "inter_fanout": dcsoc.get(
                "inter_fanout", cfg.get("dcsoc_inter_fanout", 1)
            ) if dcsoc else None,
        }
        if resolved != EXPECTED_DCSOC:
            failures.append(f"{label}: resolved dcsoc={resolved!r}")

        print(
            f"{label}: strategies={strategies}; dcsoc={resolved}; "
            f"ahbn.max_fanout={ahbn.get('max_fanout')}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print("STAGE 4 PRE-RUN COMPARATOR VALIDATION: FAIL")
        return 1

    print("STAGE 4 PRE-RUN COMPARATOR VALIDATION: PASS")
    print("No simulations were run and no result files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
