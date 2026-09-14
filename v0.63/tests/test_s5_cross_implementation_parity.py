from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ahbn.control import AHBNController, AHBNParams, NodeControlState


CASES = (
    pytest.param(-0.30, 2, id="exact-minus-0.30"),
    pytest.param(-0.25, 2, id="exact-minus-0.25"),
    pytest.param(0.00, 3, id="exact-0.00"),
    pytest.param(0.25, 4, id="exact-0.25"),
    pytest.param(0.50, 4, id="exact-0.50"),
    pytest.param(0.90, 5, id="exact-0.90"),
    pytest.param(1.00, 5, id="exact-1.00"),
    pytest.param(1.50, 6, id="exact-1.50"),
    pytest.param(1.60, 6, id="exact-1.60"),
    pytest.param(-0.250001, 2, id="adjacent-minus-0.250001"),
    pytest.param(-0.250000, 2, id="adjacent-minus-0.250000"),
    pytest.param(-0.249999, 3, id="adjacent-minus-0.249999"),
    pytest.param(0.249999, 3, id="adjacent-0.249999"),
    pytest.param(0.250000, 4, id="adjacent-0.250000"),
    pytest.param(0.250001, 4, id="adjacent-0.250001"),
    pytest.param(0.899999, 4, id="adjacent-0.899999"),
    pytest.param(0.900000, 5, id="adjacent-0.900000"),
    pytest.param(0.900001, 5, id="adjacent-0.900001"),
    pytest.param(1.499999, 5, id="adjacent-1.499999"),
    pytest.param(1.500000, 6, id="adjacent-1.500000"),
    pytest.param(1.500001, 6, id="adjacent-1.500001"),
)


def _load_requested_fanout(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load production policy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.requested_fanout


GKE_APP = (
    Path(__file__).resolve().parents[4]
    / "AHBN_GKEProj"
    / "ahbn2_gke"
    / "app"
)
GKE_REQUESTED_FANOUT = {
    name: _load_requested_fanout(name, GKE_APP / f"{name}_final_actuator_policy.py")
    for name in ("k5", "k6", "k7")
}


def _controlsim_requested_fanout(score: float) -> int:
    if score < 0.0:
        state = NodeControlState(d_hat=-score)
    else:
        state = NodeControlState(
            l_hat=min(score, 1.0),
            u_hat=max(score - 1.0, 0.0),
        )
    AHBNController(AHBNParams()).decide_mode_and_fanout(state)
    assert state.score == score
    return state.fanout


@pytest.mark.parametrize(("score", "expected"), CASES)
def test_requested_s5_cross_implementation_parity(score: float, expected: int) -> None:
    controlsim = _controlsim_requested_fanout(score)
    k5 = GKE_REQUESTED_FANOUT["k5"]("S5", score)
    k6 = GKE_REQUESTED_FANOUT["k6"]("S5", score)
    k7 = GKE_REQUESTED_FANOUT["k7"]("S5", score)

    assert controlsim == k5 == k6 == k7 == expected
