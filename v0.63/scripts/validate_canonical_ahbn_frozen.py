from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.control import AHBNController, AHBNParams, NodeControlState
from run_batch import build_ahbn_params as build_batch_params
from run_one import build_ahbn_params as build_one_params


def assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (
        actual,
        expected,
    )


def validate_defaults(params: AHBNParams) -> None:
    assert_close(params.alpha, 0.30)
    assert (params.d0, params.l0, params.u0, params.c0) == (0.0, 0.0, 0.0, 0.0)
    assert (params.w_d, params.w_l, params.w_u, params.w_c) == (-1.0, 1.0, 1.0, 1.0)
    assert_close(params.kappa, 1.0)
    assert_close(params.beta, 1.0)
    assert_close(params.mode_threshold, 0.5)
    assert (params.min_fanout, params.max_fanout) == (2, 6)


def validate_anchor(observations: tuple[float, float, float, float], expected: tuple[float, str, int]) -> None:
    controller = AHBNController(AHBNParams())
    state = NodeControlState(
        d_hat=observations[0],
        l_hat=observations[1],
        u_hat=observations[2],
        c_hat=observations[3],
    )
    controller.decide_mode_and_fanout(state)
    expected_score, expected_mode, expected_fanout = expected
    assert_close(state.score, expected_score)
    assert_close(state.weight, AHBNController.sigmoid(expected_score))
    assert state.mode == expected_mode
    assert state.fanout == expected_fanout


def validate_score_boundary(score: float, expected_fanout: int) -> None:
    controller = AHBNController(AHBNParams())
    state = NodeControlState()
    controller.compute_score = lambda _state: score
    controller.decide_mode_and_fanout(state)
    assert_close(state.score, score)
    assert state.fanout == expected_fanout, (score, state.fanout, expected_fanout)


def validate_directional_signs() -> None:
    controller = AHBNController(AHBNParams())
    neutral = NodeControlState()
    base = controller.compute_score(neutral)
    assert controller.compute_score(NodeControlState(d_hat=0.1)) < base
    assert controller.compute_score(NodeControlState(l_hat=0.1)) > base
    assert controller.compute_score(NodeControlState(u_hat=0.1)) > base
    assert controller.compute_score(NodeControlState(c_hat=0.1)) > base


def main() -> None:
    for params in (AHBNParams(), build_batch_params({}), build_one_params({})):
        validate_defaults(params)

    validate_anchor((0.0, 0.0, 0.0, 0.0), (0.0, "gossip", 3))
    validate_anchor((0.5, 0.0, 0.0, 0.0), (-0.5, "cluster", 2))
    validate_anchor((0.0, 0.5, 0.0, 0.0), (0.5, "gossip", 4))
    validate_anchor((0.0, 0.0, 0.5, 0.0), (0.5, "gossip", 4))
    validate_anchor((0.25, 0.0, 0.0, 0.0), (-0.25, "cluster", 2))
    validate_anchor((0.0, 0.25, 0.0, 0.0), (0.25, "gossip", 4))

    for score, fanout in (
        (-0.50, 2),
        (-0.25, 2),
        (-0.249999, 3),
        (0.0, 3),
        (0.249999, 3),
        (0.25, 4),
        (0.899999, 4),
        (0.90, 5),
        (1.499999, 5),
        (1.50, 6),
        (2.0, 6),
    ):
        validate_score_boundary(score, fanout)

    validate_directional_signs()

    print("PASS: frozen canonical AHBN defaults, signs, mode, and final GKE actuator boundaries")


if __name__ == "__main__":
    main()
