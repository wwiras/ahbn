from __future__ import annotations
from typing import Optional

import math
from dataclasses import dataclass


# ============================================================
# Per-node AHBN controller state
# ============================================================

@dataclass
class NodeControlState:
    # EWMA-smoothed normalized observations [0, 1]
    d_hat: float = 0.0   # duplicate pressure
    l_hat: float = 0.0   # latency pressure
    u_hat: float = 0.0   # utilization / processing pressure
    c_hat: float = 0.0   # churn / instability pressure

    # Controller outputs
    score: float = 0.0
    weight: float = 0.5

    # Dissemination decision
    mode: str = "gossip"
    fanout: int = 3


# ============================================================
# Canonical AHBN parameters
# ============================================================

@dataclass
class AHBNParams:

    # --------------------------------------------------------
    # Stage 1: EWMA smoothing
    # --------------------------------------------------------
    # Provisional value.
    # Final value will be selected through alpha sensitivity.
    alpha: float = 0.3

    # --------------------------------------------------------
    # Stage 0: neutral threshold centres
    # --------------------------------------------------------
    # Frozen canonical centres.
    d0: float = 0.0
    l0: float = 0.0
    u0: float = 0.0
    c0: float = 0.0

    # --------------------------------------------------------
    # Canonical controller coefficients
    # --------------------------------------------------------
    # Positive score => stronger preference toward Gossip.
    #
    # latency ↑      => Gossip preference ↑
    # churn ↑        => Gossip preference ↑
    # duplicates ↑   => Gossip preference ↓
    # utilization ↑  => Gossip preference ↑
    #
    # These signs follow our RO2-derived behavioural rationale.
    w_d: float = -1.0
    w_l: float = 1.0
    w_u: float = 1.0
    w_c: float = 1.0

    # --------------------------------------------------------
    # Stage 2: sigmoid sensitivity
    # --------------------------------------------------------
    # Provisional value.
    kappa: float = 1.0

    # --------------------------------------------------------
    # Stage 3: fanout response sensitivity
    # --------------------------------------------------------
    # Provisional value.
    beta: float = 1.0

    # Canonical forwarding bounds
    min_fanout: int = 2
    max_fanout: int = 6

    # weight >= threshold => Gossip
    # weight < threshold  => Structured
    mode_threshold: float = 0.5


# ============================================================
# Canonical AHBN controller
# ============================================================

class AHBNController:

    def __init__(self, params: AHBNParams) -> None:
        self.params = params

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def ewma(self, old: float, new: float) -> float:
        """
        EWMA:
            x_hat(t) =
                alpha * x(t)
                + (1 - alpha) * x_hat(t-1)
        """
        alpha = self.params.alpha

        # Defensive clamp: canonical observations must be [0, 1].
        new = self.clamp(new, 0.0, 1.0)

        return alpha * new + (1.0 - alpha) * old

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Numerically stable logistic sigmoid.
        """
        if x >= 0.0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)

        z = math.exp(x)
        return z / (1.0 + z)

    # --------------------------------------------------------
    # Observation update
    # --------------------------------------------------------

    def update_metrics(
        self,
        state: NodeControlState,
        duplicate_obs: Optional[float] = None,
        latency_obs: Optional[float] = None,
        utilization_obs: Optional[float] = None,
        churn_obs: Optional[float] = None,
    ) -> None:

        if duplicate_obs is not None:
            state.d_hat = self.ewma(
                state.d_hat,
                duplicate_obs,
            )

        if latency_obs is not None:
            state.l_hat = self.ewma(
                state.l_hat,
                latency_obs,
            )

        if utilization_obs is not None:
            state.u_hat = self.ewma(
                state.u_hat,
                utilization_obs,
            )

        if churn_obs is not None:
            state.c_hat = self.ewma(
                state.c_hat,
                churn_obs,
            )


    # --------------------------------------------------------
    # Canonical AHBN score
    # --------------------------------------------------------

    def compute_score(
        self,
        state: NodeControlState,
    ) -> float:
        """
        Canonical centred AHBN controller score:

            S_t =
                w_d (d_hat - d0)
              + w_l (l_hat - l0)
              + w_u (u_hat - u0)
              + w_c (c_hat - c0)

        Positive score:
            preference shifts toward Gossip.

        Negative score:
            preference shifts toward Structured.
        """

        p = self.params

        return (
            p.w_d * (state.d_hat - p.d0)
            + p.w_l * (state.l_hat - p.l0)
            + p.w_u * (state.u_hat - p.u0)
            + p.w_c * (state.c_hat - p.c0)
        )

    # --------------------------------------------------------
    # Score -> Gossip preference
    # --------------------------------------------------------

    def compute_weight(
        self,
        state: NodeControlState,
    ) -> float:
        """
        Map the controller score to a bounded Gossip preference:

            W_t = sigmoid(kappa * S_t)

        W_t -> 1:
            stronger Gossip preference.

        W_t -> 0:
            stronger Structured preference.
        """

        score = self.compute_score(state)

        weight = self.sigmoid(
            self.params.kappa * score
        )

        return self.clamp(weight, 0.0, 1.0)

    # --------------------------------------------------------
    # Mode + fanout decision
    # --------------------------------------------------------

    def decide_mode_and_fanout(
        self,
        state: NodeControlState,
    ) -> None:
        """
        Produce the canonical AHBN dissemination decision.

        Step 1:
            observations -> score

        Step 2:
            score -> sigmoid weight

        Step 3:
            weight -> mode

        Step 4:
            score thresholds -> bounded fanout
        """

        p = self.params

        # -----------------------------------------
        # Controller score
        # -----------------------------------------
        state.score = self.compute_score(state)

        # -----------------------------------------
        # Gossip preference
        # -----------------------------------------
        state.weight = self.sigmoid(
            p.kappa * state.score
        )

        state.weight = self.clamp(
            state.weight,
            0.0,
            1.0,
        )

        # -----------------------------------------
        # Dissemination mode
        # -----------------------------------------
        if state.weight >= p.mode_threshold:
            state.mode = "gossip"
        else:
            state.mode = "cluster"

        # -----------------------------------------
        # Adaptive forwarding fanout
        # -----------------------------------------
        #
        if state.score <= -0.25:
            state.fanout = 2
        elif state.score < 0.25:
            state.fanout = 3
        elif state.score < 0.90:
            state.fanout = 4
        elif state.score < 1.50:
            state.fanout = 5
        else:
            state.fanout = 6

    # --------------------------------------------------------
    # Logging / experiment trace
    # --------------------------------------------------------

    def snapshot_state(
        self,
        state: NodeControlState,
    ) -> dict:
        """
        Return the complete canonical AHBN decision state.

        This provides traceability from observations to
        controller decision for experimental analysis.
        """

        return {
            "d_hat": state.d_hat,
            "l_hat": state.l_hat,
            "u_hat": state.u_hat,
            "c_hat": state.c_hat,
            "score": state.score,
            "weight": state.weight,
            "mode": state.mode,
            "fanout": state.fanout,
        }
