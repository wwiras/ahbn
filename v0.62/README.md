# Adaptive Hybrid Broadcast Network (AHBN)

This repository contains the improved **canonical implementation of the Adaptive Hybrid Broadcast Network (AHBN)** used to investigate adaptive message dissemination under dynamic blockchain peer-to-peer (P2P) network conditions.

AHBN dynamically adjusts its dissemination behaviour using local observations of:

* message duplication,
* propagation latency,
* node utilization, and
* network churn.

The implementation follows a controlled **“no more, no less” experimental plan**. The objective is to define, validate, justify, freeze, and evaluate one clearly specified AHBN controller without introducing unnecessary mechanisms or experiment-specific post-hoc tuning.

---

## 1. Research Motivation from RO2

The design of AHBN is motivated by empirical findings obtained from the preceding RO2 experiments.

RO2 investigated how different dissemination mechanisms behave under changes in:

* forwarding fanout,
* cluster-head bottlenecks,
* network density,
* peer and cluster-head failures,
* network churn, and
* resource or delay asymmetry.

These findings provide the empirical motivation for introducing runtime adaptation in AHBN.

The overall research progression is:

```text id="j4mnnk"
RO2 Findings
     │
     │  Empirical motivation:
     │  fanout, bottleneck, density,
     │  failure, churn, asymmetry
     │
     ▼
Canonical AHBN
     │
     ▼
Validation and Parameter Justification
     │
     ▼
Freeze AHBN
     │
     ▼
Comparative Evaluation
     │
     ▼
Scientific Analysis and RO3 Synthesis
```

---

## 2. Canonical AHBN Controller

The canonical AHBN controller maintains exponentially weighted moving averages (EWMA) of four local observations:

$$
\hat{d}_t,\quad
\hat{\ell}_t,\quad
\hat{u}_t,\quad
\hat{\rho}_t
$$

where:

| Variable       | Meaning                       |
| -------------- | ----------------------------- |
| $\hat{d}_t$    | Duplicate-message observation |
| $\hat{\ell}_t$ | Latency observation           |
| $\hat{u}_t$    | Node utilization observation  |
| $\hat{\rho}_t$ | Churn observation             |

The EWMA update for an observation $x_t$ is:

$$
\hat{x}*t =
\alpha x_t +
(1-\alpha)\hat{x}*{t-1}
$$

where $\alpha$ controls how strongly the controller responds to the newest observation.

---

### 2.1 Adaptive Score

The controller computes an adaptive score:

$$
z_t =
w_d(\hat{d}_t-d_0)
+
w_l(\hat{\ell}_t-l_0)
+
w_u(\hat{u}_t-u_0)
+
w_c(\hat{\rho}_t-c_0)
$$

where:

* $d_0$ is the duplicate reference centre,
* $l_0$ is the latency reference centre,
* $u_0$ is the utilization reference centre,
* $c_0$ is the churn reference centre, and
* $w_d$, $w_l$, $w_u$, and $w_c$ determine the direction and influence of each observation.

The canonical weighting is:

$$
w_d=-1,\quad
w_l=+1,\quad
w_u=-1,\quad
w_c=+1
$$

Therefore:

* higher duplication reduces Gossip preference,
* higher utilization reduces Gossip preference,
* higher latency increases Gossip preference, and
* higher churn increases Gossip preference.

---

### 2.2 Sigmoid Preference

The adaptive score is converted into a bounded Gossip preference:

$$
p_t = \sigma(\kappa z_t)
$$

where:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

and $\kappa$ controls the steepness of the sigmoid response.

Therefore:

$$
0 < p_t < 1
$$

A higher value of $p_t$ represents stronger preference toward Gossip dissemination.

---

### 2.3 Mode Selection

The dissemination mode is determined using the canonical mode threshold:

$$
\text{mode}*t =
\begin{cases}
\text{Gossip}, & p_t \ge \tau*{\text{mode}} \
\text{Structured}, & p_t < \tau_{\text{mode}}
\end{cases}
$$

where:

$$
\tau_{\text{mode}} = 0.5
$$

---

### 2.4 Adaptive Fanout

The controller also maps its adaptive preference into a bounded forwarding fanout.

The fanout remains constrained by:

$$
f_{\min} \le f_t \le f_{\max}
$$

with the current canonical bounds:

$$
f_{\min}=2,\qquad f_{\max}=4
$$

The fanout mechanism allows AHBN to change dissemination intensity while preserving explicitly defined lower and upper bounds.

---

## 3. Canonical Parameter Configuration

The starting canonical configuration is:

```yaml id="tql4zn"
ahbn:
  alpha: 0.30

  d0: 0.50
  l0: 0.50
  u0: 0.50
  c0: 0.50

  w_d: -1.0
  w_l: 1.0
  w_u: -1.0
  w_c: 1.0

  kappa: 1.0
  beta: 1.0

  mode_threshold: 0.50

  min_fanout: 2
  max_fanout: 4
  default_fanout: 3
```

These values form the reference configuration used during controller validation and parameter-sensitivity analysis.

They are not intended to be continuously tuned during the final comparative experiments.

---

# 4. Experimental Development Plan

The improved AHBN follows a deliberately controlled sequence of stages.

The guiding principle is:

> **Define only what is required, validate it, justify the parameters, freeze the controller, and then evaluate it without experiment-specific retuning.**

The complete workflow is:

```text id="wcc45z"
RO2 FINDINGS
     │
     ▼
STAGE 0 — Canonical AHBN Implementation
     │
     ▼
STAGE 1 — Canonical Sanity Validation
     │
     ▼
STAGE 2 — Parameter Sensitivity
     │
     ▼
════════════════════════════════
       FREEZE CANONICAL AHBN
════════════════════════════════
     │
     ▼
STAGE 3 — External Hybrid Baseline
     │
     ▼
       BASELINES FROZEN
     │
     ▼
STAGE 4 — Final Comparative Evaluation
     │
     ▼
STAGE 5 — Statistical Validation
     │
     ▼
STAGE 6 — Scientific Analysis
     │
     ▼
STAGE 7 — RO3 Synthesis
     │
     ▼
STAGE 8 — Manuscript + Chapter Revision
     │
     ▼
STAGE 9 — Reproducibility Package
     │
     ▼
STAGE 10 — Final Response Package
```

---

## 5. Stage 0 — Canonical AHBN Implementation

**Purpose:** Define one authoritative AHBN implementation before performance evaluation.

Stage 0 establishes:

* the canonical controller equation,
* EWMA observations,
* adaptive score,
* sigmoid mapping,
* mode-selection rule,
* adaptive fanout rule,
* controller state variables,
* YAML configuration,
* adaptive trace structure, and
* consistent controller logic across simulation and Kubernetes environments.

The objective is to avoid having different variants of AHBN operating in different experiments.

### Stage 0 output

```text id="as2hqs"
Canonical controller
        │
        ▼
Canonical parameters
        │
        ▼
Canonical YAML configuration
        │
        ▼
Same AHBN logic across environments
```

**Status:** Completed.

---

## 6. Stage 1 — Canonical Sanity Validation

**Purpose:** Verify that the implementation behaves exactly as defined by the canonical mathematical formulation.

The validation verifies:

* $\hat{d}_t$,
* $\hat{\ell}_t$,
* $\hat{u}_t$,
* $\hat{\rho}_t$,
* controller score,
* sigmoid probability,
* Gossip versus Structured decisions,
* fanout movement,
* `mode_switched`,
* `fanout_changed`, and
* expected controller behaviour under Exp07–Exp12 conditions.

The internal controller path is:

```text id="ksk7o7"
Raw observations
       │
       ▼
EWMA observations
       │
       ▼
Adaptive score
       │
       ▼
Sigmoid preference
       │
       ▼
Mode decision
       │
       ▼
Adaptive fanout
       │
       ▼
Adaptive trace
```

Automatic checks verify that:

* observations remain bounded,
* EWMA states remain bounded,
* the score equation is reproduced correctly,
* sigmoid calculations are correct,
* the selected mode matches the controller probability,
* fanout calculations are correct, and
* fanout remains within its configured bounds.

### Stage 1 result

```text id="b512qt"
CANONICAL SANITY VALIDATION
            │
            ▼
          PASS
```

**Status:** Completed — sanity and equation checks PASS.

---

## 7. Stage 2 — Parameter Sensitivity

**Purpose:** Demonstrate how the main AHBN parameters influence controller behaviour and justify the final selected configuration.

The objective is **not** to search exhaustively for the parameter combination that produces the best performance.

Instead, Stage 2 asks:

> **Does AHBN behave predictably across a reasonable parameter region, and can the selected canonical values be justified?**

Stage 2 is divided into four controlled sensitivity analyses.

---

### 7.1 Stage 2A — Threshold Centres

The operating centres are:

$$
d_0,\quad l_0,\quad u_0,\quad c_0
$$

These determine the reference operating points around which the controller changes its preference.

The analysis examines whether reasonable changes to these centres produce understandable shifts in AHBN behaviour.

---

### 7.2 Stage 2B — EWMA Sensitivity

The EWMA smoothing parameter is:

$$
\alpha
$$

For:

$$
0 < \alpha \le 1
$$

a larger $\alpha$ gives more weight to the newest observation, while a smaller $\alpha$ produces stronger smoothing.

The purpose is to study AHBN responsiveness without selecting $\alpha$ purely from final performance.

---

### 7.3 Stage 2C — Sigmoid Sensitivity

The sigmoid steepness parameter is:

$$
\kappa
$$

used in:

$$
p_t=\sigma(\kappa z_t)
$$

A larger $\kappa$ creates a sharper transition between Structured and Gossip preference, whereas a smaller $\kappa$ produces a more gradual transition.

---

### 7.4 Stage 2D — Fanout Sensitivity

The fanout-control parameter is:

$$
\beta
$$

It determines how strongly the controller preference affects adaptive fanout movement within the permitted range.

The objective is to verify that the fanout response remains bounded and interpretable.

---

### Stage 2 purpose

Together, Stage 2A–2D are intended to:

* demonstrate parameter influence,
* justify the selected canonical values,
* identify unstable or overly sensitive settings,
* demonstrate that conclusions are not dependent on one arbitrary configuration, and
* prevent post-hoc parameter selection during final evaluation.

The sequence is:

```text id="4theym"
2A — Centres
      │
      ▼
2B — alpha
      │
      ▼
2C — kappa
      │
      ▼
2D — beta
      │
      ▼
FREEZE AHBN
```

**Status:** In progress.

---

# 8. Freeze Canonical AHBN

After Stage 2 is completed, the final canonical AHBN configuration is frozen.

```text id="kgp3zz"
Parameter sensitivity completed
              │
              ▼
Final parameter justification
              │
              ▼
════════════════════════════
       AHBN FROZEN
════════════════════════════
              │
              ▼
   No experiment-specific
        retuning
```

After this point:

* AHBN parameters are not changed to improve individual experimental results,
* the same controller is used across all final evaluation scenarios, and
* observed weaknesses are reported rather than tuned away.

This separates:

> **Controller design and justification**

from:

> **Controller performance evaluation**

---

## 9. Stage 3 — External Hybrid Baseline

**Purpose:** Introduce one literature-derived hybrid or mixed dissemination comparator.

The planned external comparator is currently a **DC-SoC-inspired hybrid baseline**.

Stage 3 includes:

* finalizing one external hybrid design,
* implementing the comparator,
* validating its implementation,
* defining fixed baseline parameters, and
* ensuring the comparator is evaluated fairly.

The purpose is not to reproduce every existing hybrid dissemination system.

Instead, one defensible literature-derived mixed/hybrid comparator is included to determine whether AHBN's runtime adaptation provides value beyond architectural hybridization.

### Baselines after Stage 3

```text id="fo37kl"
BASELINES FROZEN

├─ Gossip
├─ Structured
├─ External Hybrid
└─ AHBN
```

Once baseline configurations are frozen, they remain fixed during final evaluation.

---

## 10. Stage 4 — Final Comparative Evaluation

**Purpose:** Compare all frozen dissemination strategies under equivalent experimental conditions.

The comparison is:

```text id="zp7f9j"
Gossip
   vs
Structured
   vs
External Hybrid
   vs
AHBN
```

The same canonical AHBN is used throughout the evaluation.

No parameter tuning is performed to favour AHBN in a particular scenario.

---

### 10.1 Simulation Experiments

| Experiment | Primary Variable  | Evaluation Objective                      |
| ---------- | ----------------- | ----------------------------------------- |
| Exp07      | Forwarding Fanout | Latency–duplication trade-off boundaries  |
| Exp08      | CH Overload       | Abstract cluster-head bottleneck analysis |
| Exp09      | Network Density   | Duplicate amplification in dense overlays |

Simulation experiments provide controlled analysis over configurable network conditions.

---

### 10.2 Kubernetes Experiments

| Experiment | Primary Variable  | Evaluation Objective           |
| ---------- | ----------------- | ------------------------------ |
| Exp08(K8s) | CH Overload       | Runtime bottleneck adaptation  |
| Exp10      | Peer/CH Failures  | Robustness and recovery        |
| Exp11      | Pod Churn         | Join–leave adaptation          |
| Exp12      | Asymmetric Delays | Resource/structural adaptation |

Kubernetes experiments provide cloud-native runtime validation under actual distributed execution conditions.

---

### Stage 4 principles

All strategies should be evaluated using:

* the same canonical AHBN,
* equivalent experimental conditions,
* documented configurations,
* comparable workload definitions, and
* no parameter tuning intended to favour AHBN.

---

## 11. Stage 5 — Statistical Validation

**Purpose:** Ensure the reported differences are supported by repeated observations rather than individual runs.

The final experiments should include:

* repeated independent runs,
* documented random seeds,
* documented topology realizations,
* arithmetic mean,
* standard deviation and/or 95% confidence intervals, and
* appropriate statistical comparisons where applicable.

The analysis should distinguish between:

```text id="bz0ew1"
Observed difference
        │
        ▼
Run-to-run variation
        │
        ▼
Statistical uncertainty
        │
        ▼
Supported interpretation
```

The aim is not to use statistical testing mechanically, but to quantify uncertainty around the reported experimental findings.

---

## 12. Stage 6 — Scientific Analysis

**Purpose:** Interpret the final experimental evidence in relation to the original research questions.

The primary evaluation dimensions include:

* propagation latency,
* duplication and dissemination overhead,
* delivery reliability,
* recovery behaviour where applicable, and
* AHBN adaptive behaviour.

Each experiment should answer its original scientific question rather than merely reporting which method produced the lowest numerical value.

For AHBN, adaptive traces can additionally explain:

* why the controller selected Gossip or Structured dissemination,
* when fanout changed,
* when mode changes occurred, and
* whether those changes corresponded to changing network conditions.

---

## 13. Stage 7 — RO3 Synthesis

**Purpose:** Combine the experimental results into the overall RO3 scientific argument.

The synthesis should answer the following progression:

```text id="uz7v70"
What did RO2 reveal?
        │
        ▼
Why did those findings motivate AHBN?
        │
        ▼
What is the difference between
fixed/static dissemination
and runtime adaptation?
        │
        ▼
What does AHBN improve?
        │
        ▼
Where does AHBN NOT improve?
        │
        ▼
Does runtime adaptation add value
beyond Gossip, Structured,
and architectural hybridization?
```

The intention is not to claim that AHBN dominates every alternative under every condition.

Instead, the scientific contribution is determined by identifying:

* where adaptation provides measurable value,
* which trade-offs remain,
* which conditions favour simpler approaches, and
* whether runtime adaptation complements or improves upon fixed architectural strategies.

---

## 14. Stage 8 — Manuscript and Chapter 5 Revision

**Purpose:** Update the written research claims using the evidence generated by the frozen experimental design.

The revision should include:

* rewriting claims using the new experimental evidence,
* strengthening RO3 Claim 1,
* strengthening the RO2-to-RO3 linkage,
* clarifying static versus dynamic dissemination,
* explaining AHBN design challenges,
* discussing delivery-ratio implications,
* reporting scalability limitations,
* reporting security limitations, and
* documenting other limitations and future work.

The written claims should follow the evidence produced by the final experiments rather than the experiments being changed to support pre-existing claims.

---

## 15. Stage 9 — Reproducibility Package

**Purpose:** Provide sufficient material for the final experimental workflow to be inspected and reproduced.

The package should include:

* canonical source code,
* YAML configurations,
* experiment scripts,
* seeds and run configurations,
* raw experimental results,
* adaptive traces,
* result-processing scripts,
* statistical-analysis scripts,
* figures,
* README documentation,
* environment information,
* GitHub repository, and
* archived release with a Zenodo DOI.

A frozen release should correspond to the version of the code used to produce the reported results.

---

## 16. Stage 10 — Final Response Package

The final research response package should contain:

* revised Scientific Reports manuscript,
* response to the Editor,
* responses to Reviewers 1–5, and
* authoritative Chapter 5 RO3 results.

The purpose of this final stage is to ensure that the manuscript, reviewer responses, thesis chapter, source code, and archived experimental evidence all refer to the same canonical implementation and final results.

---

# 17. Adaptive Trace

AHBN records controller decisions through an adaptive trace.

The trace provides visibility into the complete adaptive path:

```text id="6gyj1n"
Raw observation
      │
      ▼
EWMA state
      │
      ▼
Controller score
      │
      ▼
Sigmoid preference
      │
      ▼
Selected mode
      │
      ▼
Selected fanout
```

Typical trace fields include:

```text id="quh7xs"
event_type

duplicate_obs
latency_obs
utilization_obs
churn_obs

d_hat
l_hat
u_hat
rho_hat

score
weight

mode
fanout

mode_switched
fanout_changed
```

The adaptive trace is used primarily for:

* controller validation,
* interpretation of adaptive behaviour,
* debugging canonical implementation consistency, and
* explaining why AHBN changed its dissemination behaviour.

It is not treated as an additional performance metric.

---

# 18. “No More, No Less” Principle

The canonical AHBN evaluation deliberately limits changes to those required to answer the defined research questions.

### We do **not**:

* continuously redesign AHBN after observing experimental outcomes,
* perform experiment-specific controller tuning,
* exhaustively search for the parameter combination giving the best result,
* add additional mechanisms simply to make AHBN outperform a comparator,
* change baseline configurations during final evaluation, or
* introduce additional research questions that are unnecessary for the current RO3 validation.

### We do:

* use RO2 evidence to motivate AHBN,
* define one canonical controller,
* validate the implementation,
* investigate necessary parameter sensitivity,
* justify the final parameter configuration,
* freeze AHBN before comparative evaluation,
* freeze the baseline configurations,
* perform comparable experiments,
* quantify uncertainty,
* report advantages and disadvantages, and
* preserve limitations where they genuinely exist.

The simplified principle is:

```text id="8y7bp3"
RO2 evidence
     ↓
Define
     ↓
Validate
     ↓
Justify
     ↓
Freeze
     ↓
Compare
     ↓
Analyse
     ↓
Report
```

The key rule is:

> **After the canonical AHBN is frozen, experimental results may change the scientific interpretation, but they should not trigger experiment-specific retuning of the controller.**

---

# 19. Current Progress

* [x] RO2 findings established
* [x] Stage 0 — Canonical AHBN implementation
* [x] Canonical YAML files prepared
* [x] Stage 1 — Canonical sanity validation
* [ ] Stage 2A — Threshold-centre sensitivity
* [ ] Stage 2B — $\alpha$ sensitivity
* [ ] Stage 2C — $\kappa$ sensitivity
* [ ] Stage 2D — $\beta$ sensitivity
* [ ] Freeze canonical AHBN
* [ ] Stage 3 — External hybrid baseline
* [ ] Freeze all baseline configurations
* [ ] Stage 4 — Final comparative evaluation
* [ ] Stage 5 — Statistical validation
* [ ] Stage 6 — Scientific analysis
* [ ] Stage 7 — RO3 synthesis
* [ ] Stage 8 — Manuscript and Chapter 5 revision
* [ ] Stage 9 — Reproducibility package
* [ ] Stage 10 — Final response package

This section should be updated as each stage is completed.

---

# 20. Research Scope

AHBN is intended to investigate whether **runtime adaptation based on decentralized local observations** can improve or complement fixed dissemination strategies under changing blockchain P2P network conditions.

The central question is:

> **Does runtime adaptation add value beyond Gossip, Structured dissemination, and architectural hybridization under dynamic network conditions?**

The implementation is intentionally restricted to the mechanisms required to answer this question.

AHBN is therefore not intended to represent a complete production blockchain networking stack, nor is the current study intended to optimize every possible dissemination parameter or network condition.
