# Adaptive Hybrid Broadcast Network (AHBN)

This repository contains the improved **canonical implementation of the Adaptive Hybrid Broadcast Network (AHBN)** used to investigate adaptive message dissemination under dynamic blockchain peer-to-peer (P2P) network conditions.

AHBN dynamically adjusts its dissemination behaviour using local observations of:

* message duplication,
* propagation latency,
* node utilization, and
* network churn.

The implementation follows a controlled **“no more, no less” experimental plan**. The objective is to validate and evaluate one clearly defined AHBN controller without introducing unnecessary mechanisms or post-hoc optimization.

---

## 1. Canonical AHBN

The canonical AHBN controller maintains exponentially weighted moving averages (EWMA) of four local observations:

$$
\hat{d}_t,\quad
\hat{\ell}_t,\quad
\hat{u}_t,\quad
\hat{\rho}_t
$$

representing:

| Variable       | Meaning                       |
| -------------- | ----------------------------- |
| $\hat{d}_t$    | Duplicate-message observation |
| $\hat{\ell}_t$ | Latency observation           |
| $\hat{u}_t$    | Node utilization observation  |
| $\hat{\rho}_t$ | Churn observation             |

The controller computes the score:

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

and converts it into a Gossip preference:

$$
p_t = \sigma(\kappa z_t)
$$

where $\sigma(\cdot)$ is the sigmoid function:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The controller then determines the dissemination mode and forwarding fanout.

Higher latency and churn increase the preference toward **Gossip**, while higher duplication and utilization increase the preference toward **Structured dissemination**.

---

## 2. Canonical Parameter Configuration

The starting canonical configuration is:

```yaml
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

These parameters form the reference configuration used during controller validation and parameter-sensitivity analysis.

---

## 3. AHBN Validation and Experimental Stages

The canonical AHBN is developed and evaluated using a deliberately limited sequence of stages.

The principle is:

> **Validate what is necessary to justify the controller, freeze it, and then evaluate it without further tuning.**

The workflow is:

```text
Canonical AHBN
      │
      ▼
Stage 0 — Canonical Configuration
      │
      ▼
Stage 1 — Canonical Sanity Validation
      │
      ▼
Stage 2 — Parameter Sensitivity
      │
      ▼
FREEZE AHBN
      │
      ▼
Comparative Evaluation
      │
      ▼
Statistical Analysis + Kubernetes Validation
```

---

### Stage 0 — Canonical Configuration

**Purpose:** Define one consistent AHBN implementation and parameter configuration.

This stage aligns:

* controller equations,
* YAML configuration,
* forwarding behaviour,
* adaptive fanout,
* dissemination mode selection, and
* adaptive trace logging.

The same canonical controller should subsequently be used throughout the experiments.

**Expected outcome:** One clearly defined AHBN implementation suitable for validation.

**Status:** Completed.

---

### Stage 1 — Canonical Sanity Validation

**Purpose:** Verify that the implementation behaves exactly as defined by the AHBN equations.

Synthetic controller cases are used to test conditions such as:

* neutral observations,
* high duplication,
* high utilization,
* high latency,
* high churn,
* strong Gossip preference, and
* strong Structured preference.

The validation checks:

```text
observations
     ↓
EWMA
     ↓
controller score
     ↓
sigmoid weight
     ↓
mode decision
     ↓
adaptive fanout
```

Automatic checks verify:

* score equation,
* sigmoid calculation,
* mode decision,
* fanout equation,
* bounded observations,
* bounded controller weight, and
* bounded fanout.

**Expected outcome:** The code reproduces the mathematical definition of AHBN.

**Status:** Completed — sanity and equation checks PASS.

---

### Stage 2 — Parameter Sensitivity

**Purpose:** Determine whether reasonable changes to controller parameters preserve sensible AHBN behaviour.

The objective is **not** to search for the parameter combination that gives the best experimental result.

Instead, the question is:

> **Does AHBN continue to behave sensibly across a reasonable parameter region?**

Sensitivity analysis examines the main controller parameters, including:

#### Operating Centres

$$
d_0,\quad l_0,\quad u_0,\quad c_0
$$

These determine the reference operating conditions around which the controller reacts.

#### EWMA Smoothing

$$
\alpha
$$

This controls how quickly AHBN responds to new observations versus historical observations.

#### Sigmoid Sensitivity

$$
\kappa
$$

This controls how strongly changes in the controller score influence the Gossip preference.

#### Fanout Sensitivity

$$
\beta
$$

This controls how strongly the controller weight influences adaptive fanout.

The intended sequence is:

```text
Centres
   ↓
alpha
   ↓
kappa
   ↓
beta
   ↓
FREEZE
```

**Expected outcome:** Reasonable parameter changes produce predictable trade-offs rather than unstable or arbitrary controller behaviour.

---

## 4. Controller Freeze

After parameter sensitivity is completed, the canonical AHBN configuration is **frozen**.

```text
Parameter Sensitivity Completed
              │
              ▼
     Final Parameters Selected
              │
              ▼
          AHBN FROZEN
              │
              ▼
       No Further Tuning
```

After this point, parameters should not be changed simply because another value produces a better experimental result.

This separation is important because it distinguishes:

> **Controller justification**

from

> **Controller performance evaluation**

Once frozen, the same canonical AHBN controller is used throughout the subsequent comparative experiments.

---

## 5. Comparative Evaluation

After the controller is frozen, AHBN is evaluated against dissemination baselines under the same experimental conditions.

The intended comparison is:

```text
Gossip
   vs
Structured
   vs
Fixed/External Hybrid Baseline
   vs
AHBN
```

The main question is:

> **Can runtime adaptation outperform or complement fixed dissemination strategies and architectural hybridization under dynamic network conditions?**

---

## 6. Experimental Scenarios

The experimental suite examines different sources of dissemination stress.

| Experiment | Environment | Primary Variable            | Evaluation Objective                      |
| ---------- | ----------- | --------------------------- | ----------------------------------------- |
| Exp07      | Simulation  | Forwarding Fanout           | Latency–duplication trade-off boundaries  |
| Exp08      | Simulation  | CH Overload                 | Cluster-head bottleneck behaviour         |
| Exp09      | Simulation  | Network Density             | Duplicate amplification in dense overlays |
| Exp08(K8s) | Kubernetes  | CH Overload                 | Runtime bottleneck adaptation             |
| Exp10      | Kubernetes  | Peer/CH Failures            | Robustness and recovery                   |
| Exp11      | Kubernetes  | Pod Churn                   | Adaptation to join–leave dynamics         |
| Exp12      | Kubernetes  | Asymmetric Resources/Delays | Resource-aware adaptation                 |

Simulation experiments are used for controlled analysis, while Kubernetes experiments provide cloud-native runtime validation.

---

## 7. Adaptive Trace

AHBN records controller decisions through an adaptive trace.

The trace provides visibility into:

```text
raw observation
      │
      ▼
EWMA state
      │
      ▼
controller score
      │
      ▼
sigmoid weight
      │
      ▼
selected mode
      │
      ▼
selected fanout
```

Typical trace fields include:

```text
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

The adaptive trace is primarily intended for controller validation and interpretation rather than as an additional performance metric.

---

## 8. “No More, No Less” Principle

The canonical AHBN evaluation deliberately avoids unnecessary extensions during the validation process.

### We do **not**:

* continuously redesign the controller after observing results,
* search for an artificially optimal parameter combination,
* add new mechanisms simply to improve individual experiments, or
* tune AHBN separately for each evaluation scenario.

### We do:

* verify the equations,
* test reasonable parameter sensitivity,
* justify the final configuration,
* freeze the controller,
* compare strategies under equivalent conditions, and
* report both advantages and limitations.

This keeps the evaluation focused on the research question rather than post-hoc optimization.

In short:

```text
Define
  ↓
Validate
  ↓
Test Sensitivity
  ↓
Freeze
  ↓
Evaluate
  ↓
Report
```

---

## 9. Current Progress

* [x] Stage 0 — Canonical AHBN configuration
* [x] Stage 1 — Canonical sanity validation
* [ ] Stage 2 — Parameter sensitivity
* [ ] Freeze canonical AHBN
* [ ] Implement/validate external hybrid baseline
* [ ] Comparative simulation evaluation
* [ ] Statistical analysis and 95% confidence intervals
* [ ] Kubernetes validation
* [ ] Final reproducibility package

This section should be updated as each stage is completed.

---

## 10. Reproducibility

The final reproducibility package should contain:

```text
source code
configuration files
experiment scripts
random seeds
raw experimental results
adaptive traces
analysis scripts
figures
environment information
```

The final archived release can be deposited in a persistent research repository and referenced using its DOI.

---

## 11. Research Scope

AHBN is intended to investigate whether **runtime adaptation based on decentralized local observations** can improve the latency–duplication trade-off of blockchain P2P message dissemination under changing network conditions.

The implementation is therefore intentionally focused on this research objective rather than attempting to provide a complete production blockchain networking stack.
