# Problem Definition — DRL for CBM of Multi-Component Systems (Zhang & Si, 2020)

Nov 17, 2026 · @han

## Overview

The paper asks one question: at each periodic inspection of a K-component system, which components should be replaced so that the long-run average cost rate is minimized?

Instead of optimizing per-component maintenance thresholds, it searches directly for a decision function g that maps the inspected condition (o\_j, d\_j) to a replacement vector a\_j ∈ {0,1}^K. The system combines stochastic dependency (shared degradation), economic dependency (shared downtime) and competing failure risks. The problem is cast as a Markov decision process (MDP) and solved with a Double Deep Q-Network (DDQN).

Paper: Zhang, N. & Si, W. (2020). Deep reinforcement learning for condition-based maintenance planning of multi-component systems under dependent competing risks. *Reliability Engineering and System Safety*, 203, 107094. [doi:10.1016/j.ress.2020.107094](https://doi.org/10.1016/j.ress.2020.107094)

## Motivation: limits of prior work

Threshold-based CBM stops scaling once many components interact, and that is the gap this paper targets.

- **Threshold search explodes with K.** Classic CBM sets maintenance thresholds per component and tunes them, usually by grid search. This works for one or two components but becomes impractical as K grows.
- **Components depend on each other.**
  - Stochastic dependency: component degradations are correlated.
  - Economic dependency: maintaining parts together saves setup cost, and a system stop makes the downtime cost shared by all parts.
  - Structural dependency: parts physically bound together must be maintained together. Not considered in this paper.
- **Competing risks are often ignored.** A real system stops as soon as any one part fails, yet few models combine this with the dependencies above.
- **Earlier DRL work is limited.** Wei et al. dropped thresholds too, but discretized degradation into a few states and ignored component dependencies.

## System model (the environment)

Each component degrades through its own process plus one process shared by all components, and the system fails as soon as any component crosses its failure threshold.

```latex
d_i(t) = X_i(t) + X_s(t), \qquad i = 1, \dots, K
```

- **X\_i(t)** is the component-specific degradation (process S\_i).
- **X\_s(t)** is a system-level degradation shared by every component (process S\_s). This shared term creates the stochastic dependency. The paper's example: each tire of a car wears on its own, but a speed bump hits all four at once.
- Both processes have stationary, independent, non-negative increments. Two versions are studied:
  - Compound Poisson process (CPP): shocks arrive by a homogeneous Poisson process HPP(λ), and shock sizes follow a normal distribution truncated at zero.
  - Gamma process (GP): the increment over \[t1, t2\] follows Gamma(α(t2 − t1), β).
- **Failure:** component i fails when d\_i(t) ≥ δ\_i. The system fails when any component fails (a series structure, i.e. competing risks). The first failure stops the system, so later component failures are never observed (Fig. 1).
- **Survival:** the probability of surviving to t2, given the degradations at t1, follows from the joint distribution of the increments (Eq. 4).

```latex
S\big(t_2 \mid d(t_1)\big) = \Pr\big(\Delta_1(t_1,t_2) < \delta_1 - d_1(t_1),\ \dots,\ \Delta_K(t_1,t_2) < \delta_K - d_K(t_1)\big)
```

## Decision structure

At every inspection the decision maker sees the condition of all components and picks a replacement vector, aiming for the lowest long-run average cost rate.

- **Inspections** happen periodically at τ, 2τ, 3τ, …, and nothing is known between them. τ is given; when it is optimized (Scenario C), it is chosen by grid search outside the DRL.
- **Observation** at inspection j: system status o\_j (1 = no failure, 0 = failure observed) and degradation vector d\_j = \[d\_1j, …, d\_Kj\].
- **Action:** a\_j = \[a\_1j, …, a\_Kj\] with a\_ij ∈ {0, 1}. 1 = replace component i with a new one (perfect, negligible time); 0 = do nothing. There are 2^K possible actions.
- **Costs:** inspection cost c\_I per inspection, replacement cost c\_R,k per component, and downtime cost c\_D per unit time while the system is down.
- **Objective:** find the decision function g that minimizes the long-run average cost rate over horizon H. E(C\_H) includes inspection, replacement and downtime costs.

```latex
\min_{g} \; \frac{E(C_H)}{H}, \qquad a_j = g(o_j, d_j)
```

&#91;embedded content: one inspection cycle · 5 steps, 1 decision\]

Only the shaded step is learned; every other step is the environment, simulated from the CPP or GP model.

## MDP reformulation

Because increments are independent, today's degradation fully determines the future, so the inspect–replace loop is an MDP (S, A, P, γ, R).

- **State:** s\_j = \[o\_1j, …, o\_Kj, d\_1j, …, d\_Kj\]. Degradation stays continuous; nothing is discretized.
- **Action:** a\_j ∈ {0,1}^K.
- **Transition P:** replaced components reset to zero, d\_ij ← d\_ij(1 − a\_ij); then every component degrades stochastically for τ.
- **Reward:** the negative of the downtime and replacement costs incurred between inspections j and j+1 (Eq. 6). Here ε\_j is the downtime in that interval: a failure between inspections stays undetected until the next one.

```latex
R(s_j, a_j) = -\Big(\varepsilon_j(s_j, a_j)\, c_D + \sum_{k=1}^{K} a_{k,j}\, c_{R,k}\Big)
```

- **Objective:** maximize the discounted return (Eq. 5), with γ = 0.99 in the experiments.

```latex
R_j = \sum_{l=0}^{\infty} \gamma^{l} \, R(s_{j+l}, a_{j+l})
```

- **Solution:** the optimal policy π\*(s) = argmax\_a Q\*(s, a) is the function g. The paper approximates Q\* with a DDQN whose network takes s\_j as input and outputs 2^K action values.

## Notation

The symbols below are the ones needed to read Sections 2–3 of the paper.

| Symbol | Meaning |
| --- | --- |
| K | Number of components |
| d\_i(t) | Cumulative degradation of component i at time t |
| X\_i(t), X\_s(t) | Component-specific and shared degradation |
| δ\_i | Failure threshold of component i |
| λ\_i, λ\_s | Shock arrival rates (CPP) |
| α, β | Shape and rate parameters (GP) |
| τ | Inspection interval |
| o\_j | Status observed at inspection j (1 = running, 0 = failed) |
| d\_j | Degradation vector observed at inspection j |
| a\_j ∈ {0,1}^K | Replacement decision at inspection j |
| c\_I, c\_D, c\_R,k | Inspection cost, downtime cost per unit time, replacement cost |
| ε\_j | Downtime between inspections j and j+1 |
| H | Evaluation horizon (1000 in the experiments) |
| γ | Discount factor (0.99) |
| g, π | Decision function / policy mapping state to action |

## Points to check when reading

The formulation is clean, but a few gaps between the stated problem and what is trained are worth noticing.

- **Objective mismatch.** The problem is stated as minimizing E(C\_H)/H, but the agent maximizes a discounted return with γ = 0.99. Results are reported back as (TC\_I + TC\_D + TC\_R)/H.
- **No inspection cost in the reward (Eq. 6).** With τ fixed it is a constant that cannot change the policy, which is why τ is tuned separately by grid search.
- **o is defined two ways.** Section 2.3 uses a single system status o\_j; the state vector in Section 3.1 lists component-wise o\_1j … o\_Kj.
- **Economic dependency enters mainly through shared downtime.** Eq. (6) simply sums replacement costs. A joint-replacement discount appears only in experiment 4.2.2 (USD 60 for both parts vs USD 20 + 50).
- **Maintenance is perfect.** A replaced part is as good as new; the paper leaves imperfect maintenance to future work.
