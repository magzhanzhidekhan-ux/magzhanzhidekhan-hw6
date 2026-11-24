# HW6 submission

## AI-use declaration

I used Claude to understand the value iteration method described in the lecture, as well as to set 
up the repository, the PyCharm environment, and the virtual environment. All manual calculations and 
their accompanying explanations for Part A were completed by me personally.

## Setup and reproduction

- Python 3.12, virtual environment `.venv`
- Versions: see [outputs/versions.txt](outputs/versions.txt)
- Install:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install gymnasium numpy pandas

## Part A — Paper calculations

Scans of my handwritten work (MDP description, V1 map, V2 calculations at
(1,2), (1,3), (2,3), best-action arrows, and why two rounds are not enough):

- [partA/img.png]
- [partA/img_1.png]
- [partA/img_2.png]

Summary: V2(1,2) = 8 (right), V2(1,3) = 10 (right), V2(2,3) = 5.21 (up).
No ties at these positions. Two rounds only look two steps ahead, so the
values have not converged and the policy can still change in later rounds.

## Part B — FrozenLake

### Fixed policy, written before the experiments

State the action for each nonterminal state and your reasoning.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |

Link episode rows and example traces in outputs/. Explain slipping, variation across batches and why these runs do not train a policy.

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

