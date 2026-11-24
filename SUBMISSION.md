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

Committed in [part_b/policy.py](part_b/policy.py) before the first run.
Actions: 0=left, 1=down, 2=right, 3=up.

| State | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action | down | up | up | up | down | left | right | down | left | right | right |

Reasoning: without slipping the route is 0 → 4 → 8 → 9 → 13 → 14 → 15, which
avoids all holes. With slipping, Gymnasium moves in the chosen direction with
probability 1/3 and in each perpendicular direction with 1/3. I checked the
three outcomes of each action. States 9, 13 and 14 have no hole among their
outcomes. Actions in the top row keep the agent on that row. The two risky
states are 4 (down can slip right into hole 5) and 8 (right can slip down into
hole 12), each with a 1/3 hole risk per attempt; I accepted this because the
policy must also work without slipping.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | fixed | 11 | 50 | 0 | 0 | 100 | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100 | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100 | 6.00 |
| False | random | 11 | 0 | 50 | 0 | 0 | 7.98 |
| False | random | 22 | 0 | 50 | 0 | 0 | 6.42 |
| False | random | 33 | 1 | 49 | 0 | 2 | 6.66 |
| True | fixed | 11 | 3 | 45 | 2 | 6 | 16.86 |
| True | fixed | 22 | 4 | 45 | 1 | 8 | 22.78 |
| True | fixed | 33 | 1 | 49 | 0 | 2 | 12.72 |
| True | random | 11 | 0 | 50 | 0 | 0 | 8.30 |
| True | random | 22 | 2 | 48 | 0 | 4 | 7.76 |
| True | random | 33 | 0 | 50 | 0 | 0 | 6.74 |

Files: [episode rows](outputs/episodes_b.csv),
[batch summary](outputs/summary_b.csv),
[example traces](outputs/traces_b.txt),
[one rendered episode](outputs/episode_render.txt).

**What slipping changed.** 
Without slipping the fixed policy reached the goal
in 150/150 episodes, always in 6 steps. With slipping it reached the goal in
only 8/150 (5.3%), ended in a hole in 139 and hit the time limit in 3, and the
episodes got longer (13-23 steps on average) because the agent drifts along
the top row. The random policy was poor in both settings (1/150 and 2/150
goals), and that difference is within noise.

**Consistency across batches.** 
Without slipping, the fixed batches are
identical because nothing is random. With slipping, the fixed policy gave 6%,
8% and 2%. At p ≈ 5% and n = 50 the standard error is about 3 percentage
points, so this spread is consistent with sampling noise. Fixed (5.3% over 150
episodes) beats random (1.3%) by about two standard errors: suggestive, not
conclusive.

**Why one successful episode is not enough.** 
With slipping, even the random
policy reached the goal 2 times in 150 runs, so one success cannot separate a
good policy from a lucky one. Success rate has to be estimated over many
episodes, with an error margin. (Without slipping the environment is
deterministic, so a single episode already shows everything.)

**Traces.** All traces are in [outputs/traces_b.txt](outputs/traces_b.txt).

- *Successful (slippery, fixed, episode seed 11020):* the goal was reached
  after 34 steps instead of the 6 needed without slipping. The agent was
  pushed sideways many times (for example it stayed in state 4 on the first
  "down" command and wandered between states 8, 9 and 10 for a long time),
  but it never slipped into a hole at the risky states 4 and 8, so it
  eventually reached the goal. This success is largely luck.
- *Failed, hole (slippery, fixed, episode seed 11000):* after drifting along
  the top row for about 35 steps, the agent went down to state 4 and on step
  38 the "down" command slipped it to the right into hole 5. This is the 1/3
  risk at state 4 that I accepted when designing the policy.
- *Time limit (slippery, fixed, episode seed 11011):* the agent stayed on the
  top row for all 100 steps. Under slipping, "up" in states 1-3 only keeps it
  there, so the policy can get stuck. This is a design weakness I only saw
  after running the experiments; I did not change the policy.

Outcomes never observed (copied from the last line of traces_b.txt):
`('slippery=False', 'random', 'time limit'), ('slippery=False', 'fixed', 'hole'),
('slippery=False', 'fixed', 'time limit'), ('slippery=True', 'random', 'time limit')`.
Without slipping, the fixed policy never hit a hole or the time limit, and the
random policy never hit the time limit (it always ended in a hole or the goal
first). With slipping, the random policy never hit the time limit either.

**Why this is not training.** The policy is a hand-written table that never
changes. The agent receives rewards but updates nothing from them; the runs
only measure how good a finished policy is.

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

