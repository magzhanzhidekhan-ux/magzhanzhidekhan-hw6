# HW6 submission

## AI-use declaration

## AI-use declaration

I used Claude throughout this homework. Parts A and B: it explained the lecture's value iteration method, 
helped set up the repository, PyCharm and the virtual environment, and cross-checked my Part A numbers; 
I did the Part A paper calculations myself. Claude also helped me reason about the fixed FrozenLake policy 
and wrote the first versions of the Part B code. Part C: Claude wrote the first versions of `floors.py`, 
`solver.py`, `simulate.py` and `checks.py`. I ran all programs on my machine, compared the solver output 
with my Part A paper results (they matched), and read the code to understand it. I also got Claude's help 
with debugging and with the statistics for interpreting the batch results. The explanations in this file 
are written by me, and all numbers come from my own runs. No other AI tools were used.

## Setup and reproduction

- Python 3.12, virtual environment `.venv`
- Versions: see [outputs/versions.txt](outputs/versions.txt)
- Install:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install gymnasium numpy pandas
python part_b/run_b.py
python part_c/solver.py 
python part_c/simulate.py
python part_c/checks.py

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

Code: [floors.py](part_c/floors.py) (floor rules), [solver.py](part_c/solver.py)
(transitions, value iteration, policy extraction), [simulate.py](part_c/simulate.py),
[checks.py](part_c/checks.py). Floor rules are stored in a dictionary in
`floors.py`, so a new floor is one new line there and `solver.py` is unchanged.

### Compare code with paper

The solver's V1 (all nonterminal tiles -1 except (1,3) = 10) equals my paper
map. V2 from the code: (1,2) = 8, (1,3) = 10, (2,3) = 5.21, equal to my hand
calculations. No discrepancies, so nothing had to be corrected.

### Values, policies and convergence

Value maps: [outputs/value_maps.txt](outputs/value_maps.txt). Policy maps:
[outputs/policy_maps.txt](outputs/policy_maps.txt).

| Map | Iterations | Final max change | Converged |
| --- | --- | --- | --- |
| Part A map | 12 | 2.165e-07 | yes |
| Extended map | 11 | 2.406e-07 | yes |

The stopping threshold was 1e-6 and the 10,000-iteration safeguard was never
reached. Tie-breaking: among actions whose Q-value is within 1e-9 of the best
one, the first in the order up, right, down, left is chosen. No ties occurred
in either map.

Final values and policies (rows 1-3, columns 1-4):

| Part A map V | | | | Policy | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4.200 | 8.000 | 10.000 | G | → | → | → | G |
| 2.780 | # | 5.824 | H | ↑ | # | ↑ | H |
| 1.536 | 2.818 | 4.242 | 2.818 | → | → | ↑ | ← |

| Extended map V | | | | Policy | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4.200 | 8.000 | 10.000 | G | → | → | → | G |
| 2.780 | # | 5.824 | H | ↑ | # | ↑ | H |
| 1.502 | 2.818 | 4.242 | 2.818 | ↑ | → | ↑ | ← |

### Simulation results

Policies computed by value iteration, simulated from S with episode seeds
1000 × batch seed + i, 100-step limit. Episodes: [outputs/runs_c.csv](outputs/runs_c.csv).

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| Part A map | 11 | 47 | 3 | 0 | 5.04 | 2.230 |
| Part A map | 22 | 44 | 6 | 0 | 4.96 | 1.442 |
| Part A map | 33 | 42 | 8 | 0 | 4.88 | 0.970 |
| Extended map | 11 | 50 | 0 | 0 | 5.00 | 1.502 |
| Extended map | 22 | 50 | 0 | 0 | 5.00 | 1.502 |
| Extended map | 33 | 50 | 0 | 0 | 5.00 | 1.502 |

### Explain your results

**What changed with sand and water.** Only one action changed: at (3,1) the
best move went from right (Part A map) to up (extended map). On the Part A map
the cheapest route goes along the bottom row and then up through the ice tile.
Putting sand (-2) at (3,2) makes that route more expensive: moving right from
(3,1) is now worth -2 + 0.9·2.818 = 0.536, while moving up is worth
-1 + 0.9·2.78 = 1.502. The new best route is S, up, up, right, right, right
along the top row through the mud. The value at S fell from 1.536 to 1.502.
Water (-5) did not change any value or action: the optimal policy never enters
it, so it only acts as a deterrent. (3,4) is the only tile where it appears
and nothing in the policy leads there.

**Where cost and risk mattered.** On the Part A map the policy accepts risk to
save cost: it goes through the ice. At the ice tile the best action is up. It
reaches (1,3) with probability 0.8, stays on the ice with probability 0.1
(slipping left into the wall) and enters the hole with probability 0.1
(slipping right). Repeating after staying gives a hole probability of
0.1 / 0.9 ≈ 11%, and the simulation gave 17 holes in 150 episodes (11.3%).
On the extended map, the sand makes the ice route costlier than the mud route,
which costs -3 for entering the mud but is deterministic, so the policy
avoids the ice completely: 0 holes in 150 episodes. So the extra floor cost
bought safety here.

**Simulation versus computed values.** V*(S) is the expectation of the
discounted return over an infinite horizon, whereas a simulation is a sample
mean over 50 episodes truncated at 100 steps. The truncation is negligible
(0.9^100 ≈ 2.7·10^-5). On the Part A map V*(S) = 1.536 and the batch means
were 2.230, 1.442 and 0.970 (average 1.547): they scatter around V*(S) because
a return is either about +3.5 (goal) or negative (hole), so 50 episodes are
noisy. On the extended map the policy is fully deterministic, so every episode
returns exactly 1.502 = V*(S); this is a special case, not a general rule.

### Rule checks

[outputs/checks.txt](outputs/checks.txt): 24 checks pass. They cover moves into
walls and outside the map (the agent stays and receives the reward of the tile
where it stays), ice probabilities 0.8/0.1/0.1 summing to 1 for all four
actions, rewards for mud (-3), sand (-2), water (-5), goal (+10) and hole
(-10), terminal values equal to 0, and an episode that stops at the goal with
no rewards afterwards (return 1.502 matching a hand calculation).

