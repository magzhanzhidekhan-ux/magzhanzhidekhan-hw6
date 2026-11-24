import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from floors import MAP_A, MAP_EXT
from solver import (GAMMA, transitions, is_terminal, value_iteration, extract_policy)

BATCH_SEEDS = [11, 22, 33]
N, MAX_STEPS = 50, 100


def find_start(grid):
    for r, row in enumerate(grid):
        for c, t in enumerate(row):
            if t == 'S':
                return r, c


def run_episode(grid, pol, seed):
    rng = np.random.default_rng(seed)
    r, c = find_start(grid)
    ret, steps, disc = 0.0, 0, 1.0
    while steps < MAX_STEPS:
        outs = transitions(grid, r, c, pol[(r, c)])
        i = rng.choice(len(outs), p=[o[0] for o in outs])
        _, r, c, rew = outs[i]
        ret += disc * rew                          # r0 + 0.9 r1 + 0.9^2 r2 + ...
        disc *= GAMMA
        steps += 1
        if is_terminal(grid, r, c):                # no rewards after termination
            return steps, ret, "goal" if grid[r][c] == 'G' else "hole"
    return steps, ret, "time limit"


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    rows = []
    for name, grid in (("part_a_map", MAP_A), ("extended_map", MAP_EXT)):
        V, *_ = value_iteration(grid)
        pol, _ = extract_policy(grid, V)
        sr, sc = find_start(grid)
        print(f"{name}: V*(S) = {V[sr][sc]:.4f}")
        for bs in BATCH_SEEDS:
            for i in range(N):
                seed = 1000 * bs + i
                steps, ret, outcome = run_episode(grid, pol, seed)
                rows.append(dict(map=name, batch_seed=bs, episode_seed=seed,
                                 steps=steps, discounted_return=ret, outcome=outcome))
    with open("outputs/runs_c.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("part_a_map", "extended_map"):
        for bs in BATCH_SEEDS:
            sub = [x for x in rows if x["map"] == name and x["batch_seed"] == bs]
            cnt = lambda o: sum(x["outcome"] == o for x in sub)
            lines.append(f"| {name} | {bs} | {cnt('goal')} | {cnt('hole')} | {cnt('time limit')} | "
                         f"{np.mean([x['steps'] for x in sub]):.2f} | "
                         f"{np.mean([x['discounted_return'] for x in sub]):.3f} |")
    open("outputs/summary_c.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))