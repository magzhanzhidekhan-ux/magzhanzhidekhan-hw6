import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from floors import FLOORS, WALL, MAP_A, MAP_EXT

GAMMA = 0.9
ORDER = ['U', 'R', 'D', 'L']                      # tie-break order
DIRS = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}
PERP = {'U': 'LR', 'D': 'LR', 'L': 'UD', 'R': 'UD'}
ARROW = {'U': '↑', 'R': '→', 'D': '↓', 'L': '←'}


def is_terminal(grid, r, c):
    return grid[r][c] != WALL and FLOORS[grid[r][c]]['terminal']


def is_state(grid, r, c):
    return grid[r][c] != WALL


def move(grid, r, c, d):
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])) or grid[nr][nc] == WALL:
        return r, c                                # wall/edge: stay in place
    return nr, nc


def transitions(grid, r, c, a):
    """List of (prob, next_r, next_c, reward). Slipping depends on the tile we LEAVE."""
    slip = FLOORS[grid[r][c]]['slip']
    if slip > 0:
        outcomes = [(1 - 2 * slip, a)] + [(slip, p) for p in PERP[a]]
    else:
        outcomes = [(1.0, a)]
    res = []
    for p, d in outcomes:
        nr, nc = move(grid, r, c, d)
        res.append((p, nr, nc, FLOORS[grid[nr][nc]]['reward']))
    return res


def q_value(grid, V, r, c, a):
    return sum(p * (rew + GAMMA * V[nr][nc]) for p, nr, nc, rew in transitions(grid, r, c, a))


def nonterminal_states(grid):
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
            if is_state(grid, r, c) and not is_terminal(grid, r, c)]


def value_iteration(grid, eps=1e-6, max_iter=10000, record=None):
    R, C = len(grid), len(grid[0])
    V = [[0.0] * C for _ in range(R)]
    states = nonterminal_states(grid)
    delta = None
    for k in range(1, max_iter + 1):
        newV = [row[:] for row in V]
        delta = 0.0
        for r, c in states:                        # only previous-round V is read
            newV[r][c] = max(q_value(grid, V, r, c, a) for a in ORDER)
            delta = max(delta, abs(newV[r][c] - V[r][c]))
        V = newV
        if record:
            record(k, V)
        if delta < eps:
            return V, k, delta, True
    return V, max_iter, delta, False                # safeguard reached: not converged


def extract_policy(grid, V, tol=1e-9):
    """Greedy w.r.t. V. Ties (within tol) -> first action in ORDER U,R,D,L."""
    pol, ties = {}, {}
    for r, c in nonterminal_states(grid):
        qs = {a: q_value(grid, V, r, c, a) for a in ORDER}
        best = max(qs.values())
        tied = [a for a in ORDER if qs[a] >= best - tol]
        pol[(r, c)] = tied[0]
        if len(tied) > 1:
            ties[(r, c)] = tied
    return pol, ties


def fmt_values(grid, V, w=8):
    lines = []
    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            t = grid[r][c]
            row.append('#'.center(w) if t == WALL else
                       (t if is_terminal(grid, r, c) else f"{V[r][c]:.3f}").center(w))
        lines.append(' '.join(row))
    return '\n'.join(lines)


def fmt_policy(grid, pol):
    lines = []
    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            t = grid[r][c]
            row.append(t if (t == WALL or is_terminal(grid, r, c)) else ARROW[pol[(r, c)]])
        lines.append(' '.join(x.center(3) for x in row))
    return '\n'.join(lines)


def solve_and_report(name, grid, out_values, out_policy):
    snaps = {}
    def rec(i, v):
        if i <= 2:
            snaps[i] = [row[:] for row in v]
    V, k, delta, ok = value_iteration(grid, record=rec)
    pol, ties = extract_policy(grid, V)
    head = f"=== {name} ==="
    out_values.write(head + "\n")
    out_values.write("V1:\n" + fmt_values(grid, snaps[1]) + "\n")
    out_values.write("V2:\n" + fmt_values(grid, snaps[2]) + "\n")
    out_values.write(f"Final V:\n{fmt_values(grid, V)}\n")
    out_values.write(f"iterations={k}, final max change={delta:.3e}, converged={ok}\n\n")
    out_policy.write(head + "\n" + fmt_policy(grid, pol) + "\n")
    out_policy.write("Tie-break: first of U, R, D, L among actions within 1e-9 of the best Q.\n")
    tied = {(r + 1, c + 1): [ARROW[a] for a in t] for (r, c), t in ties.items()}
    out_policy.write(f"Tied states (row, col, 1-based): {tied}\n\n")
    return V, pol, snaps, k, delta, ok


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # arrows in Windows console
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/value_maps.txt", "w", encoding="utf-8") as fv, \
         open("outputs/policy_maps.txt", "w", encoding="utf-8") as fp:
        for name, grid in (("Part A map", MAP_A), ("Extended map", MAP_EXT)):
            V, pol, snaps, k, delta, ok = solve_and_report(name, grid, fv, fp)
            print(f"=== {name} ===")
            print("V1:\n" + fmt_values(grid, snaps[1]))
            if grid is MAP_A:
                print("V2 at (1,2), (1,3), (2,3):",
                      [round(snaps[2][r][c], 4) for r, c in ((0, 1), (0, 2), (1, 2))])
            print("Final V:\n" + fmt_values(grid, V))
            print("Policy:\n" + fmt_policy(grid, pol))
            print(f"iterations={k}, final change={delta:.3e}, converged={ok}\n")