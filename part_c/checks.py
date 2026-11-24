import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from floors import FLOORS, MAP_A, MAP_EXT
from solver import transitions, move, is_terminal, value_iteration, nonterminal_states, ORDER
from simulate import run_episode, MAX_STEPS

log = []
def check(label, cond):
    assert cond, "FAILED: " + label
    log.append("PASS: " + label)

# 1. walls and boundaries (0-based indices; paper (r,c) = (r+1,c+1))
check("up from top row (0,0) stays at (0,0)", move(MAP_A, 0, 0, 'U') == (0, 0))
check("left from left edge (1,0) stays", move(MAP_A, 1, 0, 'L') == (1, 0))
check("down from bottom row (2,0) stays", move(MAP_A, 2, 0, 'D') == (2, 0))
check("right into wall from (1,0) stays", move(MAP_A, 1, 0, 'R') == (1, 0))
check("blocked move still pays reward of the tile where it ends (-1)",
      transitions(MAP_A, 1, 0, 'R') == [(1.0, 1, 0, -1)])
check("blocked move on mud stays on mud and pays -3",
      transitions(MAP_A, 0, 1, 'D') == [(1.0, 0, 1, -3)])

# 2. ice probabilities
for a in ORDER:
    ts = transitions(MAP_A, 1, 2, a)
    check(f"ice action {a}: probabilities sum to 1 ({[p for p, *_ in ts]})",
          abs(sum(p for p, *_ in ts) - 1) < 1e-12)
ts = transitions(MAP_A, 1, 2, 'U')
check("ice up: 0.8 / 0.1 / 0.1", sorted(p for p, *_ in ts) == [0.1, 0.1, 0.8])
check("ice 'up', slip left hits wall -> stays on ice, reward -1", (0.1, 1, 2, -1) in ts)
check("ice 'up', slip right enters hole, reward -10", (0.1, 1, 3, -10) in ts)
check("non-ice tiles are deterministic",
      all(len(transitions(MAP_EXT, r, c, a)) == 1
          for r, c in nonterminal_states(MAP_EXT) if MAP_EXT[r][c] != 'I' for a in ORDER))

# 3. floor rewards
check("mud reward -3", transitions(MAP_A, 0, 0, 'R')[0][3] == -3)
check("sand reward -2", transitions(MAP_EXT, 2, 0, 'R')[0][3] == -2)
check("water reward -5", transitions(MAP_EXT, 2, 2, 'R')[0][3] == -5)
check("entering goal gives +10 (no extra floor cost)", transitions(MAP_A, 0, 2, 'R')[0][3] == 10)
check("entering hole gives -10", transitions(MAP_A, 2, 3, 'U')[0][3] == -10)
check("sand and water moves are deterministic", FLOORS['W']['slip'] == 0 and FLOORS['D']['slip'] == 0)

# 4. terminal handling
for g in (MAP_A, MAP_EXT):
    V, *_ = value_iteration(g)
    check("terminal values are 0",
          all(V[r][c] == 0 for r in range(len(g)) for c in range(len(g[0])) if is_terminal(g, r, c)))
check("terminals are not updated states",
      all(not is_terminal(MAP_A, r, c) for r, c in nonterminal_states(MAP_A)))
pol = {s: 'R' for s in nonterminal_states(MAP_EXT)}
pol.update({(2, 0): 'U', (1, 0): 'U'})            # S up, up, then right along the top row
steps, ret, out = run_episode(MAP_EXT, pol, 1)
expected = -1 - 0.9 - 3 * 0.9**2 - 0.9**3 + 10 * 0.9**4
check("path S,U,U,R,R,R: stops at goal after 5 steps, return %.4f (got %.4f, %s)"
      % (expected, ret, out), out == "goal" and steps == 5 and abs(ret - expected) < 1e-12)

os.makedirs("outputs", exist_ok=True)
open("outputs/checks.txt", "w").write("\n".join(log) + "\n")
print("\n".join(log))