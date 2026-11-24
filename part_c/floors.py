"""Floor rules, kept separate from the solver.
reward:   reward when a move ENDS on this tile
slip:     probability of EACH perpendicular direction when starting a move here
          (the chosen direction then has probability 1 - 2*slip)
terminal: episode ends when entering this tile
To add a floor, add one line here; solver.py does not change."""
FLOORS = {
    '.': dict(reward=-1,  slip=0.0, terminal=False),
    'S': dict(reward=-1,  slip=0.0, terminal=False),
    'M': dict(reward=-3,  slip=0.0, terminal=False),
    'I': dict(reward=-1,  slip=0.1, terminal=False),
    'G': dict(reward=10,  slip=0.0, terminal=True),
    'H': dict(reward=-10, slip=0.0, terminal=True),
    'D': dict(reward=-2,  slip=0.0, terminal=False),
    'W': dict(reward=-5,  slip=0.0, terminal=False),
}
WALL = '#'

MAP_A = [".M.G",
         ".#IH",
         "S..."]
MAP_EXT = [".M.G",
           ".#IH",
           "SD.W"]