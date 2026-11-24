# Actions in FrozenLake: 0=left, 1=down, 2=right, 3=up
# State = row*4 + col. Map:
#  S F F F     0  1  2  3
#  F H F H     4  5  6  7
#  F F F H     8  9 10 11
#  H F F G    12 13 14 15
FIXED_POLICY = {
    0: 1,   # down
    1: 3,   # up
    2: 3,   # up
    3: 3,   # up
    4: 1,   # down
    6: 0,   # left
    8: 2,   # right
    9: 1,   # down
    10: 0,  # left
    13: 2,  # right
    14: 2,  # right
}