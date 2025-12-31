import numpy as np
from lib import CudaProblem, Coord


# Puzzle 1: Map
# Implement a "kernel" (GPU function) that adds 10 to each position of vector a and stores it in vector out.
# You have 1 thread per position.

def map_spec(a):
    return a + 10

def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 lines)
        out[local_i] = map_spec(a[local_i])

    return call

SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)
problem.show()
problem.check()
