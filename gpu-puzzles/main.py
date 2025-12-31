import numba
import numpy as np
from lib import CudaProblem, Coord

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Puzzle 1: Map
# Implement a "kernel" (GPU function) that adds 10 to each position of vector a and stores it in vector out.
# You have 1 thread per position.

def map_spec(a):
    return a + 10

def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 lines)
        out[local_i] = a[local_i] + 10

    return call

SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)
problem.show()
problem.check()

# Puzzle 2 - Zip
# Implement a kernel that adds together each position of a and b and stores it in out.
# You have 1 thread per position.

def zip_spec(a, b):
    return a + b


def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + b[local_i]

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)
problem.show()
problem.check()

# Puzzle 3 - Guards
# Implement a kernel that adds 10 to each position of a and stores it in out.
# You have more threads than positions.

def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        if local_i < size:
            out[local_i] = a[local_i] + 10

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Guard",
    map_guard_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(8, 1),
    spec=map_spec,
)
problem.show()

# Puzzle 4 - Map 2D
# Implement a kernel that adds 10 to each position of a and stores it in out.
# Input a is 2D and square. You have more threads than positions.

def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        if local_i < size and local_j < size:
            out[local_j, local_i] = a[local_j, local_i] + 10

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
problem = CudaProblem(
    "Map 2D", map_2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), spec=map_spec
)
problem.show()
problem.check()

# Puzzle 5 - Broadcast
# Implement a kernel that adds a and b and stores it in out. 
# Inputs a and b are vectors. You have more threads than positions.

def broadcast_test(cuda):
    def call(out, a, b, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        if local_i < size and local_j < size:
            out[local_j, local_i] = a[local_j, 0] + b[0, local_i]

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE).reshape(SIZE, 1)
b = np.arange(SIZE).reshape(1, SIZE)
problem = CudaProblem(
    "Broadcast",
    broadcast_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    spec=zip_spec,
)
problem.show()
problem.check()

# Puzzle 6 - Blocks
# Implement a kernel that adds 10 to each position of a and stores it in out.
# You have fewer threads per block than the size of a.

def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < size:
            out[i] = a[i] + 10

    return call


SIZE = 9
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Blocks",
    map_block_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(4, 1),
    blockspergrid=Coord(3, 1),
    spec=map_spec,
)
problem.show()
problem.check()

# Puzzle 7 - Blocks 2D
# Implement the same kernel in 2D. You have fewer threads per block than the size of a in both directions.

def map_block2D_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if i < size and j < size:
            out[j, i] = a[j, i] + 10

    return call


SIZE = 5
out = np.zeros((SIZE, SIZE))
a = np.ones((SIZE, SIZE))

problem = CudaProblem(
    "Blocks 2D",
    map_block2D_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    blockspergrid=Coord(2, 2),
    spec=map_spec,
)
problem.show()
problem.check()

# Puzzle 8 - Shared
# Implement a kernel that adds 10 to each position of a and stores it in out. 
# You have fewer threads per block than the size of a.

TPB = 4
def shared_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()

        if i < size:
            out[i] = shared[local_i] + 10

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.ones(SIZE)
problem = CudaProblem(
    "Shared",
    shared_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(2, 1),
    spec=map_spec,
)
problem.show()
problem.check()

# Puzzle 9 - Pooling
# Implement a kernel that sums together the last 3 position of a and stores it in out. 
# You have 1 thread per position. You only need 1 global read and 1 global write per thread.

def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


TPB = 8
def pool_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        #
        temp = a[i] # last 3, local accumulator
        shared[i] = temp
        cuda.syncthreads()
        for prev in range(max(i - 2, 0), i):
            temp += shared[prev]
        out[i] = temp

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.arange(SIZE)
problem = CudaProblem(
    "Pooling",
    pool_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(1, 1),
    spec=pool_spec,
)
problem.show()
problem.check()

# Puzzle 10 - Dot Product
# Implement a kernel that computes the dot-product of a and b and stores it in out.
# You have 1 thread per position. You only need 2 global reads and 1 global write per thread.

def dot_spec(a, b):
    return a @ b

TPB = 8
def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 9 lines)

        temp = a[i] * b[i]
        shared[i] = temp
        cuda.syncthreads()

        # Problems with tree reduce:
        # - Using global indexes but that's fine with current shapes
        # - syncthreads in conditionals can lead to deadlock?

        # (Bad, first) Tree reduce - O(log TBP)
        # Score (Max Per Thread):
        # |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
        # |             6 |             1 |             4 |             3 |
        if i % 2 == 0:
            temp += shared[i + 1] # 0 has 0, 1 results
            shared[i] = temp
            cuda.syncthreads()
            if i == 0 or i == 4:
                temp += shared[i + 2] # 0 has 0, 1, 2, 3 results
                cuda.syncthreads()
                if i == 4:
                    shared[0] = temp
                cuda.syncthreads()
                if i == 0:
                    out[0] = temp + shared[0]
        
        # Naive loop - O(TBP)
        # Score (Max Per Thread):
        # |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
        # |             4 |             1 |             7 |             1 |
        # if i == 0:
        #     for idx in range(1, size):
        #         temp += shared[idx]
        #     out[0] = temp

    return call


SIZE = 8
out = np.zeros(1)
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "Dot",
    dot_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(SIZE, 1),
    blockspergrid=Coord(1, 1),
    spec=dot_spec,
)
problem.show()
problem.check()
