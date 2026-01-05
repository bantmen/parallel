import numba
from numba import cuda
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

# Puzzle 11 - 1D Convolution
# Implement a kernel that computes a 1D convolution between a and b and stores it in out.
# You need to handle the general case. You only need 2 global reads and 1 global write per thread.

def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
SHARED_A = TPB + MAX_CONV - 1  # a core + right halo
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        #

        shared_a = cuda.shared.array(SHARED_A, numba.float32)
        shared_b = cuda.shared.array(MAX_CONV, numba.float32)

        if i < a_size: # divide up a between blocks
            shared_a[local_i] = a[i]
            if local_i == TPB - 1: # the last thread also writes the halo
                for offset in range(MAX_CONV):
                    if i + offset < a_size:
                        shared_a[local_i + offset] = a[i + offset]

        if local_i < b_size: # store b for each block
            shared_b[local_i] = b[local_i]
        
        cuda.syncthreads()
    
        if i < a_size:
            temp = 0
            for offset in range(MAX_CONV):
                if local_i + offset < a_size and offset < b_size:
                    # shared_a: [0..SHARED_A-1]
                    # shared_b: [0..b_size-1]
                    temp += shared_a[local_i + offset] * shared_b[offset]
            
            out[i] = temp

    return call


# Test 1

SIZE = 6
CONV = 3
out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(CONV)
problem = CudaProblem(
    "1D Conv (Simple)",
    conv_test,
    [a, b],
    out,
    [SIZE, CONV],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()
problem.check()

# Test 2

out = np.zeros(15)
a = np.arange(15)
b = np.arange(4)
problem = CudaProblem(
    "1D Conv (Full)",
    conv_test,
    [a, b],
    out,
    [15, 4],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()
problem.check()

# Puzzle 12 - Prefix Sum
# Implement a kernel that computes a sum over a and stores it in out.
# If the size of a is greater than the block size, only store the sum of each block.

# NOTE: This is not really a prefix sum. We'll implement that later.

TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out


def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        #

        cache[local_i] = a[i] if i < size else 0.0 # prevent partial block from corrupting the sum
        cuda.syncthreads()
        
        # Tree reduction
        stride = 1 # 1, 2, 4
        while stride < TPB:
            participates = local_i % (stride * 2) == 0
            if participates:
                temp = cache[local_i] + cache[local_i + stride]
            cuda.syncthreads() # let all threads reach to prevent deadlock
            if participates:
                cache[local_i] = temp
            cuda.syncthreads()
            stride *= 2

        if local_i == 0:
            out[cuda.blockIdx.x] = cache[0]

    return call


# Test 1

SIZE = 8
out = np.zeros(1)
inp = np.arange(SIZE)
problem = CudaProblem(
    "Sum (Simple)",
    sum_test,
    [inp],
    out,
    [SIZE],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()
problem.check()

# Test 2

SIZE = 15
out = np.zeros(2)
inp = np.arange(SIZE)
problem = CudaProblem(
    "Sum (Full)",
    sum_test,
    [inp],
    out,
    [SIZE],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()
problem.check()

# Puzzle 13 - Axis Sum
# Implement a kernel that computes a sum over each column of a and stores it in out.

TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[..., j] = a[..., i : i + TPB].sum(-1)
    return out


# not working, why does it think it's kernel rather than device function?
@cuda.jit(device=True, inline=True)
def block_reduce_sum(cache, tid):
    stride = 1
    while stride < TPB:
        participates = (tid % (2 * stride) == 0)
        if participates:
            tmp = cache[tid] + cache[tid + stride]
        cuda.syncthreads()
        if participates:
            cache[tid] = tmp
        cuda.syncthreads()
        stride *= 2


def axis_sum_test(cuda): # each block sums across its column
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        batch = cuda.blockIdx.y

        # 

        # cache over column
        cache[local_i] = a[batch, i] if i < size else 0.0
        cuda.syncthreads()

        # block_reduce_sum(cache, local_i)
        # tree reduce
        stride = 1 # 1, 2, 4
        while stride < TPB:
            participates = local_i % (stride * 2) == 0
            if participates:
                temp = cache[local_i] + cache[local_i + stride]
            cuda.syncthreads() # let all threads reach to prevent deadlock
            if participates:
                cache[local_i] = temp
            cuda.syncthreads()
            stride *= 2

        if local_i == 0:
            out[batch, 0] = cache[0]

    return call


BATCH = 4
SIZE = 6
out = np.zeros((BATCH, 1))
inp = np.arange(BATCH * SIZE).reshape((BATCH, SIZE))
problem = CudaProblem(
    "Axis Sum",
    axis_sum_test,
    [inp],
    out,
    [SIZE],
    Coord(1, BATCH),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()

# Puzzle 14 - Matrix Multiply!
# Implement a kernel that multiplies square matrices a and b and stores the result in out.

def matmul_spec(a, b):
    return a @ b


TPB = 3
def mm_oneblock_test(cuda): # not actually one block
    def call(out, a, b, size: int) -> None:
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y

        # 

        # Read the tile.
        # - A tile slides right.
        # - B tile slides down.
        # This way, we always have what we need to compute partial dot products.

        tile_offset = 0
        temp = 0
        while tile_offset < size:
            # Read tile
            a_shared[local_j, local_i] = a[j, local_i + tile_offset] if local_i + tile_offset < size and j < size else 0.0
            b_shared[local_j, local_i] = b[local_j + tile_offset, i] if i < size and local_j + tile_offset < size else 0.0
            cuda.syncthreads()
            # Accumulate dot product
            if i < size and j < size:
                for idx in range(TPB):
                    temp += a_shared[local_j, idx] * b_shared[idx, local_i]
            tile_offset += TPB
            cuda.syncthreads()
        if i < size and j < size:
            out[j, i] = temp # single global write

    return call

# Test 1

SIZE = 2
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "Matmul (Simple)",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(1, 1),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)
problem.check()

# Test 2

SIZE = 8
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "Matmul (Full)",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(3, 3),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)
problem.check()

### Custom puzzles ###

# Actual prefix sum

TPB = 8
ITEMS_PER_THREAD = 4
SIZE = TPB * ITEMS_PER_THREAD  # 32 (> TPB)

def prefix_spec(a):
    return np.cumsum(a).astype(np.float32)

def prefix_test(cuda):
    def call(out, a, size: int) -> None:
        # NOTE: Can't use local array because of weirdness with the harness, we dont need this to be shared memory
        # thread_vals = numba.cuda.local.array(SIZE, numba.float32) # holds per-element partials
        thread_vals = cuda.shared.array(SIZE, numba.float32)   # holds per-element partials
        shared_sums = cuda.shared.array(TPB, numba.float32)    # holds per-thread totals

        local_i = cuda.threadIdx.x
        # Each thread handles a contiguous "stripe" of ITEMS_PER_THREAD elements
        base = local_i * ITEMS_PER_THREAD

        # each thread cumsums its own items
        temp = 0
        for offset in range(ITEMS_PER_THREAD):
            global_idx = base + offset
            if global_idx < size:
                temp += a[global_idx]
                thread_vals[global_idx] = temp
        shared_sums[local_i] = temp
        cuda.syncthreads()

        # hillis-steele cumsum on shared_sums
        lookback = 1
        while lookback < TPB:
            participates =  local_i - lookback >= 0
            if local_i - lookback >= 0:
                temp = shared_sums[local_i] + shared_sums[local_i - lookback]
            cuda.syncthreads()
            if participates:
                shared_sums[local_i] = temp
            cuda.syncthreads()
            lookback *= 2
            
        # each thread gets the final cumsums
        prev_sum = shared_sums[local_i - 1] if local_i - 1 >= 0 else 0
        for offset in range(ITEMS_PER_THREAD):
            global_idx = base + offset
            if global_idx < size:
                out[global_idx] = thread_vals[global_idx] + prev_sum

    return call


# Single test (size > TPB)
inp = np.arange(SIZE, dtype=np.float32)
out = np.zeros(SIZE, dtype=np.float32)

problem = CudaProblem(
    name="Prefix Sum (Inclusive, size>TPB, 1 block)",
    fn=prefix_test,
    inputs=[inp],
    out=out,
    args=[SIZE],
    blockspergrid=Coord(1, 1),
    threadsperblock=Coord(TPB, 1),
    spec=prefix_spec,
)

problem.show()
problem.check()
