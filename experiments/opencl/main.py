import numpy as np
import pyopencl as cl

import time

# Define parameters
SIZE = 1024
TILE_SIZE = 32
GLOBAL_SIZE = (SIZE, SIZE)
LOCAL_SIZE = (16, 16)

# Don't ask to pick driver
context = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(context)

# Create random matrices
a = np.random.rand(SIZE, SIZE).astype(np.float32)
b = np.random.rand(SIZE, SIZE).astype(np.float32)

# Create OpenCL buffers
mf = cl.mem_flags
a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
res_buf = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)

# OpenCL program
program = cl.Program(context, """
__kernel void matmul(const int size,
                     __global const float16 *a,
                     __global const float16 *b,
                     __global float *res) {
    const int TILE_SIZE = 16;
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_group_id(0) * TILE_SIZE + row;
    const int globalCol = get_group_id(1) * TILE_SIZE + col;

    __local float16 tileA[TILE_SIZE][TILE_SIZE];
    __local float16 tileB[TILE_SIZE][TILE_SIZE];

    float16 sum = (float16)(0.0f);
    for (int m = 0; m < size/16; m+=TILE_SIZE) {
        tileA[row][col] = a[globalRow * size/16 + m + col];
        tileB[row][col] = b[globalCol * size/16 + m + row];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[row][k] * tileB[k][col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float total = 0.0f;
    total += sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7;
    total += sum.s8 + sum.s9 + sum.sa + sum.sb + sum.sc + sum.sd + sum.se + sum.sf;
    res[globalRow * size6 + globalCol] = total;
}
""").build()

# warm up
count = 5
for _ in range(10):
    matmul = program.matmul
    matmul(queue, GLOBAL_SIZE, LOCAL_SIZE, np.int32(SIZE), a_buf, b_buf, res_buf).wait()
queue.finish()
    

# run 10 times and average time
count = 10
start = time.time_ns()
for _ in range(count):
    event = matmul(queue, GLOBAL_SIZE, LOCAL_SIZE, np.int32(SIZE), a_buf, b_buf, res_buf)
event.wait()
took_time = (time.time_ns() - start) / count
print(f"Opencl time: {round(took_time / 1e6, 2)} ms.")


# we transpose it for numpy instead of our own implementation
b = b.T
# same for numpy
start = time.time_ns()
for _ in range(count):
    res_numpy = a @ b
took_time = (time.time_ns() - start) / count
print(f"Numpy time: {round(took_time / 1e6, 2)} ms.")

# copy result to numpy array
res_opencl = np.empty((SIZE, SIZE)).astype(np.float32)
cl.enqueue_copy(queue, res_opencl, res_buf)

if np.allclose(res_numpy, res_opencl):
    print("algorithm is correct")
else:
    print("algorithm is wrong")
    print(res_numpy[:10, 0])
    print(res_opencl[:10, 0])
