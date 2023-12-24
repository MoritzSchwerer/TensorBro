import numpy as np
import pyopencl as cl

import time

# Define parameters
SIZE = 1024 * 8
GLOBAL_SIZE = (SIZE, SIZE//16)
LOCAL_SIZE = None

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
__kernel void mul(const int size,
                     __global const float16 *a,
                     __global const float16 *b,
                     __global float *res) {
    const int numOps = 1;
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float16 temp = 0.0f;
    for (int k = 0; k < numOps; k++) {
        temp = a[(i*numOps + k) * size/16 + j] * b[(i*numOps + k) * size/16 + j];
        res[fma(i, numOps, k)* size + j*16 + 0] = temp.s0;
        res[(i*numOps + k)* size + j*16 + 1] = temp.s1;
        res[(i*numOps + k)* size + j*16 + 2] = temp.s2;
        res[(i*numOps + k)* size + j*16 + 3] = temp.s3;
        res[(i*numOps + k)* size + j*16 + 4] = temp.s4;
        res[(i*numOps + k)* size + j*16 + 5] = temp.s5;
        res[(i*numOps + k)* size + j*16 + 6] = temp.s6;
        res[(i*numOps + k)* size + j*16 + 7] = temp.s7;
        res[(i*numOps + k)* size + j*16 + 8] = temp.s8;
        res[(i*numOps + k)* size + j*16 + 9] = temp.s9;
        res[(i*numOps + k)* size + j*16 + 10] = temp.sa;
        res[(i*numOps + k)* size + j*16 + 11] = temp.sb;
        res[(i*numOps + k)* size + j*16 + 12] = temp.sc;
        res[(i*numOps + k)* size + j*16 + 13] = temp.sd;
        res[(i*numOps + k)* size + j*16 + 14] = temp.se;
        res[(i*numOps + k)* size + j*16 + 15] = temp.sf;
    }
}
""").build()

mul = program.mul
# warm up
count = 5
for _ in range(10):
    mul(queue, GLOBAL_SIZE, LOCAL_SIZE, np.int32(SIZE), a_buf, b_buf, res_buf).wait()
queue.finish()
    

# run 10 times and average time
count = 10
start = time.time_ns()
for _ in range(count):
    event = mul(queue, GLOBAL_SIZE, LOCAL_SIZE, np.int32(SIZE), a_buf, b_buf, res_buf)
event.wait()
took_time = (time.time_ns() - start) / count
print(f"Opencl time: {round(took_time / 1e6, 2)} ms.")


# same for numpy
start = time.time_ns()
for _ in range(count):
    res_numpy = a * b
took_time = (time.time_ns() - start) / count
print(f"Numpy time: {round(took_time / 1e6, 2)} ms.")

# copy result to numpy array
res_opencl = np.empty((SIZE, SIZE)).astype(np.float32)
cl.enqueue_copy(queue, res_opencl, res_buf)

if np.allclose(res_numpy, res_opencl):
    print("algorithm is correct")
else:
    print("algorithm is wrong")
    print(res_numpy[0, :10])
    print(res_opencl[0, :10])
