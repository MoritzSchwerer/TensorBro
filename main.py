from tensorbro.lazy import LazyBuffer
from tensorbro.ops import UnaryOps
from tensorbro.linearizer import linearize

def show_buffer_2d(buffer, shape, num_decimals=4):
    assert len(shape) <= 2, f"cannot show {len(shape)}-dimensional tensor."
    print('[', end='')
    for i in range(shape[0]):
        for j in range(shape[1]):
            print(round(buffer.base[i*shape[1]+j], num_decimals), end='' if i == shape[0]-1 and j == shape[1]-1 else ", ")
        print("", end="" if i == shape[0]-1 and j==shape[1]-1 else "\n")
    print(']')

b1 = LazyBuffer.rand((2, 10), 'CPU', seed=1) 
operations = [UnaryOps.NEG, UnaryOps.SIN, UnaryOps.SQRT, UnaryOps.EXP2, UnaryOps.LOG2]
results = [b1.elementwise(op) for op in operations]
print('='*99)
for result in results:
    print(result)
print("="*99)


for result in results:
    linearize(result.schedule())()
    show_buffer_2d(result, (2, 10), num_decimals=3)

print("base value")
