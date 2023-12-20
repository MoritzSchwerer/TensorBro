from src.lazy import LazyBuffer
from src.linearizer import linearize
from src.ops import UnaryOps


b1 = LazyBuffer.rand((10, ), 'CPU', arg=1)


ops = [
    UnaryOps.NEG,
    UnaryOps.SIN,
    UnaryOps.SQRT,
    UnaryOps.EXP2,
    UnaryOps.LOG2,
]
for op in ops:
    print("="*99)
    print(op)
    b4 = b1.elementwise(op)
    sch = b4.schedule()
    runner = linearize(sch)
    runner()

    print('[', end='')
    for j in range(10):
        print(b4.base[+j], end=', ')
    print(']')

print("="*99)
print("base value")
print('[', end='')
for j in range(10):
    print(b1.base[j], end=', ')
print(']')
