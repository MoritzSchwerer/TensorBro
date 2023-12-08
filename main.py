from src.lazy import LazyBuffer, LazyOp
from src.ops import LoadOps

# import numpy as np


if __name__ == '__main__':
    op1 = LazyOp(LoadOps.Rand, ())
    op2 = LazyOp(LoadOps.Rand, ())
    op3 = LazyOp(LoadOps.Rand, ())

    l1 = LazyBuffer(op1, 'CPU', (10, 10))
    l2 = LazyBuffer(op2, 'CPU', (10, 10))
    l3 = LazyBuffer(op2, 'CPU', (10, 10))
    l4 = l1 * l2 + l3
    sch = l4.schedule()
    for s in reversed(list(sch)):
        print(s)
