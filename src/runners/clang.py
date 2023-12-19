import ctypes as c

from typing import TYPE_CHECKING, List, Callable

if TYPE_CHECKING:
    from ..lazy import ScheduleItem


class _CAllocator:
    @staticmethod
    def alloc(dtype, size):
        return (dtype * size)()

    def free(self, pointer):
        c.free(pointer)


CAllocator = _CAllocator()


def linearize(schedule: List[ScheduleItem]) -> Callable:
    """
    Linearizes code from ast

    Args: schedule
    returns: program that can be run


    Turns asts into assignment sequences
    each essignment has:
        Source(s): ex. inp1, inp2
        Target: out
        dtype: float32 (only support float32)
        programm: c code that takes the inputs and produces the output
    """
    programms = []
    # buffers = []
    for s in schedule:
        op = s.op
        shape = s.target.shape
        prg = CProgram(op.op, shape, c.c_float, arg=op.arg)
        programms.append(prg)

    def runner():
        for prg, s in zip(programms, schedule):
            target = s.target
            target.realize()
            prg(target.base, *[src.base for src in s.srcs])

    return runner
