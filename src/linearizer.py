import ctypes as c

from typing import List, Callable, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# TODO: This dependency is sucks
from .code_gen.clang import CProgram

if TYPE_CHECKING:
    from .lazy import LazyOp, LazyBuffer


@dataclass(frozen=True)
class ScheduleItem:
    op: 'LazyOp'
    target: 'LazyBuffer'
    srcs: Tuple[Union['LazyBuffer', 'LazyOp'], ...]


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
