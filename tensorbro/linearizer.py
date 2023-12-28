
from typing import List, Callable, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# TODO: This dependency sucks
from .code_gen.clang import CProgram

from .lazy import LazyOp

if TYPE_CHECKING:
    from .lazy import LazyBuffer



@dataclass(frozen=True)
class ScheduleItem:
    op: LazyOp
    target: 'LazyBuffer'
    srcs: Tuple[Union['LazyBuffer', LazyOp], ...]


def print_tree(si: ScheduleItem, depth=0):
    if isinstance(si, ScheduleItem):
        print(
            '    ' * depth + str(si.op.op.name),
            '' if si.op.arg is None else str(si.op.arg),
        )
        for src in si.op.srcs:
            print_tree(src, depth=depth + 1)
    elif isinstance(si, LazyOp):
        print('   ' * depth + str(si.op.name), '' if si.arg is None else str(si.arg))
        for src in si.srcs:
            print_tree(src, depth=depth + 1)
    else:
        print('    ' * depth + str(type(si).__name__))


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
    # print(*schedule, sep='\n')
    programms = []
    # buffers = []
    for s in schedule:
        prg = CProgram(s)
        programms.append(prg)

    def runner():
        for prg, s in zip(programms, schedule):
            target = s.target
            target.realize()
            prg(target.base, *[src.base for src in s.srcs])

    return runner
