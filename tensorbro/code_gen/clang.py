import ctypes as c
import subprocess
import math

from cgen import (
    FunctionBody,
    FunctionDeclaration,
    POD,
    Value,
    Pointer,
    Module,
    Block,
    Assign,
    For,
    Statement,
    Include,
)
from pathlib import Path
from typing import List, Tuple, Any, Type

from ..ops import BinaryOps, LoadOps, OpType, UnaryOps, ReduceOps, MovementOps

ALWAYS_RECOMPILE = True

CHARACTERS = list(map(chr, range(97, 123)))

def permute_shape(shape, dim):
    return tuple(map(lambda i: shape[i], dim))

bops_to_cstyle = {
    BinaryOps.MUL: '*',
    BinaryOps.ADD: '+',
    BinaryOps.SUB: '-',
    BinaryOps.DIV: '/',
}

uops_to_cstyle = {
    UnaryOps.NEG: lambda name: f"-{name}",
    UnaryOps.SIN: lambda name: f"sinf({name})",
    UnaryOps.SQRT: lambda name: f"sqrtf({name})",
    UnaryOps.EXP2: lambda name: f"exp2f({name})",
    UnaryOps.LOG2: lambda name: f"log2f({name})",
}


def ctype2str(t):
    return t.__name__[2:]

def gen_indices_strided(shape, stride, chars=None):
    if chars is None:
        chars = CHARACTERS
    idx_calc = ""
    for i in range(len(shape)):
        if stride[i] != 1:
            continue
        idx_calc += f"{chars[i]}"
        for j, s in enumerate(shape[i+1:]):
            if stride[i+1+j] != 1:
                continue
            idx_calc += f" * {s}"
        idx_calc += " + "
    idx_calc = idx_calc[:-3]

    return idx_calc

def gen_n_for_loops(shape: Tuple[int, ...], body: str):
    char = CHARACTERS[len(shape) - 1]
    loops = For(
        f'int {char} = 0',
        f'{char}<{shape[-1]}',
        f'{char}++',
        body
    )
    for i, dim in enumerate(reversed(shape[:-1])):
        char = CHARACTERS[len(shape) - 2 - i]
        loops = For(f'int {char} = 0',
            f'{char}<{shape[len(shape)-2 -i]}',
            f'{char}++',
            Block([loops])
        )
    return loops

def c_movement(function_name: str, op: MovementOps, shape: Tuple[int, ...], stride: Tuple[int, ...] = (), permute_dim: Tuple[int, ...] = (), dtype=c.c_float) -> Module:
    new_shape = permute_shape(shape, permute_dim)
    new_chars = permute_shape(CHARACTERS, permute_dim)
    idx = Assign("int idx", gen_indices_strided(shape, stride))
    new_idx = Assign("int newIdx", gen_indices_strided(new_shape, tuple([1 for _ in shape]), new_chars))
    assignment = Assign("out[newIdx]", "inp1[idx]")
    body = Block([idx, new_idx, assignment])
    body = gen_n_for_loops(shape, body)
    code = Module(
        [
            Include("math.h"),
            FunctionBody(
                FunctionDeclaration(
                    Value('void', function_name),
                    arg_decls=[Pointer(POD(dtype, name)) for name in ['out', "inp1"]],
                ),
                Block(
                    [
                        body
                    ]
                ),
            )
        ]
    )
    return code

def gen_accumulating_loops(shape: Tuple[int, ...], reduce_dim: int, body):
    char = CHARACTERS[len(shape) - 1]
    loops = For(
        f'int {char} = 0',
        f'{char}<{shape[-1]}',
        f'{char}++',
        body
    )
    for i, dim in enumerate(reversed(shape[:-1])):
        char = CHARACTERS[len(shape) - 2 - i]
        loops = For(f'int {char} = 0',
            f'{char}<{shape[len(shape) - 2 - i]}',
            f'{char}++',
            Block([loops])
        )
    return loops

def gen_index_accumulating(shape, stride, acc_dim):
    idx_calc = ""
    for i in range(len(shape)):
        if i == acc_dim:
            continue
        idx_calc += f"{CHARACTERS[i]}"
        for j in range(i+1,len(shape)):
            idx_calc += f" * {shape[j]}" if j != acc_dim else ""
        idx_calc += " + "
    idx_calc = idx_calc[:-3]
    return idx_calc

def gen_index_from_shape(shape, stride):
    idx_calc = ""
    for i in range(len(shape)):
        if stride[i] != 1:
            continue
        idx_calc += f"{CHARACTERS[i]}"
        for j in range(i+1,len(shape)):
            idx_calc += f" * {shape[j]}" if stride[j] == 1 else ""
        idx_calc += " + "
    idx_calc = idx_calc[:-3]
    return idx_calc
    
def c_reduce(function_name: str, op: ReduceOps, shape: Tuple[int, ...], stride: Tuple[int, ...], dtype=c.c_float, arg=0) -> Module:
    accIdx = Assign("int accIdx", gen_index_accumulating(shape, stride, arg))
    idx = Assign("int idx", gen_index_from_shape(shape, stride))
    if op is ReduceOps.SUM:
        acc_op = Assign("out[accIdx]", "out[accIdx] + inp1[idx]") 
    elif op is ReduceOps.MAX: 
        acc_op = Assign("out[accIdx]", "(out[accIdx] > inp1[idx]) ? out[accIdx] : inp1[idx]") 
    else:
        raise NotImplementedError(f"op: {op} not implemented for clang reduce")

    body = Block([idx, accIdx, acc_op])
    body = gen_accumulating_loops(shape, arg, body)

    code = Module(
        [
            Include("math.h"),
            FunctionBody(
                FunctionDeclaration(
                    Value('void', function_name),
                    arg_decls=[Pointer(POD(dtype, name)) for name in ['out', "inp1"]],
                ),
                Block(
                    [
                        body
                    ]
                ),
            )
        ]
    )
    return code



def c_unary(function_name: str, op: UnaryOps, shape: Tuple[int, ...], dtype=c.c_float) -> Module:
    assignment = Assign("out[i]", uops_to_cstyle[op]('inp1[i]'))
    code = Module(
        [
            Include("math.h"),
            FunctionBody(
                FunctionDeclaration(
                    Value('void', function_name),
                    arg_decls=[Pointer(POD(dtype, name)) for name in ['out', "inp1"]],
                ),
                Block(
                    [
                        For(
                            'int i = 0',
                            f'i<{math.prod(shape)}',
                            'i++',
                            Block([assignment]),
                        ),
                    ]
                ),
            )
        ]
    )
    return code


def c_binary(function_name: str, op: BinaryOps, shape: Tuple[int, ...], *strides: Tuple[int, ...], dtype=c.c_float) -> Module:
    includes = []
    if op is BinaryOps.MAX:
        assignment = Assign('out[outIdx]', 'fmax(inp1[idx1], inp2[idx2])')
        includes.append(Include("math.h"))
    else:
        assignment = Assign('out[outIdx]', f'inp1[idx1] {bops_to_cstyle[op]} inp2[idx2]')

    idx1 = Assign('int idx1', gen_indices_strided(shape, strides[0]))
    idx2 = Assign('int idx2', gen_indices_strided(shape, strides[1]))
    out_idx = Assign('int outIdx', gen_indices_strided(shape, tuple([1 for _ in range(len(strides[0]))])))
    block = Block([
        idx1,
        idx2,
        out_idx,
        assignment
    ])
    loops = gen_n_for_loops(shape ,block)
    code = Module(
        [
            *includes,
            FunctionBody(
                FunctionDeclaration(
                    Value('void', function_name),
                    arg_decls=[Pointer(POD(dtype, name)) for name in ['out', 'inp1', 'inp2']],
                ),
                Block(
                    [
                        loops
                    ]
                ),
            )
        ]
    )
    return code




def c_load(function_name: str, op, shape, dtype=c.c_float, arg=None):
    if op is LoadOps.RAND:
        assert arg is not None, 'We need to provide a seed for the rand function'
        includes = ['stdlib.h']
        prefix = Statement(f'srand({arg})')
        assignment = Assign('out[i]', '(float)rand() / (float)(RAND_MAX)')
    elif op is LoadOps.CONST:
        assert arg is not None, 'Need to provide const value'
        includes = []
        prefix = None
        assignment = Assign('out[i]', f'{float(arg)}')
    elif op is LoadOps.EMPTY:
        includes = []
        prefix = None
        assignment = Assign('out[i]', f'{float(0)}')
    else:
        raise NotImplementedError(f'c_load not implemented for {op}.')
    code = Module(
        [
            *[Include(include) for include in includes],
            FunctionBody(
                FunctionDeclaration(Value('void', function_name), arg_decls=[Pointer(POD(dtype, 'out'))]),
                Block(
                    [
                        prefix,
                        For('int i = 0', f'i<{math.prod(shape)}', 'i++', Block([assignment])),
                    ]
                    if prefix is not None
                    else [For('int i = 0', f'i<{math.prod(shape)}', 'i++', Block([assignment]))]
                ),
            ),
        ]
    )
    return code


def c_generator(func_name: str, op: OpType, shape, *strides, dtype=c.c_float, arg=None) -> Module:
    if op in BinaryOps:
        return c_binary(func_name, op, shape, *strides, dtype=dtype) # type: ignore
    elif op in UnaryOps:
        strided_shape = tuple([sh//st for sh,st in zip(shape, strides[0])])
        return c_unary(func_name, op, strided_shape, dtype=dtype) # type: ignore
    elif op in LoadOps:
        strided_shape = tuple([sh//st for sh,st in zip(shape, strides)])
        return c_load(func_name, op, strided_shape, dtype=dtype, arg=arg)
    elif op in ReduceOps:
        return c_reduce(func_name, op, shape, strides[0], dtype=dtype, arg=arg) # type: ignore
    elif op in MovementOps:
        return c_movement(func_name, op, shape, stride=strides[0], permute_dim=arg, dtype=dtype) # type: ignore
    else:
        raise NotImplementedError(f'c_generator for {op.op} not implemented yet.') # type: ignore


class CProgram:
    incudes: List[str] = ['stdio.h', 'stdlib.h', 'time.h']
    kernel_prefix: str = 'void'

    def __init__(self, si: 'ScheduleItem'):
        self.op = si.op.op
        self.shape = si.target.shape
        self.dtype = c.c_float
        self.arg = si.op.arg
        if len(si.srcs) > 0:
            self.srcs = si.srcs
            self.strides = tuple([lb.st.stride for lb in si.srcs])
        else:
            self.srcs = tuple()
            self.strides = si.target.st.stride

        self._write_codepy()

    def __call__(self, output, *inputs):
        self._program(output, *inputs)

    @property
    def program(self):
        return self._program

    def _gen_func_name_args(self) -> Tuple[str, Any]:
        args: Tuple[Type[c._Pointer[c.c_float]], ...] 
        if self.op in BinaryOps:
            str_shape = '_'.join([str(s) for s in self.shape])
            func_name = f'{self.op.name}_{str_shape}_{ctype2str(self.dtype)}'
            args = (c.POINTER(c.c_float), c.POINTER(c.c_float), c.POINTER(c.c_float))
        elif self.op in UnaryOps:
            str_shape = '_'.join([str(s) for s in self.shape])
            func_name = f'{self.op.name}_{str_shape}_{ctype2str(self.dtype)}'
            args = (c.POINTER(c.c_float), c.POINTER(c.c_float))
        elif self.op in LoadOps:
            strided_shape = tuple([sh//st for sh,st in zip(self.shape, self.strides)])
            str_shape = str(math.prod(strided_shape))
            li = self.shape + strided_shape
            str_shape = '_'.join([str(s) for s in li])
            func_name = f'load_{self.op.name}_{str_shape}_{ctype2str(self.dtype)}'
            func_name += '' if self.arg is None else f'_{int(self.arg)}'
            args = (c.POINTER(c.c_float),)
        elif self.op in ReduceOps:
            str_shape = '_'.join([str(s) for s in self.srcs[0].shape])
            func_name = f"reduce_{self.op.name}_{str_shape}_{ctype2str(self.dtype)}_{self.arg}"
            args = (c.POINTER(c.c_float), c.POINTER(c.c_float))
        elif self.op in MovementOps:
            str_shape = '_'.join([str(s) for s in self.srcs[0].shape + self.arg])
            func_name = f"movement_{self.op.name}_{str_shape}_{ctype2str(self.dtype)}"
            args = (c.POINTER(c.c_float), c.POINTER(c.c_float))
        else: 
            raise NotImplementedError(f"op: {self.op} not implemented in _get_func_name_args")
        return func_name, args

    def _write_codepy(self) -> None:
        func_name, args = self._gen_func_name_args()
        func_file_cmp = Path('/tmp') / f'{func_name}.out'
        # check if function was already compiled
        if func_file_cmp.exists() and not ALWAYS_RECOMPILE:
            lib = c.CDLL(str(func_file_cmp))
            lib[func_name].argtypes = args
            self._program = lib[func_name]
            return

        shape = self.shape
        if self.op in ReduceOps or self.op is MovementOps.PERMUTE:
            shape = self.srcs[0].shape

        code = c_generator(func_name, self.op, shape, *self.strides, dtype=self.dtype, arg=self.arg)
        print("="*99)
        print(func_name)
        print(code)

        func_file_code = Path('/tmp') / f'{func_name}.c'
        # save program to file and compile it
        with func_file_code.open('w') as f:
            f.write(str(code))
        subprocess.run(['clang', '-shared', '-O2', func_file_code, '-o', func_file_cmp], check=True)
        lib = c.CDLL(str(func_file_cmp))
        lib[func_name].argtypes = args
        self._program = lib[func_name]
