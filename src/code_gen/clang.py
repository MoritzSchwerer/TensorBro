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

from ..ops import BinaryOps, LoadOps, OpType, UnaryOps


bops_to_cstyle = {
    BinaryOps.MUL: '*',
    BinaryOps.ADD: '+',
    BinaryOps.SUB: '-',
    BinaryOps.DIV: '/',
    BinaryOps.MOD: '%',
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


def c_binary(function_name: str, op: BinaryOps, shape: Tuple[int, ...], dtype=c.c_float) -> Module:
    includes = []
    if op is BinaryOps.MAX:
        assignment = Assign("out[i]", "fmax(inp1[i], inp2[i])")
        includes.append(Include("math.h"))
    else:
        assignment = Assign('out[i]', f'inp1[i] {bops_to_cstyle[op]} inp2[i]')
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


def c_generator(func_name: str, op: OpType, shape, dtype=c.c_float, arg=None) -> Module:
    if op in BinaryOps:
        return c_binary(func_name, op, shape, dtype) # type: ignore
    elif op in UnaryOps:
        return c_unary(func_name, op, shape, dtype) # type: ignore
    elif op in LoadOps:
        return c_load(func_name, op, shape, dtype, arg)
    else:
        raise NotImplementedError(f'c_generator for {op.op} not implemented yet.') # type: ignore


class CProgram:
    incudes: List[str] = ['stdio.h', 'stdlib.h', 'time.h']
    kernel_prefix: str = 'void'

    def __init__(self, op, shape, dtype, arg=None):
        self.op = op
        self.shape = shape
        self.dtype = dtype
        self.arg = arg

        self._write_codepy()

    def __call__(self, output, *inputs):
        self._program(output, *inputs)

    @property
    def program(self):
        return self._program

    def _gen_func_name_args(self) -> Tuple[str, Any]:
        str_shape = str(math.prod(self.shape))
        args: Tuple[Type[c._Pointer[c.c_float]], ...] 
        if self.op in BinaryOps:
            func_name = f'{self.op.name}_{str_shape}_{ctype2str(self.dtype)}'
            args = (c.POINTER(c.c_float), c.POINTER(c.c_float), c.POINTER(c.c_float))
        elif self.op in UnaryOps:
            func_name = f'{self.op.name}_{str_shape}_{ctype2str(self.dtype)}'
            args = (c.POINTER(c.c_float), )
        elif self.op in LoadOps:
            func_name = f'load_{self.op.name}_{str_shape}_{ctype2str(self.dtype)}'
            func_name += '' if self.arg is None else f'_{int(self.arg)}'
            args = (c.POINTER(c.c_float),)
        return func_name, args

    def _write_codepy(self) -> None:
        func_name, args = self._gen_func_name_args()
        func_file_cmp = Path('/tmp') / f'{func_name}.out'
        # check if function was already compiled
        if func_file_cmp.exists():
            lib = c.CDLL(str(func_file_cmp))
            lib[func_name].argtypes = args
            self._program = lib[func_name]
            return

        code = c_generator(func_name, self.op, self.shape, self.dtype, arg=self.arg)

        func_file_code = Path('/tmp') / f'{func_name}.c'
        # save program to file and compile it
        with func_file_code.open('w') as f:
            f.write(str(code))
        subprocess.run(['clang', '-shared', '-O2', func_file_code, '-o', func_file_cmp], check=True)
        lib = c.CDLL(str(func_file_cmp))
        lib[func_name].argtypes = args
        self._program = lib[func_name]
