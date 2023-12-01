from typing import Self


class BaseBuff:
    """
    Defines the datastructure that is used in the Tensor class
    to store arrays.

    This can be for example a numpy array or something compleatly
    custom to enable gpu or even tpu support.

    Subclasses of this class will have to implement the different
    math operations.
    """

    def __add__(self, other: Self) -> Self:
        raise NotImplementedError(
            f'__add__ not implemented for class{self.__class__.__name__}'
        )

    def __sub__(self, other: Self) -> Self:
        raise NotImplementedError(
            f'__sub__ not implemented for class{self.__class__.__name__}'
        )

    def __mul__(self, other: Self) -> Self:
        raise NotImplementedError(
            f'__mul__ not implemented for class{self.__class__.__name__}'
        )
