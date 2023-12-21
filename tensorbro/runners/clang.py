import ctypes as c


class _CAllocator:
    @staticmethod
    def alloc(dtype, size):
        return (dtype * size)()

    def free(self, pointer):
        c.free(pointer)


CAllocator = _CAllocator()
