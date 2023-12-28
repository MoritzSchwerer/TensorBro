# TensorBro

This library is inspired by [TinyGrad](https://github.com/tinygrad/tinygrad).
If you want to play around with a resonably fast library then you should go to [TinyGrad](https://github.com/tinygrad/tinygrad).

What makes this library different than PyTorch is that it is inherently lazy, meaning
the calculation will only be executed once the results are absolutely needed. This enables fancy optimizations
like reordering operations and merging multiple operations into one.


It is also relatively simple to implement new devices (like TPU/GPUs), since every device will only have
to implement around 26~27 operations.
This is/will be accomplished by easily switching between different backends
like clang, opencl and in the future maybe cuda and so on.


The api is relatively simple and if you know pytorch it is almost the same.
The main part of this library is the Tensor class that handles all the gradient
calculations in the back without you ever needing to see it.


Obviously this is just my personal project, it is fun to play around with and tought
me a lot so far in will keep doing so in the future, but it will never get close to
the speed of PyTorch or other popular libraries.


### Features:
- lazy evaluation
- zero cost reshape/expand operations


### TODOs:
- [x] implement basic Clang backend
- [x] write tests to be able to refactor with confidence
- [x] implement reduce operations (sum, max, min, ... )
- [x] make reduce ops work with expanded(strided) tensors
- [ ] implement matmul
- [ ] Tenor class ops with gradients
- [ ] implement backward grad function
- [ ] think about and implement op merging
- [ ] implement slice
- [ ] make slice zero cost








