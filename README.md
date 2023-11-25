# TensorBro
Pytorch inspired Deep Learning framework from scratch for educational purposes.

This project utilizes the Tensor class from PyTorch to ensure efficient gpu
computing, but there is no use of torch.autograd.

The gradient functionality is implemented per module,
meaning each module knows how to compute it's backward pass.
This means we will create a chain of modules that will be executed
in sequence and the for the backwards pass we just reverse the order
and instead of calling the forward method of the module we call backward.

This approach is currently limited because doesn't allow to have 2 branches
converge in the forward pass. This can be fixed by instead of building a module
list we build a module graph, but this is not currently implemented.

### Current Features



### Planed Features
- [ ] forward and backward a linear layer
- [ ] linear layer
