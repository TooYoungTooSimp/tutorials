import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)


import torch
from cutlass.torch import dtype as torch_dtype
import cutlass.cute.runtime as cute_rt

a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())


# create_tensor_from_ptr(ptr_a)
@cute.jit
def print_tensor_coord(a: cute.Tensor):
    coord_tensor = cute.make_identity_tensor(a.layout.shape)
    print(coord_tensor)
    cute.print_tensor(coord_tensor)


a = torch.randn(8, 4, dtype=torch_dtype(cutlass.Float32))
print_tensor_coord(from_dlpack(a))
