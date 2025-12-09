import cutlass
import cutlass.cute as cute


@cute.kernel
def kernel():

    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        cute.printf("Hello world")


@cute.jit
def hello_world():

    cute.printf("hello world")

    kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


cutlass.cuda.initialize_cuda_context()
print("Running hello_world()...")
hello_world()
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)
from cutlass.cute import KeepPTX, KeepCUBIN

print("Compiling with PTX/CUBIN dumped...")
hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)
print("Running compiled version...")
hello_world_compiled()
