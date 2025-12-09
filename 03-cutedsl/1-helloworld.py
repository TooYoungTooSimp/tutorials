import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if tidx == 0:
        cute.printf("Hello world")

@cute.jit
def hello_world():
    # Print hello world from host code
    cute.printf("hello world")

    # Launch kernel
    kernel().launch(
        grid=(1, 1, 1),  # Single thread block
        block=(32, 1, 1),  # One warp (32 threads) per thread block
    )


# Initialize CUDA context for launching a kernel with error checking
# We make context initialization explicit to allow users to control the context creation
# and avoid potential issues with multiple contexts
cutlass.cuda.initialize_cuda_context()

# Method 1: Just-In-Time (JIT) compilation - compiles and runs the code immediately
print("Running hello_world()...")
hello_world()

# Method 2: Compile first (useful if you want to run the same code multiple times)
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)

# Dump PTX/CUBIN files while compiling
from cutlass.cute import KeepPTX, KeepCUBIN

print("Compiling with PTX/CUBIN dumped...")
# Alternatively, compile with string based options like
# cute.compile(hello_world, options="--keep-ptx --keep-cubin") would also work.
hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)

# Run the pre-compiled version
print("Running compiled version...")
hello_world_compiled()
