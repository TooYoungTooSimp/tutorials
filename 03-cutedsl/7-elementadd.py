import torch
from functools import partial
from typing import List
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import os

os.environ["PYTHONUNBUFFERED"] = "1"


@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    cute.printf("Thread ID: {}", thread_idx)
    print("sta Thread ID: {}", thread_idx)
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    gC[mi, ni] = a_val + b_val


@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 1024
    m, n = mA.shape
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    print("launching kernel...")
    cute.printf("launching kernel... (rt)\n")
    kernel(
        grid=(1, 1, 1),
        block=(2, 1, 1),
    )


M, N = 16384, 8192
a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
num_elements = sum([a.numel(), b.numel(), c.numel()])
a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)
naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)
print(a[0, 0] + b[0, 0], " vs ", c[0, 0])
print(a[0, 1] + b[0, 1], " vs ", c[0, 1])
naive_elementwise_add_(a_, b_, c_)
naive_elementwise_add_(a_, b_, c_)
print(a[0, 0] + b[0, 0], " vs ", c[0, 0])
print(a[0, 1] + b[0, 1], " vs ", c[0, 1])


def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )
    dtype = a_.element_type
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")
