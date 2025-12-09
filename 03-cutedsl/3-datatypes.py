import cutlass
import cutlass.cute as cute


@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    print("a(static) =", a)
    cute.printf("a(dynamic) = {}", a)
    b = cutlass.Int32(5)
    print("b(static) =", b)
    cute.printf("b(dynamic) = {}", b)


bar()
bar()
