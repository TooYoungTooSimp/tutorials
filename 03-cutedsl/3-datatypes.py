import cutlass
import cutlass.cute as cute

@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    print("a(static) =", a)  # prints `a(static) = ?`
    cute.printf("a(dynamic) = {}", a)  # prints `a(dynamic) = 3.140000`

    b = cutlass.Int32(5)
    print("b(static) =", b)  # prints `b(static) = 5`
    cute.printf("b(dynamic) = {}", b)  # prints `b(dynamic) = 5`


bar()
bar()
