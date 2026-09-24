"""ex01 向量加法：理解 thread / block / grid 与全局索引。

知识点:
  - 一个 kernel 由 grid 个 block 执行，每个 block 有 blockDim 个 thread
  - 全局索引 i = blockIdx.x * blockDim.x + threadIdx.x
  - 线程总数一般会向上取整，所以要做边界检查 if i < n
"""
from common import cuda, np, check


@cuda.jit
def vector_add(a, b, c):
    # TODO: 计算全局索引 i，做边界检查后写 c[i]
    pass


if __name__ == "__main__":
    n = 1000  # 故意不是 256 的倍数
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c = np.zeros_like(a)
    threads = 256
    blocks = (n + threads - 1) // threads  # 向上取整
    vector_add[blocks, threads](a, b, c)
    check("vector_add", c, a + b)
