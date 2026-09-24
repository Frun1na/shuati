"""ex02 grid-stride 循环 + 2D 索引。

知识点:
  - 当数据量 > 线程总数时，每个线程用步长 gridDim.x*blockDim.x 处理多个元素
    (cuda.grid(1) / cuda.gridsize(1) 是 numba 的简写)
  - 2D grid: row = blockIdx.y*blockDim.y+threadIdx.y, col = blockIdx.x*blockDim.x+threadIdx.x
  - 让 threadIdx.x 对应"列"(连续内存)，相邻线程访问相邻地址 => 合并访存 (coalescing)
"""
from common import cuda, np, check


@cuda.jit
def saxpy(alpha, x, y, out):
    # TODO: 用 grid-stride 循环: 从全局索引开始，步长 = gridDim.x * blockDim.x
    pass


@cuda.jit
def matrix_add(A, B, C):
    # TODO: 用 2D 索引算出 row / col (threadIdx.x 对应列)，边界检查后相加
    pass


if __name__ == "__main__":
    n = 5000
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    out = np.zeros_like(x)
    saxpy[4, 64](np.float32(2.0), x, y, out)  # 只有 256 个线程，处理 5000 个元素
    check("saxpy (grid-stride)", out, 2 * x + y)

    M, N = 37, 53
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.zeros_like(A)
    tpb = (16, 16)  # (x, y) => (列, 行)
    bpg = ((N + 15) // 16, (M + 15) // 16)
    matrix_add[bpg, tpb](A, B, C)
    check("matrix_add (2D)", C, A + B)
