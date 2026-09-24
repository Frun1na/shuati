"""ex05 分块矩阵乘法 (tiled GEMM)：CUDA 优化的经典入门。

知识点:
  - 朴素版: 每个线程算 C 的一个元素，读 A 的一行和 B 的一列 => 全局内存读取 2K 次
  - 分块版: block 协作把 A、B 的 TILE x TILE 子块搬进 shared memory，
    块内每个数被复用 TILE 次，全局访存量降为 1/TILE
  - 每轮 tile: 加载 -> syncthreads -> 计算 -> syncthreads(防止下一轮覆盖还在用的数据)
"""
from common import cuda, np, check
from numba import float32

TILE = 8


@cuda.jit
def matmul_naive(A, B, C):
    # numba 的 cuda.grid(2) 返回 (x, y)。这里让 x 对应行、y 对应列（和 launch 配置一致）。
    # 思考题: 这样相邻线程 (x 相邻) 访问的是 C 的相邻"行"，访存合并吗？tiled 版是怎么安排的？
    # TODO: 每个线程算 C[row, col] = sum_k A[row,k]*B[k,col]
    pass


@cuda.jit
def matmul_tiled(A, B, C):
    # TODO: 按 TILE 分块循环: 协作加载 sA/sB (越界补0) -> sync -> 累加 -> sync
    pass


if __name__ == "__main__":
    M, K, N = 20, 27, 18
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    C1 = np.zeros((M, N), dtype=np.float32)
    matmul_naive[((M + 7) // 8, (N + 7) // 8), (8, 8)](A, B, C1)
    check("matmul_naive", C1, A @ B, rtol=1e-4)

    C2 = np.zeros((M, N), dtype=np.float32)
    matmul_tiled[((N + TILE - 1) // TILE, (M + TILE - 1) // TILE), (TILE, TILE)](A, B, C2)
    check("matmul_tiled", C2, A @ B, rtol=1e-4)
