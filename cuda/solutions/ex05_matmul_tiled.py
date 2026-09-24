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
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        acc = 0.0
        for k in range(A.shape[1]):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc


@cuda.jit
def matmul_tiled(A, B, C):
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)
    tx = cuda.threadIdx.x  # 列方向
    ty = cuda.threadIdx.y  # 行方向
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx
    K = A.shape[1]

    acc = float32(0.0)
    for t in range((K + TILE - 1) // TILE):
        # 协作加载，越界补 0
        k_a = t * TILE + tx
        k_b = t * TILE + ty
        sA[ty, tx] = A[row, k_a] if (row < A.shape[0] and k_a < K) else 0.0
        sB[ty, tx] = B[k_b, col] if (k_b < K and col < B.shape[1]) else 0.0
        cuda.syncthreads()

        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]
        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc


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
