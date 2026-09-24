"""ex03 并行归约求和：shared memory + __syncthreads()。

知识点:
  - cuda.shared.array: 同一个 block 内线程共享、比全局内存快得多
  - cuda.syncthreads(): block 内屏障，写完 shared 之后、读别人写的值之前必须同步
  - 树形归约: stride 从 blockDim/2 每次减半，O(log n) 步
  - 每个 block 产生一个部分和，再用 atomic 或第二个 kernel 合并
"""
from common import cuda, np, check
from numba import float32

TPB = 128  # shared array 大小必须是编译期常量


@cuda.jit
def block_sum(x, partial):
    sdata = cuda.shared.array(TPB, dtype=float32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + tid

    # 1. 每个线程搬一个元素到 shared memory（越界补 0）
    sdata[tid] = x[i] if i < x.size else 0.0
    cuda.syncthreads()

    # 2. 树形归约
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()  # 注意：必须在 if 外面，所有线程都要到达
        s //= 2

    # 3. 0 号线程写出本 block 的结果
    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]


if __name__ == "__main__":
    n = 3000
    x = np.random.rand(n).astype(np.float32)
    blocks = (n + TPB - 1) // TPB
    partial = np.zeros(blocks, dtype=np.float32)
    block_sum[blocks, TPB](x, partial)
    check("block partial sums", partial,
          np.array([x[k * TPB:(k + 1) * TPB].sum() for k in range(blocks)]), rtol=1e-4)
    check("total sum", partial.sum(), x.sum(), rtol=1e-4)
