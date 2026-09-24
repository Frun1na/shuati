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
    # TODO: 1) 搬数据到 shared  2) syncthreads  3) 树形归约 (stride 减半)  4) tid==0 写 partial
    pass


if __name__ == "__main__":
    n = 3000
    x = np.random.rand(n).astype(np.float32)
    blocks = (n + TPB - 1) // TPB
    partial = np.zeros(blocks, dtype=np.float32)
    block_sum[blocks, TPB](x, partial)
    check("block partial sums", partial,
          np.array([x[k * TPB:(k + 1) * TPB].sum() for k in range(blocks)]), rtol=1e-4)
    check("total sum", partial.sum(), x.sum(), rtol=1e-4)
