"""ex04 直方图：原子操作与竞争条件。

知识点:
  - 多个线程同时 hist[k] += 1 会丢失更新 (read-modify-write 竞争)
  - cuda.atomic.add 保证原子性
  - 优化: 先在 shared memory 里做 block 私有直方图，最后再原子加到全局，减少全局原子冲突
"""
from common import cuda, np, check
from numba import int32

NBINS = 16
TPB = 64


@cuda.jit
def histogram_global(data, hist):
    i = cuda.grid(1)
    if i < data.size:
        cuda.atomic.add(hist, data[i], 1)


@cuda.jit
def histogram_shared(data, hist):
    local = cuda.shared.array(NBINS, dtype=int32)
    tid = cuda.threadIdx.x
    # 清零 (线程数可能多于或少于 bin 数，用循环)
    for k in range(tid, NBINS, cuda.blockDim.x):
        local[k] = 0
    cuda.syncthreads()

    i = cuda.grid(1)
    if i < data.size:
        cuda.atomic.add(local, data[i], 1)
    cuda.syncthreads()

    for k in range(tid, NBINS, cuda.blockDim.x):
        cuda.atomic.add(hist, k, local[k])


if __name__ == "__main__":
    n = 4000
    data = np.random.randint(0, NBINS, n).astype(np.int32)
    expected = np.bincount(data, minlength=NBINS)
    blocks = (n + TPB - 1) // TPB

    h1 = np.zeros(NBINS, dtype=np.int32)
    histogram_global[blocks, TPB](data, h1)
    check("histogram_global", h1, expected)

    h2 = np.zeros(NBINS, dtype=np.int32)
    histogram_shared[blocks, TPB](data, h2)
    check("histogram_shared", h2, expected)
