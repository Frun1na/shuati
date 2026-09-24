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
    # TODO: 对 data[i] 所在的 bin 做 cuda.atomic.add
    pass


@cuda.jit
def histogram_shared(data, hist):
    # TODO: shared 私有直方图: 清零 -> sync -> 原子累加到 shared -> sync -> 原子合并到全局
    pass


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
