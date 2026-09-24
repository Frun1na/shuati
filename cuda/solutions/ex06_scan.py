"""ex06 前缀和 (inclusive scan)，Hillis-Steele 算法，单 block 版本。

知识点:
  - out[i] = x[0] + ... + x[i]
  - 每步 offset 翻倍: 若 tid >= offset, val += s[tid - offset]
  - 陷阱: 同一步里读和写同一个 shared 数组会产生竞争，
    必须 "先读到寄存器 -> sync -> 再写 -> sync"（或用双缓冲）
"""
from common import cuda, np, check
from numba import float32

N = 256  # 单 block 处理 N 个元素


@cuda.jit
def scan_block(x, out):
    s = cuda.shared.array(N, dtype=float32)
    tid = cuda.threadIdx.x
    s[tid] = x[tid]
    cuda.syncthreads()

    offset = 1
    while offset < N:
        val = s[tid]
        if tid >= offset:
            val += s[tid - offset]
        cuda.syncthreads()  # 所有人读完后才能写
        s[tid] = val
        cuda.syncthreads()  # 所有人写完后才能进入下一轮读
        offset *= 2

    out[tid] = s[tid]


if __name__ == "__main__":
    x = np.random.rand(N).astype(np.float32)
    out = np.zeros_like(x)
    scan_block[1, N](x, out)
    check("inclusive scan", out, np.cumsum(x), rtol=1e-4)
