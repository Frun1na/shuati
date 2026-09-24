# CUDA 刷题练习

这个环境**没有 GPU**，所以分两条线来学：

| 目录 | 内容 | 怎么跑 |
|---|---|---|
| `exercises/` | 练习题（kernel 函数体是 TODO，需要你写） | `./run.sh ex01` |
| `solutions/` | 参考答案 | `./run.sh ex01 --sol` |
| `cpp/` | 对应的 **CUDA C++** 写法（真实项目用的形式） | `nvcc` 编译 / 看 PTX |

Python 练习用的是 [Numba](https://numba.readthedocs.io/en/stable/cuda/index.html) 的 CUDA 语法，
编程模型和 CUDA C++ 一一对应（`cuda.threadIdx.x` ↔ `threadIdx.x`、
`cuda.syncthreads()` ↔ `__syncthreads()`、`cuda.shared.array` ↔ `__shared__`、`kernel[grid, block](...)` ↔ `kernel<<<grid, block>>>(...)`）。
`NUMBA_ENABLE_CUDASIM=1` 会让 kernel 在 CPU 上**模拟**执行：结果正确性可以验证，但速度没有参考意义。

## 环境准备（新开的 session 需要重新装）

```bash
pip install numba                                   # 必需：跑练习
apt-get install -y nvidia-cuda-toolkit              # 可选：nvcc，编译 .cu / 看 PTX
```

## 题目列表

| 题号 | 主题 | 核心知识点 |
|---|---|---|
| ex01 | 向量加法 | thread/block/grid、全局索引、边界检查 |
| ex02 | SAXPY + 矩阵加法 | grid-stride 循环、2D 索引、合并访存 |
| ex03 | 归约求和 | shared memory、`__syncthreads`、树形归约 |
| ex04 | 直方图 | 竞争条件、原子操作、shared 私有化 |
| ex05 | 矩阵乘法 | 分块 (tiling)、数据复用 |
| ex06 | 前缀和 | Hillis-Steele scan、读写竞争 |

## CUDA C++ 部分

```bash
cd cpp
nvcc -arch=sm_80 -o build/vec 01_vector_add.cu        # 编译（能编过，但这里没 GPU 跑不了）
nvcc -arch=sm_80 -ptx 03_reduction.cu -o -            # 看生成的 PTX 虚拟汇编
nvcc -arch=sm_80 --resource-usage -c 05_matmul_tiled.cu  # 看寄存器 / shared memory 用量
```

## 建议的学习方式

1. 先读每道题文件顶部的"知识点"，再填 TODO，`./run.sh exNN` 直到 PASS
2. 对照 `solutions/` 和 `cpp/` 里的 C++ 版本
3. 做"破坏实验"：把 ex03 的 `syncthreads()` 删掉、把 ex04 的 atomic 换成 `+=`，看结果怎么错
