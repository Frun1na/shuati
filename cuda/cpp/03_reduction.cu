// ex03 的 CUDA C++ 版本：shared memory 树形归约
#include <cstdio>
#include <cuda_runtime.h>

constexpr int TPB = 128;

__global__ void block_sum(const float* x, float* partial, int n) {
    __shared__ float sdata[TPB];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// 进阶：最后 32 个线程在同一个 warp 内，可以用 warp shuffle 代替 shared memory
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

int main() {
    const int n = 3000, blocks = (n + TPB - 1) / TPB;
    float *x, *partial;
    // 统一内存 (Unified Memory)：host 和 device 都能访问，省掉 cudaMemcpy
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&partial, blocks * sizeof(float));
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    block_sum<<<blocks, TPB>>>(x, partial, n);
    cudaDeviceSynchronize();

    float total = 0;
    for (int b = 0; b < blocks; ++b) total += partial[b];
    printf("sum = %.1f (expected %d) %s\n", total, n, total == n ? "PASS" : "FAIL");
    cudaFree(x); cudaFree(partial);
}
