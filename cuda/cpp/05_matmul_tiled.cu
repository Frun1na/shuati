// ex05 的 CUDA C++ 版本：分块矩阵乘法，C[M,N] = A[M,K] * B[K,N]，行主序
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ka = t * TILE + tx, kb = t * TILE + ty;
        sA[ty][tx] = (row < M && ka < K) ? A[row * K + ka] : 0.0f;
        sB[ty][tx] = (kb < K && col < N) ? B[kb * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

int main() {
    const int M = 100, K = 70, N = 90;
    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));
    for (int i = 0; i < M * K; ++i) A[i] = (i % 7) * 0.1f;
    for (int i = 0; i < K * N; ++i) B[i] = (i % 5) * 0.2f;

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;  // 用 CUDA event 计时
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_tiled<<<grid, block>>>(A, B, C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    int errors = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float ref = 0;
            for (int k = 0; k < K; ++k) ref += A[i * K + k] * B[k * N + j];
            errors += fabsf(ref - C[i * N + j]) > 1e-3f;
        }
    printf("%s, kernel time %.3f ms\n", errors ? "FAIL" : "PASS", ms);
    cudaFree(A); cudaFree(B); cudaFree(C);
}
