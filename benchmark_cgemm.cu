#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cublasGetStatusString(st__));               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int NUM_ITERS = 10;
static constexpr int WARMUP_ITERS = 1;
static constexpr int MAX_STREAMS = 32;

struct Timer {
    cudaEvent_t start;
    cudaEvent_t stop;
    Timer() {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
    }
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start, 0));
    }
    float end_ms() {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

static void emit_csv(int batchSize, int n, int m, int k, float overlap,
                     const char* method, float ms) {
    std::printf("%d,%d,%d,%d,%.4f,%s,%.6f\n", batchSize, n, m, k, overlap,
                method, ms);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s <batchSize> <n> <m> <k> <overlapFactor>\n"
                     "  batchSize     : number of matrices in the batch\n"
                     "  n, m, k       : C(n x m) = A(n x k) * B(k x m)\n"
                     "  overlapFactor : 0.0 (no overlap) .. 1.0 (50%% overlap)\n",
                     argv[0]);
        return 1;
    }

    int batchSize = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int m = std::atoi(argv[3]);
    int k = std::atoi(argv[4]);
    float overlap = std::atof(argv[5]);

    if (batchSize <= 0 || n <= 0 || m <= 0 || k <= 0) {
        std::fprintf(stderr, "batchSize, n, m, k must all be positive\n");
        return 1;
    }
    if (overlap < 0.0f || overlap > 1.0f) {
        std::fprintf(stderr, "overlapFactor must be in [0, 1]\n");
        return 1;
    }

    long long matA = (long long)n * k;
    long long matB = (long long)k * m;
    long long matC = (long long)n * m;

    long long strideA = (long long)std::llround((double)matA * (1.0 - overlap / 2.0));
    long long strideB = (long long)std::llround((double)matB * (1.0 - overlap / 2.0));
    long long strideC = matC;
    if (strideA < 1) strideA = 1;
    if (strideB < 1) strideB = 1;

    long long sizeA = strideA * (long long)(batchSize - 1) + matA;
    long long sizeB = strideB * (long long)(batchSize - 1) + matB;
    long long sizeC = strideC * (long long)batchSize;

    std::fprintf(stderr,
                 "Config: batchSize=%d n=%d m=%d k=%d overlap=%.4f\n"
                 "  strideA=%lld strideB=%lld strideC=%lld\n"
                 "  sizeA=%lld sizeB=%lld sizeC=%lld elements\n"
                 "  memory: A=%.2f MB, B=%.2f MB, C=%.2f MB\n",
                 batchSize, n, m, k, overlap, strideA, strideB, strideC,
                 sizeA, sizeB, sizeC,
                 (double)(sizeA * sizeof(cuComplex)) / (1024.0 * 1024.0),
                 (double)(sizeB * sizeof(cuComplex)) / (1024.0 * 1024.0),
                 (double)(sizeC * sizeof(cuComplex)) / (1024.0 * 1024.0));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<cuComplex> h_A(sizeA), h_B(sizeB);
    for (long long i = 0; i < sizeA; ++i) h_A[i] = make_cuFloatComplex(dist(rng), dist(rng));
    for (long long i = 0; i < sizeB; ++i) h_B[i] = make_cuFloatComplex(dist(rng), dist(rng));

    cuComplex *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(cuComplex)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int numStreams = std::min(batchSize, MAX_STREAMS);
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    const cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuFloatComplex(0.0f, 0.0f);

    Timer timer;

    std::printf("batchSize,n,m,k,overlap,method,latency_ms\n");
    std::fflush(stdout);

    // Method 1: cublasCgemm per-stream
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        timer.begin();
        for (int b = 0; b < batchSize; ++b) {
            CHECK_CUBLAS(cublasSetStream(handle, streams[b % numStreams]));
            CHECK_CUBLAS(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &alpha,
                                     d_A + b * strideA, n,
                                     d_B + b * strideB, k,
                                     &beta,
                                     d_C + b * strideC, n));
        }
        float ms = timer.end_ms();
        if (iter >= WARMUP_ITERS) emit_csv(batchSize, n, m, k, overlap, "cublasCgemm", ms);
    }
    CHECK_CUBLAS(cublasSetStream(handle, 0));

    // Method 2: cublasCgemm3m per-stream
    {
        CHECK_CUBLAS(cublasSetStream(handle, streams[0]));
        cublasStatus_t probe = cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                              n, m, k, &alpha,
                                              d_A, n, d_B, k, &beta, d_C, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUBLAS(cublasSetStream(handle, 0));
        if (probe == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::fprintf(stderr, "cublasCgemm3m not supported for these dimensions -- skipping\n");
        } else if (probe != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasCgemm3m probe failed: %s -- skipping\n",
                         cublasGetStatusString(probe));
        } else {
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                timer.begin();
                for (int b = 0; b < batchSize; ++b) {
                    CHECK_CUBLAS(cublasSetStream(handle, streams[b % numStreams]));
                    CHECK_CUBLAS(cublasCgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, m, k, &alpha,
                                               d_A + b * strideA, n,
                                               d_B + b * strideB, k,
                                               &beta,
                                               d_C + b * strideC, n));
                }
                float ms = timer.end_ms();
                if (iter >= WARMUP_ITERS) emit_csv(batchSize, n, m, k, overlap, "cublasCgemm3m", ms);
            }
            CHECK_CUBLAS(cublasSetStream(handle, 0));
        }
    }

    // Method 3: cublasCgemmStridedBatched
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        timer.begin();
        CHECK_CUBLAS(cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, m, k, &alpha,
                                               d_A, n, strideA,
                                               d_B, k, strideB,
                                               &beta,
                                               d_C, n, strideC,
                                               batchSize));
        float ms = timer.end_ms();
        if (iter >= WARMUP_ITERS) emit_csv(batchSize, n, m, k, overlap, "cublasCgemmStridedBatched", ms);
    }

    // Method 4: cublasCgemm3mStridedBatched
    {
        cublasStatus_t probe = cublasCgemm3mStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
            d_A, n, strideA, d_B, k, strideB, &beta, d_C, n, strideC, batchSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (probe == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::fprintf(stderr, "cublasCgemm3mStridedBatched not supported -- skipping\n");
        } else if (probe != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasCgemm3mStridedBatched probe failed: %s -- skipping\n",
                         cublasGetStatusString(probe));
        } else {
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                timer.begin();
                CHECK_CUBLAS(cublasCgemm3mStridedBatched(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    d_A, n, strideA, d_B, k, strideB, &beta,
                    d_C, n, strideC, batchSize));
                float ms = timer.end_ms();
                if (iter >= WARMUP_ITERS) emit_csv(batchSize, n, m, k, overlap,
                                                    "cublasCgemm3mStridedBatched", ms);
            }
        }
    }

    // Method 5: cublasGemmStridedBatchedEx
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        timer.begin();
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha,
            d_A, CUDA_C_32F, n, strideA,
            d_B, CUDA_C_32F, k, strideB,
            &beta,
            d_C, CUDA_C_32F, n, strideC,
            batchSize,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
        float ms = timer.end_ms();
        if (iter >= WARMUP_ITERS) emit_csv(batchSize, n, m, k, overlap,
                                            "cublasGemmStridedBatchedEx", ms);
    }

    // Method 6: cublasLtMatmul
    {
        cublasLtHandle_t ltHandle;
        CHECK_CUBLAS(cublasLtCreate(&ltHandle));

        cublasLtMatmulDesc_t matmulDesc;
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_C_32F));
        cublasOperation_t opN = CUBLAS_OP_N;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &opN, sizeof(opN)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &opN, sizeof(opN)));

        cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_C_32F, n, k, n));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_C_32F, k, m, k));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_C_32F, n, m, n));

        int32_t batchCount32 = batchSize;
        int64_t strideA64 = strideA, strideB64 = strideB, strideC64 = strideC;
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                     &batchCount32, sizeof(batchCount32)));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &strideA64, sizeof(strideA64)));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                     &batchCount32, sizeof(batchCount32)));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &strideB64, sizeof(strideB64)));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                     &batchCount32, sizeof(batchCount32)));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &strideC64, sizeof(strideC64)));

        size_t workspaceSize = 4 * 1024 * 1024;
        void* d_workspace = nullptr;
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            timer.begin();
            CHECK_CUBLAS(cublasLtMatmul(ltHandle, matmulDesc,
                                        &alpha,
                                        d_A, layoutA,
                                        d_B, layoutB,
                                        &beta,
                                        d_C, layoutC,
                                        d_C, layoutC,
                                        nullptr,
                                        d_workspace, workspaceSize,
                                        0));
            float ms = timer.end_ms();
            if (iter >= WARMUP_ITERS) emit_csv(batchSize, n, m, k, overlap,
                                                "cublasLtMatmul", ms);
        }

        CHECK_CUDA(cudaFree(d_workspace));
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtDestroy(ltHandle);
    }

    for (int i = 0; i < numStreams; ++i) cudaStreamDestroy(streams[i]);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
