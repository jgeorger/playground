#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
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

struct BenchContext {
    cublasHandle_t handle;
    cublasLtHandle_t ltHandle;
    std::vector<cudaStream_t> streams;
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;
    void* d_workspace;
    size_t workspaceSize;
    cuComplex alpha;
    cuComplex beta;
    FILE* outFile;
    bool verify;
    bool verbose;
};

static void timer_begin(BenchContext& ctx) {
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ctx.ev_start, 0));
}

static float timer_end(BenchContext& ctx) {
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ctx.ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ctx.ev_stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ctx.ev_start, ctx.ev_stop));
    return ms;
}

static void emit_csv(BenchContext& ctx, int batchSize, int n, int m, int k,
                     float overlap, const char* method, float ms) {
    char line[160];
    int len = std::snprintf(line, sizeof(line), "%d,%d,%d,%d,%.2f,%s,%.3f\n",
                            batchSize, n, m, k, overlap, method, ms);
    if (ctx.verbose) {
        std::fwrite(line, 1, len, stdout);
        std::fflush(stdout);
    }
    if (ctx.outFile) {
        std::fwrite(line, 1, len, ctx.outFile);
        std::fflush(ctx.outFile);
    }
}

static void emit_header(BenchContext& ctx) {
    const char* hdr = "batchSize,n,m,k,overlap,method,latency_ms\n";
    if (ctx.verbose) {
        std::fputs(hdr, stdout);
        std::fflush(stdout);
    }
    if (ctx.outFile) {
        std::fputs(hdr, ctx.outFile);
        std::fflush(ctx.outFile);
    }
}

static bool verify_methods(BenchContext& ctx, int batchSize, int n, int m, int k,
                           cuComplex* d_A, cuComplex* d_B, cuComplex* d_C,
                           long long strideA, long long strideB, long long strideC,
                           long long sizeC) {
    const cuComplex& alpha = ctx.alpha;
    const cuComplex& beta = ctx.beta;
    int numStreams = std::min(batchSize, (int)ctx.streams.size());

    // Reference: cublasCgemmStridedBatched
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
    CHECK_CUBLAS(cublasCgemmStridedBatched(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           n, m, k, &alpha,
                                           d_A, n, strideA,
                                           d_B, k, strideB,
                                           &beta,
                                           d_C, n, strideC, batchSize));
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<cuComplex> h_ref(sizeC);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_C, sizeC * sizeof(cuComplex),
                          cudaMemcpyDeviceToHost));

    double ref_norm = 0.0;
    for (long long i = 0; i < sizeC; ++i) {
        double r = h_ref[i].x, im = h_ref[i].y;
        ref_norm += r * r + im * im;
    }
    ref_norm = std::sqrt(ref_norm);

    std::vector<cuComplex> h_test(sizeC);
    auto compare = [&](const char* method, double tol) -> bool {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_test.data(), d_C, sizeC * sizeof(cuComplex),
                              cudaMemcpyDeviceToHost));
        double max_abs = 0.0;
        double diff_norm_sq = 0.0;
        for (long long i = 0; i < sizeC; ++i) {
            double dr = (double)h_test[i].x - h_ref[i].x;
            double di = (double)h_test[i].y - h_ref[i].y;
            double abs_err = std::sqrt(dr * dr + di * di);
            if (abs_err > max_abs) max_abs = abs_err;
            diff_norm_sq += dr * dr + di * di;
        }
        double diff_norm = std::sqrt(diff_norm_sq);
        double rel = (ref_norm > 0.0) ? diff_norm / ref_norm : diff_norm;
        bool ok = (rel <= tol);
        std::fprintf(stderr, "  verify %-30s max_abs=%.3e rel=%.3e  %s\n",
                     method, max_abs, rel, ok ? "PASS" : "FAIL");
        return ok;
    };

    bool all_ok = true;

    // Method 1: cublasCgemm per-stream
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
    for (int b = 0; b < batchSize; ++b) {
        CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.streams[b % numStreams]));
        CHECK_CUBLAS(cublasCgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha,
                                 d_A + b * strideA, n,
                                 d_B + b * strideB, k,
                                 &beta,
                                 d_C + b * strideC, n));
    }
    CHECK_CUBLAS(cublasSetStream(ctx.handle, 0));
    all_ok &= compare("cublasCgemm", 1e-4);

    // Method 2: cublasCgemm3m per-stream
    {
        CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
        CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.streams[0]));
        cublasStatus_t st = cublasCgemm3m(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, m, k, &alpha,
                                          d_A, n, d_B, k, &beta, d_C, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUBLAS(cublasSetStream(ctx.handle, 0));
        if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::fprintf(stderr, "  verify cublasCgemm3m                  SKIP (not supported)\n");
        } else if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "  verify cublasCgemm3m                  SKIP (%s)\n",
                         cublasGetStatusString(st));
        } else {
            CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
            for (int b = 0; b < batchSize; ++b) {
                CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.streams[b % numStreams]));
                CHECK_CUBLAS(cublasCgemm3m(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           n, m, k, &alpha,
                                           d_A + b * strideA, n,
                                           d_B + b * strideB, k,
                                           &beta,
                                           d_C + b * strideC, n));
            }
            CHECK_CUBLAS(cublasSetStream(ctx.handle, 0));
            all_ok &= compare("cublasCgemm3m", 1e-3);
        }
    }

    // Method 3 is the reference — compare against itself (should be exact)
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
    CHECK_CUBLAS(cublasCgemmStridedBatched(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           n, m, k, &alpha,
                                           d_A, n, strideA,
                                           d_B, k, strideB,
                                           &beta,
                                           d_C, n, strideC, batchSize));
    all_ok &= compare("cublasCgemmStridedBatched", 1e-5);

    // Method 4: cublasCgemm3mStridedBatched
    {
        CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
        cublasStatus_t st = cublasCgemm3mStridedBatched(
            ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
            d_A, n, strideA, d_B, k, strideB, &beta,
            d_C, n, strideC, batchSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::fprintf(stderr, "  verify cublasCgemm3mStridedBatched    SKIP (not supported)\n");
        } else if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "  verify cublasCgemm3mStridedBatched    SKIP (%s)\n",
                         cublasGetStatusString(st));
        } else {
            all_ok &= compare("cublasCgemm3mStridedBatched", 1e-3);
        }
    }

    // Method 5: cublasGemmStridedBatchedEx
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha,
        d_A, CUDA_C_32F, n, strideA,
        d_B, CUDA_C_32F, k, strideB,
        &beta,
        d_C, CUDA_C_32F, n, strideC,
        batchSize,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    all_ok &= compare("cublasGemmStridedBatchedEx", 1e-4);

    // Method 6: cublasLtMatmul
    {
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
        int64_t sA = strideA, sB = strideB, sC = strideC;
        cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount32, sizeof(batchCount32));
        cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA, sizeof(sA));
        cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount32, sizeof(batchCount32));
        cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB, sizeof(sB));
        cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount32, sizeof(batchCount32));
        cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC, sizeof(sC));

        CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
        CHECK_CUBLAS(cublasLtMatmul(ctx.ltHandle, matmulDesc,
                                    &alpha,
                                    d_A, layoutA,
                                    d_B, layoutB,
                                    &beta,
                                    d_C, layoutC,
                                    d_C, layoutC,
                                    nullptr,
                                    ctx.d_workspace, ctx.workspaceSize,
                                    0));
        all_ok &= compare("cublasLtMatmul", 1e-4);

        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(matmulDesc);
    }

    return all_ok;
}

static void run_point(BenchContext& ctx, int batchSize, int n, int m, int k,
                      float overlap) {
    long long matA = (long long)n * k;
    long long matB = (long long)k * m;
    long long matC = (long long)n * m;

    long long strideA = (long long)std::llround((double)matA * (1.0 - overlap / 2.0));
    long long strideB = (long long)std::llround((double)matB * (1.0 - overlap / 2.0));
    long long strideC = matC;
    if (strideA < 1) strideA = 1;
    if (strideB < 1) strideB = 1;

    long long strideD = (long long)n * n;  // A * A^H result is n x n per batch

    long long sizeA = strideA * (long long)(batchSize - 1) + matA;
    long long sizeB = strideB * (long long)(batchSize - 1) + matB;
    long long sizeC = strideC * (long long)batchSize;
    long long sizeD = strideD * (long long)batchSize;

    std::fprintf(stderr,
                 "Point: batchSize=%d n=%d m=%d k=%d overlap=%.4f | "
                 "strideA=%lld strideB=%lld | mem A=%.2f B=%.2f C=%.2f D=%.2f MB\n",
                 batchSize, n, m, k, overlap, strideA, strideB,
                 (double)(sizeA * sizeof(cuComplex)) / (1024.0 * 1024.0),
                 (double)(sizeB * sizeof(cuComplex)) / (1024.0 * 1024.0),
                 (double)(sizeC * sizeof(cuComplex)) / (1024.0 * 1024.0),
                 (double)(sizeD * sizeof(cuComplex)) / (1024.0 * 1024.0));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<cuComplex> h_A(sizeA), h_B(sizeB);
    for (long long i = 0; i < sizeA; ++i) h_A[i] = make_cuFloatComplex(dist(rng), dist(rng));
    for (long long i = 0; i < sizeB; ++i) h_B[i] = make_cuFloatComplex(dist(rng), dist(rng));

    cuComplex *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_D, sizeD * sizeof(cuComplex)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
    CHECK_CUDA(cudaMemset(d_D, 0, sizeD * sizeof(cuComplex)));

    if (ctx.verify) {
        verify_methods(ctx, batchSize, n, m, k, d_A, d_B, d_C,
                       strideA, strideB, strideC, sizeC);
        CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(cuComplex)));
    }

    int numStreams = std::min(batchSize, (int)ctx.streams.size());
    const cuComplex& alpha = ctx.alpha;
    const cuComplex& beta = ctx.beta;

    // Method 1: cublasCgemm per-stream
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        timer_begin(ctx);
        for (int b = 0; b < batchSize; ++b) {
            CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.streams[b % numStreams]));
            CHECK_CUBLAS(cublasCgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, k,
                                     &alpha,
                                     d_A + b * strideA, n,
                                     d_A + b * strideA, n,
                                     &beta,
                                     d_D + b * strideD, n));
            CHECK_CUBLAS(cublasCgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &alpha,
                                     d_A + b * strideA, n,
                                     d_B + b * strideB, k,
                                     &beta,
                                     d_C + b * strideC, n));
        }
        float ms = timer_end(ctx);
        if (iter >= WARMUP_ITERS) emit_csv(ctx, batchSize, n, m, k, overlap, "cublasCgemm", ms);
    }
    CHECK_CUBLAS(cublasSetStream(ctx.handle, 0));

    // Method 2: cublasCgemm3m per-stream (with NOT_SUPPORTED probe)
    {
        CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.streams[0]));
        cublasStatus_t probe = cublasCgemm3m(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                              n, m, k, &alpha,
                                              d_A, n, d_B, k, &beta, d_C, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUBLAS(cublasSetStream(ctx.handle, 0));
        if (probe == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::fprintf(stderr, "  cublasCgemm3m not supported -- skipping\n");
        } else if (probe != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "  cublasCgemm3m probe failed: %s -- skipping\n",
                         cublasGetStatusString(probe));
        } else {
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                timer_begin(ctx);
                for (int b = 0; b < batchSize; ++b) {
                    CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.streams[b % numStreams]));
                    CHECK_CUBLAS(cublasCgemm3m(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_C,
                                               n, n, k, &alpha,
                                               d_A + b * strideA, n,
                                               d_A + b * strideA, n,
                                               &beta,
                                               d_D + b * strideD, n));
                    CHECK_CUBLAS(cublasCgemm3m(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, m, k, &alpha,
                                               d_A + b * strideA, n,
                                               d_B + b * strideB, k,
                                               &beta,
                                               d_C + b * strideC, n));
                }
                float ms = timer_end(ctx);
                if (iter >= WARMUP_ITERS) emit_csv(ctx, batchSize, n, m, k, overlap, "cublasCgemm3m", ms);
            }
            CHECK_CUBLAS(cublasSetStream(ctx.handle, 0));
        }
    }

    // Method 3: cublasCgemmStridedBatched
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        timer_begin(ctx);
        CHECK_CUBLAS(cublasCgemmStridedBatched(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_C,
                                               n, n, k, &alpha,
                                               d_A, n, strideA,
                                               d_A, n, strideA,
                                               &beta,
                                               d_D, n, strideD,
                                               batchSize));
        CHECK_CUBLAS(cublasCgemmStridedBatched(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, m, k, &alpha,
                                               d_A, n, strideA,
                                               d_B, k, strideB,
                                               &beta,
                                               d_C, n, strideC,
                                               batchSize));
        float ms = timer_end(ctx);
        if (iter >= WARMUP_ITERS) emit_csv(ctx, batchSize, n, m, k, overlap, "cublasCgemmStridedBatched", ms);
    }

    // Method 4: cublasCgemm3mStridedBatched (with NOT_SUPPORTED probe)
    {
        cublasStatus_t probe = cublasCgemm3mStridedBatched(
            ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
            d_A, n, strideA, d_B, k, strideB, &beta, d_C, n, strideC, batchSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (probe == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::fprintf(stderr, "  cublasCgemm3mStridedBatched not supported -- skipping\n");
        } else if (probe != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "  cublasCgemm3mStridedBatched probe failed: %s -- skipping\n",
                         cublasGetStatusString(probe));
        } else {
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                timer_begin(ctx);
                CHECK_CUBLAS(cublasCgemm3mStridedBatched(
                    ctx.handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, k, &alpha,
                    d_A, n, strideA, d_A, n, strideA, &beta,
                    d_D, n, strideD, batchSize));
                CHECK_CUBLAS(cublasCgemm3mStridedBatched(
                    ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    d_A, n, strideA, d_B, k, strideB, &beta,
                    d_C, n, strideC, batchSize));
                float ms = timer_end(ctx);
                if (iter >= WARMUP_ITERS) emit_csv(ctx, batchSize, n, m, k, overlap,
                                                    "cublasCgemm3mStridedBatched", ms);
            }
        }
    }

    // Method 5: cublasGemmStridedBatchedEx
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        timer_begin(ctx);
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            ctx.handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, k,
            &alpha,
            d_A, CUDA_C_32F, n, strideA,
            d_A, CUDA_C_32F, n, strideA,
            &beta,
            d_D, CUDA_C_32F, n, strideD,
            batchSize,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha,
            d_A, CUDA_C_32F, n, strideA,
            d_B, CUDA_C_32F, k, strideB,
            &beta,
            d_C, CUDA_C_32F, n, strideC,
            batchSize,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
        float ms = timer_end(ctx);
        if (iter >= WARMUP_ITERS) emit_csv(ctx, batchSize, n, m, k, overlap,
                                            "cublasGemmStridedBatchedEx", ms);
    }

    // Method 6: cublasLtMatmul (per-point matmul desc and layouts)
    {
        cublasOperation_t opN = CUBLAS_OP_N, opC = CUBLAS_OP_C;

        cublasLtMatmulDesc_t descAB;  // N, N (for A * B)
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&descAB, CUBLAS_COMPUTE_32F, CUDA_C_32F));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(descAB, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(descAB, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

        cublasLtMatmulDesc_t descAAH;  // N, C (for A * A^H)
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&descAAH, CUBLAS_COMPUTE_32F, CUDA_C_32F));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(descAAH, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(descAAH, CUBLASLT_MATMUL_DESC_TRANSB, &opC, sizeof(opC)));

        cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_C_32F, n, k, n));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_C_32F, k, m, k));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_C_32F, n, m, n));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_C_32F, n, n, n));

        int32_t batchCount32 = batchSize;
        int64_t strideA64 = strideA, strideB64 = strideB, strideC64 = strideC, strideD64 = strideD;
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
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                     &batchCount32, sizeof(batchCount32)));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(layoutD, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &strideD64, sizeof(strideD64)));

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            timer_begin(ctx);
            CHECK_CUBLAS(cublasLtMatmul(ctx.ltHandle, descAAH,
                                        &alpha,
                                        d_A, layoutA,
                                        d_A, layoutA,
                                        &beta,
                                        d_D, layoutD,
                                        d_D, layoutD,
                                        nullptr,
                                        ctx.d_workspace, ctx.workspaceSize,
                                        0));
            CHECK_CUBLAS(cublasLtMatmul(ctx.ltHandle, descAB,
                                        &alpha,
                                        d_A, layoutA,
                                        d_B, layoutB,
                                        &beta,
                                        d_C, layoutC,
                                        d_C, layoutC,
                                        nullptr,
                                        ctx.d_workspace, ctx.workspaceSize,
                                        0));
            float ms = timer_end(ctx);
            if (iter >= WARMUP_ITERS) emit_csv(ctx, batchSize, n, m, k, overlap,
                                                "cublasLtMatmul", ms);
        }

        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatrixLayoutDestroy(layoutD);
        cublasLtMatmulDescDestroy(descAB);
        cublasLtMatmulDescDestroy(descAAH);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
}

struct SweepParams {
    std::vector<int> batches;
    std::vector<int> ns;
    std::vector<int> ms;
    std::vector<int> ks;
    std::vector<float> overlaps;
};

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == delim) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static bool load_sweep(const std::string& path, SweepParams& sp) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Could not open sweep file: %s\n", path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        size_t colon = t.find(':');
        if (colon == std::string::npos) colon = t.find('=');
        if (colon == std::string::npos) continue;
        std::string key = trim(t.substr(0, colon));
        std::string val = trim(t.substr(colon + 1));
        auto tokens = split(val, ',');
        for (auto& tok : tokens) {
            std::string v = trim(tok);
            if (v.empty()) continue;
            if (key == "batch" || key == "batchSize")      sp.batches.push_back(std::atoi(v.c_str()));
            else if (key == "n")                           sp.ns.push_back(std::atoi(v.c_str()));
            else if (key == "m")                           sp.ms.push_back(std::atoi(v.c_str()));
            else if (key == "k")                           sp.ks.push_back(std::atoi(v.c_str()));
            else if (key == "overlap" || key == "overlapFactor") sp.overlaps.push_back((float)std::atof(v.c_str()));
            else std::fprintf(stderr, "Unknown sweep key '%s' -- ignoring\n", key.c_str());
        }
    }
    if (sp.batches.empty() || sp.ns.empty() || sp.ms.empty() ||
        sp.ks.empty() || sp.overlaps.empty()) {
        std::fprintf(stderr, "Sweep file missing one or more of: batch, n, m, k, overlap\n");
        return false;
    }
    return true;
}

static std::string sanitize(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) out.push_back(c);
        else if (c == ' ' || c == '-' || c == '_' || c == '.') out.push_back('_');
    }
    if (out.empty()) out = "unknown_gpu";
    return out;
}

static std::string make_output_filename(const std::string& base) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::string gpu = sanitize(prop.name);
    std::time_t now = std::time(nullptr);
    std::tm tm_local = *std::localtime(&now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_local);
    return base + "_" + ts + "_" + gpu + ".csv";
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s <batchSize> <n> <m> <k> <overlapFactor>\n"
                 "  %s --sweep <file>\n"
                 "\n"
                 "Options:\n"
                 "  --sweep <file>   Read parameter lists from <file>; overrides positional args.\n"
                 "  --out <base>     Output CSV base name (default: cgemm_benchmark).\n"
                 "                   Final file: <base>_YYYYMMDD_HHMMSS_<gpu>.csv\n"
                 "  --verify         Compare each method's output against cublasCgemmStridedBatched\n"
                 "                   (reports max absolute and relative error per point).\n"
                 "  --verbose, -v    Also print CSV lines to stdout (default: file only).\n"
                 "\n"
                 "Sweep file format (one key per line):\n"
                 "  batch:   100, 1000, 10000\n"
                 "  n:       8, 16, 32\n"
                 "  m:       8, 16, 32\n"
                 "  k:       8, 16, 32\n"
                 "  overlap: 0.0, 0.5, 1.0\n",
                 prog, prog);
}

int main(int argc, char** argv) {
    std::string sweep_path;
    std::string out_base = "cgemm_benchmark";
    bool verify = false;
    bool verbose = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sweep" && i + 1 < argc) {
            sweep_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_base = argv[++i];
        } else if (a == "--verify") {
            verify = true;
        } else if (a == "--verbose" || a == "-v") {
            verbose = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            positional.push_back(a);
        }
    }

    SweepParams sp;
    if (!sweep_path.empty()) {
        if (!load_sweep(sweep_path, sp)) return 1;
    } else {
        if (positional.size() != 5) {
            print_usage(argv[0]);
            return 1;
        }
        sp.batches.push_back(std::atoi(positional[0].c_str()));
        sp.ns.push_back(std::atoi(positional[1].c_str()));
        sp.ms.push_back(std::atoi(positional[2].c_str()));
        sp.ks.push_back(std::atoi(positional[3].c_str()));
        sp.overlaps.push_back((float)std::atof(positional[4].c_str()));
    }

    for (int v : sp.batches) if (v <= 0) { std::fprintf(stderr, "batch must be positive\n"); return 1; }
    for (int v : sp.ns)      if (v <= 0) { std::fprintf(stderr, "n must be positive\n"); return 1; }
    for (int v : sp.ms)      if (v <= 0) { std::fprintf(stderr, "m must be positive\n"); return 1; }
    for (int v : sp.ks)      if (v <= 0) { std::fprintf(stderr, "k must be positive\n"); return 1; }
    for (float v : sp.overlaps) if (v < 0.0f || v > 1.0f) {
        std::fprintf(stderr, "overlap must be in [0, 1]\n"); return 1;
    }

    size_t total = sp.batches.size() * sp.ns.size() * sp.ms.size() *
                   sp.ks.size() * sp.overlaps.size();
    std::fprintf(stderr, "Sweep: %zu combinations (batch=%zu, n=%zu, m=%zu, k=%zu, overlap=%zu)\n",
                 total, sp.batches.size(), sp.ns.size(), sp.ms.size(),
                 sp.ks.size(), sp.overlaps.size());

    std::string out_path = make_output_filename(out_base);
    FILE* outFile = std::fopen(out_path.c_str(), "w");
    if (!outFile) {
        std::fprintf(stderr, "Could not open output file: %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "Writing results to: %s\n", out_path.c_str());

    BenchContext ctx;
    CHECK_CUBLAS(cublasCreate(&ctx.handle));
    CHECK_CUBLAS(cublasLtCreate(&ctx.ltHandle));
    ctx.streams.resize(MAX_STREAMS);
    for (int i = 0; i < MAX_STREAMS; ++i) CHECK_CUDA(cudaStreamCreate(&ctx.streams[i]));
    CHECK_CUDA(cudaEventCreate(&ctx.ev_start));
    CHECK_CUDA(cudaEventCreate(&ctx.ev_stop));
    ctx.workspaceSize = 4 * 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&ctx.d_workspace, ctx.workspaceSize));
    ctx.alpha = make_cuFloatComplex(1.0f, 0.0f);
    ctx.beta = make_cuFloatComplex(0.0f, 0.0f);
    ctx.outFile = outFile;
    ctx.verify = verify;
    ctx.verbose = verbose;

    emit_header(ctx);

    size_t idx = 0;
    for (int batch : sp.batches)
        for (int n : sp.ns)
            for (int m : sp.ms)
                for (int k : sp.ks)
                    for (float o : sp.overlaps) {
                        ++idx;
                        std::fprintf(stderr, "[%zu/%zu] ", idx, total);
                        run_point(ctx, batch, n, m, k, o);
                    }

    std::fclose(outFile);
    cudaFree(ctx.d_workspace);
    cudaEventDestroy(ctx.ev_start);
    cudaEventDestroy(ctx.ev_stop);
    for (auto s : ctx.streams) cudaStreamDestroy(s);
    cublasLtDestroy(ctx.ltHandle);
    cublasDestroy(ctx.handle);
    return 0;
}
