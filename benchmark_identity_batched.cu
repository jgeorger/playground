// Latency benchmark for identity_strided_batched vs. a naive kernel
// (one thread per element, 64-bit div/mod) and vs. memset + diagonal kernel.
//
// Usage: ./build/release/benchmark_identity_batched [n batch [lda stride]] [--print]
//   With no size arguments a default sweep is run.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "identity_batched.h"

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t e_ = (expr);                                               \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr,  \
                         cudaGetErrorString(e_));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__ void naive_identity(cuComplex* A, long long total, int n, int lda, long long stride) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long long nn = static_cast<long long>(n) * n;
    const long long b = t / nn;
    const long long i = t % nn;
    const long long col = i / n;
    const long long row = i % n;
    A[b * stride + col * lda + row] = make_cuComplex(row == col ? 1.0f : 0.0f, 0.0f);
}

__global__ void set_diagonal(cuComplex* A, long long total, int n, int lda, long long stride) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long long b = t / n;
    const long long d = t % n;
    A[b * stride + d * lda + d] = make_cuComplex(1.0f, 0.0f);
}

struct Case {
    int n, batch, lda;
    long long stride;
};

template <typename Launch>
float time_us(cudaStream_t s, Launch launch) {
    constexpr int kWarmup = 50, kIters = 1000;
    for (int i = 0; i < kWarmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < kIters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms * 1000.0f / kIters;
}

void run_case(const Case& c, cudaStream_t s, bool print) {
    const size_t span = static_cast<size_t>(c.batch - 1) * c.stride +
                        static_cast<size_t>(c.n - 1) * c.lda + c.n;
    cuComplex* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, span * sizeof(cuComplex)));

    const long long total = static_cast<long long>(c.batch) * c.n * c.n;
    const long long diag_total = static_cast<long long>(c.batch) * c.n;
    const int bs = 256;

    const float t_opt = time_us(s, [&] {
        CUDA_CHECK(identity_strided_batched(d, c.n, c.lda, c.stride, c.batch, s));
    });
    const float t_naive = time_us(s, [&] {
        naive_identity<<<(total + bs - 1) / bs, bs, 0, s>>>(d, total, c.n, c.lda, c.stride);
    });
    const bool packed = (c.lda == c.n) && (c.stride == static_cast<long long>(c.n) * c.n);
    float t_memset = -1.0f;
    if (packed) {
        t_memset = time_us(s, [&] {
            CUDA_CHECK(cudaMemsetAsync(d, 0, span * sizeof(cuComplex), s));
            set_diagonal<<<(diag_total + bs - 1) / bs, bs, 0, s>>>(d, diag_total, c.n, c.lda, c.stride);
        });
    }

    // Leave the optimized result in place for printing.
    CUDA_CHECK(identity_strided_batched(d, c.n, c.lda, c.stride, c.batch, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    const double mb = total * sizeof(cuComplex) / 1e6;
    std::printf("%6d %8d %6d %12lld %10.2f | %9.2f %9.2f ", c.n, c.batch, c.lda, c.stride, mb,
                t_opt, t_naive);
    if (t_memset >= 0) std::printf("%9.2f", t_memset); else std::printf("%9s", "n/a");
    std::printf(" | %7.1f\n", mb / 1e3 / (t_opt * 1e-6));

    if (print) CUDA_CHECK(print_strided_batched(d, c.n, c.lda, c.stride, c.batch, s));
    CUDA_CHECK(cudaFree(d));
}

int main(int argc, char** argv) {
    bool print = false;
    std::vector<long long> nums;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print") == 0) print = true;
        else nums.push_back(std::atoll(argv[i]));
    }

    std::vector<Case> cases;
    if (nums.size() == 2 || nums.size() == 4) {
        Case c{static_cast<int>(nums[0]), static_cast<int>(nums[1]), static_cast<int>(nums[0]),
               nums[0] * nums[0]};
        if (nums.size() == 4) { c.lda = static_cast<int>(nums[2]); c.stride = nums[3]; }
        cases.push_back(c);
    } else if (nums.empty()) {
        for (int n : {4, 8, 16, 32, 64, 128})
            for (int batch : {1, 64, 1024, 16384})
                cases.push_back({n, batch, n, static_cast<long long>(n) * n});
        cases.push_back({30, 1024, 32, 32 * 30 + 64});
        cases.push_back({1024, 64, 1024, 1024LL * 1024});
    } else {
        std::fprintf(stderr, "Usage: %s [n batch [lda stride]] [--print]\n", argv[0]);
        return 2;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\nTimes are average per call over 1000 back-to-back launches (us)\n\n",
                prop.name);
    std::printf("%6s %8s %6s %12s %10s | %9s %9s %9s | %7s\n", "n", "batch", "lda", "stride",
                "MB", "optimized", "naive", "memset+d", "GB/s");

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (const Case& c : cases) run_case(c, s, print);
    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
