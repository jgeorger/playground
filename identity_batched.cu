#include "identity_batched.h"

#include <cstdint>
#include <vector>

// Latency notes:
//  * One kernel launch writes the whole batch (no memset + diagonal pass).
//  * No grid-stride loops: each thread does a fixed, tiny amount of work so the
//    grid drains in a single wave for small problems.
//  * Integer div/mod by runtime n, n*n, n+1 is replaced by a precomputed
//    multiply-high + shift (Granlund-Montgomery), ~2 instructions instead of ~20.
//  * Fully packed batches (lda == n, strideA == n*n) are treated as one flat
//    array and written with 16-byte stores (2 complex values per thread).

namespace {

constexpr int kBlockSize = 256;

// Fast unsigned division by a runtime-invariant divisor. Valid for
// dividends < 2^31.
struct FastDivmod {
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift;

    __host__ explicit FastDivmod(uint32_t d) : divisor(d), multiplier(0), shift(0) {
        if (d > 1) {
            uint32_t ceil_log2 = 0;
            while ((1ull << ceil_log2) < d) ++ceil_log2;
            const uint32_t p = 31 + ceil_log2;
            multiplier = static_cast<uint32_t>(((1ull << p) + d - 1) / d);
            shift = p - 32;
        }
    }

    __device__ __forceinline__ uint32_t div(uint32_t x) const {
        return divisor == 1 ? x : (__umulhi(x, multiplier) >> shift);
    }

    __device__ __forceinline__ void divmod(uint32_t x, uint32_t& q, uint32_t& r) const {
        q = div(x);
        r = x - q * divisor;
    }
};

// Packed layout: matrices are back to back, so the batch is a flat array of
// `total` elements and element e is on a diagonal iff (e mod n^2) mod (n+1) == 0.
// Each thread writes elements 2v and 2v+1 with a single float4 store.
__global__ void __launch_bounds__(kBlockSize)
identity_packed_vec2_kernel(float4* __restrict__ A, uint32_t total,
                            FastDivmod nn, FastDivmod np1) {
    const uint32_t v = blockIdx.x * kBlockSize + threadIdx.x;
    const uint32_t e0 = 2 * v;
    if (e0 >= total) return;

    uint32_t q, i0, r0;
    nn.divmod(e0, q, i0);    // i0 = index within matrix
    np1.divmod(i0, q, r0);   // r0 = i0 mod (n+1)

    // Element e0+1: index within matrix is i0+1 (wrapping to 0 at n^2, which is
    // also a diagonal), and its residue mod (n+1) is r0+1 (wrapping at n+1).
    const bool d0 = (r0 == 0);
    const bool d1 = (i0 + 1 == nn.divisor) || (r0 + 1 == np1.divisor);

    const float re0 = d0 ? 1.0f : 0.0f;
    const float re1 = d1 ? 1.0f : 0.0f;

    if (e0 + 1 < total) {
        A[v] = make_float4(re0, 0.0f, re1, 0.0f);
    } else {
        reinterpret_cast<float2*>(A)[e0] = make_float2(re0, 0.0f);
    }
}

// General strided layout: one thread per element of the n x n blocks.
// Consecutive threads map to consecutive rows of a column, so stores coalesce.
__global__ void __launch_bounds__(kBlockSize)
identity_strided_kernel(cuComplex* __restrict__ A, uint32_t total,
                        FastDivmod nn, FastDivmod n,
                        uint32_t lda, long long strideA) {
    const uint32_t t = blockIdx.x * kBlockSize + threadIdx.x;
    if (t >= total) return;

    uint32_t b, i, col, row;
    nn.divmod(t, b, i);
    n.divmod(i, col, row);

    const long long offset = static_cast<long long>(b) * strideA +
                             static_cast<long long>(col) * lda + row;
    A[offset] = make_cuComplex(row == col ? 1.0f : 0.0f, 0.0f);
}

uint32_t blocks_for(uint64_t threads) {
    return static_cast<uint32_t>((threads + kBlockSize - 1) / kBlockSize);
}

}  // namespace

namespace identity_detail {

cudaError_t identity_strided_batched_chunked(cuComplex* A, int n, int lda,
                                             long long strideA, int batch,
                                             cudaStream_t stream,
                                             uint32_t max_elems_per_launch) {
    if (n < 0 || batch < 0) return cudaErrorInvalidValue;
    if (lda < (n > 1 ? n : 1)) return cudaErrorInvalidValue;
    if (batch > 1 && strideA < static_cast<long long>(lda) * n) return cudaErrorInvalidValue;
    const uint64_t elems_per_matrix = static_cast<uint64_t>(n) * n;
    if (elems_per_matrix > INT32_MAX) return cudaErrorInvalidValue;
    if (n == 0 || batch == 0) return cudaSuccess;
    if (A == nullptr) return cudaErrorInvalidValue;
    if (max_elems_per_launch == 0 || max_elems_per_launch > INT32_MAX) {
        max_elems_per_launch = INT32_MAX;
    }

    const bool packed = (lda == n) && (batch == 1 || strideA == static_cast<long long>(elems_per_matrix));

    // Whole matrices per launch. For packed layouts with odd n^2, keep the
    // chunk even so every chunk start stays 16-byte aligned.
    uint64_t chunk = max_elems_per_launch / elems_per_matrix;
    if (chunk == 0) chunk = 1;
    if (packed && (elems_per_matrix & 1) && chunk > 1) chunk &= ~1ull;

    const FastDivmod nn(static_cast<uint32_t>(elems_per_matrix));
    const FastDivmod n_div(static_cast<uint32_t>(n));
    const FastDivmod np1(static_cast<uint32_t>(n) + 1);

    for (uint64_t b0 = 0; b0 < static_cast<uint64_t>(batch); b0 += chunk) {
        const uint64_t nb = (static_cast<uint64_t>(batch) - b0 < chunk)
                                ? static_cast<uint64_t>(batch) - b0 : chunk;
        const uint32_t total = static_cast<uint32_t>(nb * elems_per_matrix);
        cuComplex* base = A + static_cast<long long>(b0) * strideA;

        if (packed && (reinterpret_cast<uintptr_t>(base) % sizeof(float4) == 0)) {
            const uint32_t threads = (total + 1) / 2;
            identity_packed_vec2_kernel<<<blocks_for(threads), kBlockSize, 0, stream>>>(
                reinterpret_cast<float4*>(base), total, nn, np1);
        } else {
            identity_strided_kernel<<<blocks_for(total), kBlockSize, 0, stream>>>(
                base, total, nn, n_div, static_cast<uint32_t>(lda), strideA);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

}  // namespace identity_detail

cudaError_t identity_strided_batched(cuComplex* A, int n, int lda,
                                     long long strideA, int batch,
                                     cudaStream_t stream) {
    return identity_detail::identity_strided_batched_chunked(A, n, lda, strideA, batch,
                                                             stream, INT32_MAX);
}

cudaError_t print_strided_batched(const cuComplex* d_A, int n, int lda,
                                  long long strideA, int batch,
                                  cudaStream_t stream, FILE* out) {
    if (n < 0 || batch < 0 || lda < (n > 1 ? n : 1)) return cudaErrorInvalidValue;
    if (n == 0 || batch == 0) return cudaSuccess;
    if (d_A == nullptr) return cudaErrorInvalidValue;

    const size_t span = static_cast<size_t>(batch - 1) * strideA +
                        static_cast<size_t>(n - 1) * lda + n;
    std::vector<cuComplex> h(span);
    cudaError_t err = cudaMemcpyAsync(h.data(), d_A, span * sizeof(cuComplex),
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    for (int b = 0; b < batch; ++b) {
        std::fprintf(out, "Matrix %d of %d (%d x %d):\n", b, batch, n, n);
        const cuComplex* M = h.data() + static_cast<size_t>(b) * strideA;
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const cuComplex z = M[static_cast<size_t>(col) * lda + row];
                std::fprintf(out, " (%5.2f,%5.2f)", cuCrealf(z), cuCimagf(z));
            }
            std::fprintf(out, "\n");
        }
        std::fprintf(out, "\n");
    }
    return cudaSuccess;
}
