// Unit tests for identity_strided_batched.
//
// Usage:
//   ./build/release/test_identity_batched            # run tests
//   ./build/release/test_identity_batched --print    # also print the matrices
//                                                    # from the IdentityBatchedPrint tests
// Standard GoogleTest flags (e.g. --gtest_filter=*Print*) are also accepted.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "identity_batched.h"

static bool g_print = false;

#define CUDA_OK(expr)                                                     \
    do {                                                                  \
        const cudaError_t err_ = (expr);                                  \
        ASSERT_EQ(err_, cudaSuccess) << #expr << ": " << cudaGetErrorString(err_); \
    } while (0)

namespace {

// Bit pattern written to every element before the kernel runs, so we can
// verify that padding / gaps are left untouched.
constexpr uint32_t kSentinelBits = 0xDEADBEEFu;

uint64_t bits_of(cuComplex z) {
    uint64_t u;
    std::memcpy(&u, &z, sizeof(u));
    return u;
}

cuComplex sentinel() {
    cuComplex z;
    uint32_t parts[2] = {kSentinelBits, kSentinelBits};
    std::memcpy(&z, parts, sizeof(z));
    return z;
}

struct Layout {
    int n;
    int lda;
    long long stride;
    int batch;
};

// Allocates the batch (plus `offset_elems` leading elements, to test
// misaligned base pointers), fills it with the sentinel, runs `fill`, and
// checks every element of the allocation. Prints the batch when `print` is set
// and the binary was run with --print.
template <typename Fill>
void run_and_check(const Layout& L, size_t offset_elems, Fill fill, bool print = false) {
    const size_t span = (L.n == 0 || L.batch == 0)
                            ? 1
                            : static_cast<size_t>(L.batch - 1) * L.stride +
                                  static_cast<size_t>(L.n - 1) * L.lda + L.n;
    const size_t alloc = span + offset_elems + 1;  // +1 trailing guard element

    std::vector<cuComplex> h(alloc, sentinel());
    cuComplex* d = nullptr;
    CUDA_OK(cudaMalloc(&d, alloc * sizeof(cuComplex)));
    CUDA_OK(cudaMemcpy(d, h.data(), alloc * sizeof(cuComplex), cudaMemcpyHostToDevice));

    cuComplex* A = d + offset_elems;
    CUDA_OK(fill(A));
    CUDA_OK(cudaDeviceSynchronize());

    if (print && g_print && L.n > 0 && L.batch > 0) {
        CUDA_OK(print_strided_batched(A, L.n, L.lda, L.stride, L.batch));
    }

    CUDA_OK(cudaMemcpy(h.data(), d, alloc * sizeof(cuComplex), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d));

    // Build the expected image.
    std::vector<cuComplex> expect(alloc, sentinel());
    for (int b = 0; b < L.batch; ++b)
        for (int col = 0; col < L.n; ++col)
            for (int row = 0; row < L.n; ++row)
                expect[offset_elems + static_cast<size_t>(b) * L.stride +
                       static_cast<size_t>(col) * L.lda + row] =
                    make_cuComplex(row == col ? 1.0f : 0.0f, 0.0f);

    size_t mismatches = 0;
    for (size_t k = 0; k < alloc; ++k) {
        if (bits_of(h[k]) != bits_of(expect[k])) {
            if (mismatches < 10) {
                ADD_FAILURE() << "n=" << L.n << " lda=" << L.lda << " stride=" << L.stride
                              << " batch=" << L.batch << " offset=" << offset_elems
                              << ": element " << k << " = (" << h[k].x << "," << h[k].y
                              << ") expected (" << expect[k].x << "," << expect[k].y << ")";
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

void check(const Layout& L, size_t offset_elems = 0, bool print = false) {
    run_and_check(L, offset_elems, [&](cuComplex* A) {
        return identity_strided_batched(A, L.n, L.lda, L.stride, L.batch);
    }, print);
}

}  // namespace

// ---------------------------------------------------------------------------
// Packed layout (lda == n, stride == n*n): vectorized path.
// ---------------------------------------------------------------------------

TEST(IdentityBatched, PackedManySizes) {
    for (int n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100, 129}) {
        for (int batch : {1, 2, 3, 5, 64, 257}) {
            check({n, n, static_cast<long long>(n) * n, batch});
        }
    }
}

TEST(IdentityBatched, PackedLargeBatch) {
    check({4, 4, 16, 100000});
    check({3, 3, 9, 100001});  // odd n^2 and odd batch -> odd total
}

TEST(IdentityBatched, PackedLargeMatrix) {
    check({1024, 1024, 1024LL * 1024, 2});
    check({1023, 1023, 1023LL * 1023, 3});
}

// Base pointer 8-byte but not 16-byte aligned -> falls back to scalar path.
TEST(IdentityBatched, PackedMisalignedPointer) {
    for (int n : {1, 2, 3, 8, 33}) {
        check({n, n, static_cast<long long>(n) * n, 7}, /*offset_elems=*/1);
    }
}

// ---------------------------------------------------------------------------
// Strided layouts: padding rows and/or gaps between matrices.
// ---------------------------------------------------------------------------

TEST(IdentityBatched, PaddedLda) {
    for (int n : {1, 2, 3, 8, 17, 64}) {
        for (int pad : {1, 3, 16}) {
            const int lda = n + pad;
            check({n, lda, static_cast<long long>(lda) * n, 9});
        }
    }
}

TEST(IdentityBatched, PaddedStride) {
    for (int n : {1, 2, 5, 16, 33}) {
        for (long long gap : {1LL, 7LL, 100LL}) {
            check({n, n, static_cast<long long>(n) * n + gap, 11});
        }
    }
}

TEST(IdentityBatched, PaddedLdaAndStride) {
    check({6, 8, 8 * 6 + 13, 25});
    check({32, 40, 40 * 32 + 64, 10});
}

TEST(IdentityBatched, SingleMatrixIgnoresStride) {
    // With batch == 1 the stride is irrelevant (cuBLAS allows 0).
    check({5, 5, 0, 1});
    check({5, 7, 0, 1});
}

// ---------------------------------------------------------------------------
// Multi-launch chunking (normally only for > 2^31 elements).
// ---------------------------------------------------------------------------

TEST(IdentityBatched, ChunkedLaunches) {
    const std::vector<Layout> layouts = {
        {3, 3, 9, 50},        // packed, odd n^2: chunks must stay 16B aligned
        {4, 4, 16, 50},       // packed, even n^2
        {5, 7, 7 * 5 + 3, 50} // strided
    };
    for (const Layout& L : layouts) {
        for (uint32_t max_elems : {1u, 9u, 20u, 64u, 100u, 1000u}) {
            run_and_check(L, 0, [&](cuComplex* A) {
                return identity_detail::identity_strided_batched_chunked(
                    A, L.n, L.lda, L.stride, L.batch, 0, max_elems);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Edge cases and argument validation.
// ---------------------------------------------------------------------------

TEST(IdentityBatched, EmptyIsNoOp) {
    check({0, 1, 0, 10});
    check({4, 4, 16, 0});
    EXPECT_EQ(identity_strided_batched(nullptr, 0, 1, 0, 5), cudaSuccess);
    EXPECT_EQ(identity_strided_batched(nullptr, 4, 4, 16, 0), cudaSuccess);
}

TEST(IdentityBatched, InvalidArguments) {
    cuComplex* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, 64 * sizeof(cuComplex)), cudaSuccess);
    EXPECT_EQ(identity_strided_batched(d, -1, 1, 1, 1), cudaErrorInvalidValue);
    EXPECT_EQ(identity_strided_batched(d, 4, 4, 16, -1), cudaErrorInvalidValue);
    EXPECT_EQ(identity_strided_batched(d, 4, 3, 16, 1), cudaErrorInvalidValue);   // lda < n
    EXPECT_EQ(identity_strided_batched(d, 4, 4, 15, 2), cudaErrorInvalidValue);   // overlap
    EXPECT_EQ(identity_strided_batched(nullptr, 4, 4, 16, 1), cudaErrorInvalidValue);
    EXPECT_EQ(identity_strided_batched(d, 50000, 50000, 0, 1), cudaErrorInvalidValue);  // n^2 >= 2^31
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    cudaFree(d);
}

TEST(IdentityBatched, NonDefaultStream) {
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
    const Layout L{8, 8, 64, 33};
    run_and_check(L, 0, [&](cuComplex* A) {
        cudaError_t e = identity_strided_batched(A, L.n, L.lda, L.stride, L.batch, s);
        return e != cudaSuccess ? e : cudaStreamSynchronize(s);
    });
    cudaStreamDestroy(s);
}

// ---------------------------------------------------------------------------
// Small cases meant for visual inspection with --print.
// ---------------------------------------------------------------------------

TEST(IdentityBatchedPrint, Packed3x3Batch2) {
    check({3, 3, 9, 2}, 0, /*print=*/true);
}

TEST(IdentityBatchedPrint, Strided4x4Lda5Batch2) {
    check({4, 5, 5 * 4 + 2, 2}, 0, /*print=*/true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);  // strips --gtest_* flags
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print") == 0) {
            g_print = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\nUsage: %s [--print] [gtest flags]\n",
                         argv[i], argv[0]);
            return 2;
        }
    }
    return RUN_ALL_TESTS();
}
