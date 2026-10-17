#ifndef IDENTITY_BATCHED_H
#define IDENTITY_BATCHED_H

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Fill a strided batch of n x n column-major cuComplex matrices with the
// identity, using cuBLAS strided-batched conventions:
//
//   element (row, col) of matrix b lives at  A[b * strideA + col * lda + row]
//
// Only the leading n x n block of each matrix is written. Padding rows
// (lda > n) and gaps between matrices (strideA > lda * n) are left untouched.
//
// Requirements: n >= 0, batch >= 0, lda >= max(1, n), strideA >= lda * n
// (when batch > 1), n * n < 2^31. Returns cudaErrorInvalidValue otherwise.
//
// The call is asynchronous with respect to the host (single kernel launch for
// any batch with fewer than 2^31 total elements); no synchronization is done.
cudaError_t identity_strided_batched(cuComplex* A, int n, int lda,
                                     long long strideA, int batch,
                                     cudaStream_t stream = 0);

// Copy the batch to the host and print each n x n matrix to `out`.
// Synchronizes `stream`. Intended for debugging / small matrices.
cudaError_t print_strided_batched(const cuComplex* d_A, int n, int lda,
                                  long long strideA, int batch,
                                  cudaStream_t stream = 0, FILE* out = stdout);

namespace identity_detail {
// Same as identity_strided_batched, but caps the number of elements written
// per kernel launch. Exposed so tests can exercise the multi-launch path
// without allocating > 16 GB.
cudaError_t identity_strided_batched_chunked(cuComplex* A, int n, int lda,
                                             long long strideA, int batch,
                                             cudaStream_t stream,
                                             uint32_t max_elems_per_launch);
}  // namespace identity_detail

#endif  // IDENTITY_BATCHED_H
