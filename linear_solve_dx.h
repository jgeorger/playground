#ifndef LINEAR_SOLVE_DX_H
#define LINEAR_SOLVE_DX_H

#include <cuda_runtime.h>
#include <cuda/std/complex>

namespace linsol {

// Solve a batch of (D * D^H) x = D * M for x using cuBLASDx and cuSOLVERDx.
//
// d_D: batches contiguous row-major M_DIM x K_DIM matrices, strided by M_DIM*K_DIM.
// d_M: batches contiguous K_DIM x 1 vectors, strided by K_DIM.
// d_x: batches contiguous M_DIM x 1 output vectors, strided by M_DIM.
// d_info: batches ints on device, one per problem (0 = success, >0 = Cholesky failed).
// batches: number of independent problems to solve.
//
// M_DIM and K_DIM must be compile-time constants (required by cuBLASDx/cuSOLVERDx).
// Requires K_DIM >= M_DIM for D*D^H to be positive definite.
template <int M_DIM, int K_DIM>
cudaError_t solveLinearSystem(
    const cuda::std::complex<float>* d_D,
    const cuda::std::complex<float>* d_M,
    cuda::std::complex<float>* d_x,
    int* d_info,
    int batches,
    cudaStream_t stream = 0
);

} // namespace linsol

#endif // LINEAR_SOLVE_DX_H
