#ifndef LINEAR_SOLVE_DX_H
#define LINEAR_SOLVE_DX_H

#include <cuda_runtime.h>
#include <cuda/std/complex>

namespace linsol {

// Solve (D * D^H) x = D * M for x using cuBLASDx and cuSOLVERDx.
//
// D is M_DIM x K_DIM complex<float>, row-major, on device.
// M is K_DIM x 1 complex<float>, on device.
// x is M_DIM x 1 complex<float>, on device (output).
// d_info is a single int on device: 0 on success, >0 if Cholesky factorization failed
//   (i.e., D*D^H is not positive definite).
//
// M_DIM and K_DIM must be compile-time constants (required by cuBLASDx/cuSOLVERDx).
template <int M_DIM, int K_DIM>
cudaError_t solveLinearSystem(
    const cuda::std::complex<float>* d_D,
    const cuda::std::complex<float>* d_M,
    cuda::std::complex<float>* d_x,
    int* d_info,
    cudaStream_t stream = 0
);

} // namespace linsol

#endif // LINEAR_SOLVE_DX_H
