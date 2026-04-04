#include "linear_solve_dx.h"

#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

namespace linsol {

// Fused kernel: computes A = D*D^H, b = D*M, then solves Ax = b via Cholesky.
// All work is done in shared memory within a single CUDA block.
// D and M are stored row-major.
template <unsigned M, unsigned K, unsigned NT>
__global__ __launch_bounds__(NT)
void solve_kernel(
    const commondx::complex<float>* D_global,
    const commondx::complex<float>* M_global,
    commondx::complex<float>* x_global,
    int* info
) {
    using T = commondx::complex<float>;

    // Shared memory layout: Ds (m*k), Ms (k), Gs (m*m), vs (m)
    extern __shared__ __align__(16) unsigned char shared_mem[];
    auto [Ds, Ms, Gs, vs] = cusolverdx::shared_memory::slice<T, T, T, T>(
        shared_mem,
        alignof(T), M * K,
        alignof(T), K,
        alignof(T), M * M,
        alignof(T) // vs size (M elements) omitted for last pointer
    );

    // Index into correct batch
    const unsigned batch = blockIdx.x;
    D_global += batch * M * K;
    M_global += batch * K;
    x_global += batch * M;
    info     += batch;

    #ifdef __CUDA_ARCH__
        constexpr unsigned Arch = __CUDA_ARCH__;
    #else
        constexpr unsigned Arch = 1210;
    #endif

    // GEMM1: G(m,m) = D(m,k) * D^H(k,m)
    // D is row-major m×k in memory = col-major k×m.
    // A row_major: stored col-major K×M, op(A) = A^T = M×K = D
    // B col_major: stored col-major K×M (same memory), op(B) = K×M = D^T
    // With conjugate b_load_op: conj(D^T) = D^H
    // Result: D * D^H
    using GEMM1 = decltype(
        cublasdx::Size<M, M, K>() +
        cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
        cublasdx::Precision<float>() +
        cublasdx::Type<cublasdx::type::complex>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Block() +
        cublasdx::BlockDim<NT>() +
        cublasdx::SM<Arch>());

    // GEMM2: v(m,1) = D(m,k) * M(k,1)
    // A is row-major (D), B is col-major (M vector stored as k elements).
    // D row-major m×k in memory = col-major k×m. With row_major arrangement,
    //   op(A) = A^T = m×k = D.
    // M_vec as col-major k×1, op(B) = B = k×1.
    // Result: D * M = m×1
    using GEMM2 = decltype(
        cublasdx::Size<M, 1, K>() +
        cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
        cublasdx::Precision<float>() +
        cublasdx::Type<cublasdx::type::complex>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Block() +
        cublasdx::BlockDim<NT>() +
        cublasdx::SM<Arch>());

    // POSV: Cholesky-based solve G(m,m) * x = v, 1 RHS
    using POSV = decltype(
        cusolverdx::Function<cusolverdx::posv>() +
        cusolverdx::Size<M, M, 1>() +
        cusolverdx::FillMode<cusolverdx::lower>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::complex>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<NT>() +
        cusolverdx::SM<Arch>());

    // Load D (row-major m×k) and M (k×1) from global memory to shared memory
    // D row-major: M rows, K cols per row, stride = K
    cusolverdx::copy_2d<NT, M, K, cusolverdx::arrangement::row_major>(D_global, K, Ds, K);
    cusolverdx::copy_2d<NT, K, 1, cusolverdx::arrangement::col_major>(M_global, K, Ms, K);
    __syncthreads();

    // GEMM1: G = D * D^H (conjugate on B gives Hermitian transpose)
    GEMM1().execute(T(1.0f, 0.0f), Ds, Ds, T(0.0f, 0.0f), Gs,
                    cublasdx::identity{}, cublasdx::conjugate{});
    __syncthreads();

    // GEMM2: v = D * M
    GEMM2().execute(T(1.0f, 0.0f), Ds, Ms, T(0.0f, 0.0f), vs);
    __syncthreads();

    // POSV: solve G * x = v (solution overwrites vs)
    POSV().execute(Gs, vs, info);
    __syncthreads();

    // Store solution back to global memory
    cusolverdx::copy_2d<NT, M, 1, cusolverdx::arrangement::col_major>(vs, M, x_global, M);
}

template <int M_DIM, int K_DIM>
cudaError_t solveLinearSystem(
    const cuda::std::complex<float>* d_D,
    const cuda::std::complex<float>* d_M,
    cuda::std::complex<float>* d_x,
    int* d_info,
    cudaStream_t stream
) {
    constexpr unsigned NT = 128;
    constexpr unsigned smem_size =
        sizeof(commondx::complex<float>) * (M_DIM * K_DIM + K_DIM + M_DIM * M_DIM + M_DIM);

    // commondx::complex<float> and cuda::std::complex<float> are layout-compatible
    auto d_D_cast = reinterpret_cast<const commondx::complex<float>*>(d_D);
    auto d_M_cast = reinterpret_cast<const commondx::complex<float>*>(d_M);
    auto d_x_cast = reinterpret_cast<commondx::complex<float>*>(d_x);

    auto kernel = solve_kernel<M_DIM, K_DIM, NT>;

    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) return err;

    kernel<<<1, NT, smem_size, stream>>>(d_D_cast, d_M_cast, d_x_cast, d_info);
    return cudaGetLastError();
}

// Explicit template instantiations (k >= m required for D*D^H to be positive definite)
template cudaError_t solveLinearSystem<4, 8>(
    const cuda::std::complex<float>*, const cuda::std::complex<float>*,
    cuda::std::complex<float>*, int*, cudaStream_t);
template cudaError_t solveLinearSystem<8, 16>(
    const cuda::std::complex<float>*, const cuda::std::complex<float>*,
    cuda::std::complex<float>*, int*, cudaStream_t);
template cudaError_t solveLinearSystem<16, 32>(
    const cuda::std::complex<float>*, const cuda::std::complex<float>*,
    cuda::std::complex<float>*, int*, cudaStream_t);

} // namespace linsol
