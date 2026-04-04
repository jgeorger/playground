#include "linear_solve_dx.h"

#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <complex>
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>

using cfloat = std::complex<float>;

// Host reference: compute A = D * D^H (m x m), b = D * M (m x 1), solve Ax = b
// D is row-major m x k, M is k x 1
static void host_reference(int m, int k,
                           const cfloat* D, const cfloat* M,
                           cfloat* x) {
    // Compute A = D * D^H (m x m)
    // D row-major: D[i][l] = D[i * k + l]
    std::vector<cfloat> A(m * m, cfloat(0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            cfloat sum(0);
            for (int l = 0; l < k; l++) {
                sum += D[i * k + l] * std::conj(D[j * k + l]);
            }
            A[i + j * m] = sum;
        }
    }

    // Compute b = D * M (m x 1)
    std::vector<cfloat> b(m, cfloat(0));
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            b[i] += D[i * k + l] * M[l];
        }
    }

    // Solve Ax = b via Gaussian elimination with partial pivoting
    // Augmented matrix [A | b]
    std::vector<cfloat> aug(m * (m + 1));
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++)
            aug[i + j * m] = A[i + j * m];
    for (int i = 0; i < m; i++)
        aug[i + m * m] = b[i];

    // Forward elimination
    for (int col = 0; col < m; col++) {
        // Partial pivoting
        int max_row = col;
        float max_val = std::abs(aug[col + col * m]);
        for (int row = col + 1; row < m; row++) {
            float val = std::abs(aug[row + col * m]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        if (max_row != col) {
            for (int j = 0; j <= m; j++) {
                std::swap(aug[col + j * m], aug[max_row + j * m]);
            }
        }

        cfloat pivot = aug[col + col * m];
        for (int row = col + 1; row < m; row++) {
            cfloat factor = aug[row + col * m] / pivot;
            for (int j = col; j <= m; j++) {
                aug[row + j * m] -= factor * aug[col + j * m];
            }
        }
    }

    // Back substitution
    for (int i = m - 1; i >= 0; i--) {
        cfloat sum = aug[i + m * m];
        for (int j = i + 1; j < m; j++) {
            sum -= aug[i + j * m] * x[j];
        }
        x[i] = sum / aug[i + i * m];
    }
}

template <int M_DIM, int K_DIM>
static int run_test(int batches) {
    printf("Testing M=%d, K=%d, batches=%d\n", M_DIM, K_DIM, batches);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate random D (m x k) and M (k x 1) for each batch
    std::vector<cfloat> h_D(batches * M_DIM * K_DIM);
    std::vector<cfloat> h_M(batches * K_DIM);
    for (auto& v : h_D) v = cfloat(dist(rng), dist(rng));
    for (auto& v : h_M) v = cfloat(dist(rng), dist(rng));

    // Host reference solutions
    std::vector<cfloat> h_x_ref(batches * M_DIM);
    for (int b = 0; b < batches; b++) {
        host_reference(M_DIM, K_DIM,
                       h_D.data() + b * M_DIM * K_DIM,
                       h_M.data() + b * K_DIM,
                       h_x_ref.data() + b * M_DIM);
    }

    // Allocate device memory
    cuda::std::complex<float>* d_D = nullptr;
    cuda::std::complex<float>* d_M = nullptr;
    cuda::std::complex<float>* d_x = nullptr;
    int* d_info = nullptr;

    cudaMalloc(&d_D, sizeof(cfloat) * batches * M_DIM * K_DIM);
    cudaMalloc(&d_M, sizeof(cfloat) * batches * K_DIM);
    cudaMalloc(&d_x, sizeof(cfloat) * batches * M_DIM);
    cudaMalloc(&d_info, sizeof(int) * batches);

    cudaMemcpy(d_D, h_D.data(), sizeof(cfloat) * batches * M_DIM * K_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M.data(), sizeof(cfloat) * batches * K_DIM, cudaMemcpyHostToDevice);

    // Run GPU solver
    cudaError_t err = linsol::solveLinearSystem<M_DIM, K_DIM>(d_D, d_M, d_x, d_info, batches);
    if (err != cudaSuccess) {
        printf("  FAIL: kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_D); cudaFree(d_M); cudaFree(d_x); cudaFree(d_info);
        return 1;
    }
    cudaDeviceSynchronize();

    // Check solver info for each batch
    std::vector<int> h_info(batches);
    cudaMemcpy(h_info.data(), d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost);
    for (int b = 0; b < batches; b++) {
        if (h_info[b] != 0) {
            printf("  FAIL: batch %d solver returned info=%d (Cholesky factorization failed)\n", b, h_info[b]);
            cudaFree(d_D); cudaFree(d_M); cudaFree(d_x); cudaFree(d_info);
            return 1;
        }
    }

    // Copy results back
    std::vector<cfloat> h_x_gpu(batches * M_DIM);
    cudaMemcpy(h_x_gpu.data(), d_x, sizeof(cfloat) * batches * M_DIM, cudaMemcpyDeviceToHost);

    // Compute relative error across all batches
    float norm_diff = 0.0f;
    float norm_ref = 0.0f;
    float max_batch_err = 0.0f;
    for (int b = 0; b < batches; b++) {
        float bd = 0.0f, br = 0.0f;
        for (int i = 0; i < M_DIM; i++) {
            cfloat diff = h_x_gpu[b * M_DIM + i] - h_x_ref[b * M_DIM + i];
            bd += std::norm(diff);
            br += std::norm(h_x_ref[b * M_DIM + i]);
        }
        norm_diff += bd;
        norm_ref += br;
        float batch_err = (br > 0.0f) ? std::sqrt(bd) / std::sqrt(br) : std::sqrt(bd);
        if (batch_err > max_batch_err) max_batch_err = batch_err;
    }
    float rel_error = std::sqrt(norm_diff) / std::sqrt(norm_ref);
    printf("  Overall relative error: %e\n", rel_error);
    printf("  Max per-batch relative error: %e\n", max_batch_err);

    cudaFree(d_D);
    cudaFree(d_M);
    cudaFree(d_x);
    cudaFree(d_info);

    constexpr float tol = 1e-4f;
    if (max_batch_err < tol) {
        printf("  PASS\n");
        return 0;
    } else {
        printf("  FAIL (tolerance: %e)\n", tol);
        return 1;
    }
}

int main() {
    constexpr int BATCHES = 16;

    int failures = 0;
    failures += run_test<4, 8>(BATCHES);
    failures += run_test<8, 16>(BATCHES);
    failures += run_test<16, 32>(BATCHES);

    printf("\n%s (%d test(s) failed)\n",
           failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED", failures);
    return failures;
}
