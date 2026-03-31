#ifndef BPDN_CUDA_H
#define BPDN_CUDA_H

#include <cuda_runtime.h>
#include <cstdint>

/**
 * CUDA Implementation of Basis Pursuit Denoise (BPDN)
 *
 * Based on the SPGL1 algorithm by van den Berg and Friedlander (2008).
 * Optimized for solving many small BPDN problems in parallel.
 *
 * Solves: minimize ||x||_1  subject to ||Ax - b||_2 <= sigma
 *
 * This implementation uses the Spectral Projected Gradient (SPG) method
 * with L1-ball projection via soft-thresholding.
 */

namespace bpdn {

// Maximum problem dimensions (can be adjusted based on shared memory constraints)
constexpr int MAX_N = 512;        // Maximum x dimension
constexpr int MAX_M = 512;        // Maximum measurement dimension
constexpr int MAX_ITERATIONS = 500;

// Convergence tolerances
constexpr float DEFAULT_OPT_TOL = 1e-4f;
constexpr float DEFAULT_BP_TOL = 1e-6f;
constexpr float DEFAULT_DECA = 1e-4f;
constexpr float DEFAULT_STEP_MIN = 1e-16f;
constexpr float DEFAULT_STEP_MAX = 1e5f;

/**
 * Algorithm parameters for BPDN solver
 */
struct BPDNParams {
    float sigma;           // Noise level constraint: ||Ax - b||_2 <= sigma
    float optTol;          // Optimality tolerance
    float bpTol;           // Basis pursuit tolerance (for sigma=0 case)
    float decTol;          // Line search sufficient decrease parameter
    float stepMin;         // Minimum step size
    float stepMax;         // Maximum step size
    int maxIterations;     // Maximum number of iterations
    int verbosity;         // 0: silent, 1: summary, 2: per-iteration

    __host__ __device__ BPDNParams() :
        sigma(0.0f),
        optTol(DEFAULT_OPT_TOL),
        bpTol(DEFAULT_BP_TOL),
        decTol(DEFAULT_DECA),
        stepMin(DEFAULT_STEP_MIN),
        stepMax(DEFAULT_STEP_MAX),
        maxIterations(MAX_ITERATIONS),
        verbosity(0) {}
};

/**
 * Output information from BPDN solver
 */
struct BPDNInfo {
    int iterations;        // Number of iterations performed
    int status;            // Exit status code
    float rNorm;           // Final residual norm ||Ax - b||_2
    float xNorm1;          // Final L1 norm ||x||_1
    float gap;             // Duality gap at solution
    float tau;             // Final tau value (L1 ball radius)

    // Status codes
    static constexpr int STATUS_OPTIMAL = 0;
    static constexpr int STATUS_MAX_ITER = 1;
    static constexpr int STATUS_BP_SOLVED = 2;
    static constexpr int STATUS_LS_SOLVED = 3;
    static constexpr int STATUS_ROOT_FOUND = 4;
    static constexpr int STATUS_UNKNOWN = -1;
};

/**
 * Problem batch descriptor for solving multiple BPDN problems
 */
struct BPDNBatch {
    int numProblems;       // Number of problems to solve in parallel
    int m;                 // Number of measurements per problem (rows of A)
    int n;                 // Number of unknowns per problem (cols of A)

    // Device pointers (strided storage for coalesced access)
    float* d_A;            // [numProblems * m * n] - Measurement matrices (column-major per problem)
    float* d_b;            // [numProblems * m] - Observation vectors
    float* d_x;            // [numProblems * n] - Solution vectors (output)
    float* d_sigma;        // [numProblems] - Per-problem sigma values (or single value if uniform)
    BPDNInfo* d_info;      // [numProblems] - Per-problem output info

    bool uniformSigma;     // If true, use single sigma for all problems
};

/**
 * Solve a batch of BPDN problems in parallel on GPU
 *
 * @param batch Problem batch descriptor
 * @param params Algorithm parameters
 * @param stream CUDA stream for asynchronous execution
 * @return cudaError_t CUDA error code
 */
cudaError_t solveBPDNBatch(BPDNBatch& batch, const BPDNParams& params, cudaStream_t stream = 0);

/**
 * Allocate device memory for a batch of BPDN problems
 *
 * @param batch Batch descriptor (numProblems, m, n must be set)
 * @return cudaError_t CUDA error code
 */
cudaError_t allocateBPDNBatch(BPDNBatch& batch);

/**
 * Free device memory for a batch
 *
 * @param batch Batch descriptor
 * @return cudaError_t CUDA error code
 */
cudaError_t freeBPDNBatch(BPDNBatch& batch);

/**
 * Copy problem data from host to device
 *
 * @param batch Batch descriptor with allocated device memory
 * @param h_A Host array of measurement matrices [numProblems * m * n]
 * @param h_b Host array of observation vectors [numProblems * m]
 * @param h_sigma Host array of sigma values [numProblems or 1]
 * @param h_x0 Initial guess for x (optional, can be nullptr) [numProblems * n]
 * @param stream CUDA stream
 * @return cudaError_t CUDA error code
 */
cudaError_t copyToDevice(BPDNBatch& batch,
                         const float* h_A,
                         const float* h_b,
                         const float* h_sigma,
                         const float* h_x0 = nullptr,
                         cudaStream_t stream = 0);

/**
 * Copy solution from device to host
 *
 * @param batch Batch descriptor
 * @param h_x Host array for solutions [numProblems * n]
 * @param h_info Host array for info structs [numProblems]
 * @param stream CUDA stream
 * @return cudaError_t CUDA error code
 */
cudaError_t copyToHost(const BPDNBatch& batch,
                       float* h_x,
                       BPDNInfo* h_info,
                       cudaStream_t stream = 0);

// ============================================================================
// Device functions (available for use in custom kernels)
// ============================================================================

/**
 * Project vector x onto L1 ball of radius tau
 * Soft-thresholding: x_i = sign(x_i) * max(|x_i| - lambda, 0)
 * where lambda is chosen so ||projected_x||_1 = tau
 *
 * @param x Input/output vector
 * @param n Length of vector
 * @param tau L1 ball radius
 * @param workspace Temporary workspace [n elements]
 */
__device__ void projectL1Ball(float* x, int n, float tau, float* workspace);

/**
 * Compute matrix-vector product y = A * x (single problem)
 *
 * @param A Matrix [m x n] column-major
 * @param x Vector [n]
 * @param y Output vector [m]
 * @param m Number of rows
 * @param n Number of columns
 */
__device__ void matVec(const float* A, const float* x, float* y, int m, int n);

/**
 * Compute matrix-transpose-vector product y = A' * x (single problem)
 *
 * @param A Matrix [m x n] column-major
 * @param x Vector [m]
 * @param y Output vector [n]
 * @param m Number of rows
 * @param n Number of columns
 */
__device__ void matTransposeVec(const float* A, const float* x, float* y, int m, int n);

} // namespace bpdn

#endif // BPDN_CUDA_H
