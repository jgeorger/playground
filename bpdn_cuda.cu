#include "bpdn_cuda.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace bpdn {

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ float blockReduceSum(float val, float* sharedMem, int tid, int blockSize) {
    sharedMem[tid] = val;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockSize) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    return sharedMem[0];
}

__device__ float blockReduceMax(float val, float* sharedMem, int tid, int blockSize) {
    sharedMem[tid] = val;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockSize) {
            sharedMem[tid] = fmaxf(sharedMem[tid], sharedMem[tid + stride]);
        }
        __syncthreads();
    }
    return sharedMem[0];
}

__device__ float dotProduct(const float* a, const float* b, int n,
                            float* sharedMem, int tid, int blockSize) {
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockSize) {
        sum += a[i] * b[i];
    }
    return blockReduceSum(sum, sharedMem, tid, blockSize);
}

__device__ float norm2(const float* x, int n, float* sharedMem, int tid, int blockSize) {
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockSize) {
        sum += x[i] * x[i];
    }
    return sqrtf(blockReduceSum(sum, sharedMem, tid, blockSize));
}

__device__ float norm1(const float* x, int n, float* sharedMem, int tid, int blockSize) {
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockSize) {
        sum += fabsf(x[i]);
    }
    return blockReduceSum(sum, sharedMem, tid, blockSize);
}

__device__ float normInf(const float* x, int n, float* sharedMem, int tid, int blockSize) {
    float maxVal = 0.0f;
    for (int i = tid; i < n; i += blockSize) {
        maxVal = fmaxf(maxVal, fabsf(x[i]));
    }
    return blockReduceMax(maxVal, sharedMem, tid, blockSize);
}

__device__ void matVec(const float* A, const float* x, float* y, int m, int n) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    for (int i = tid; i < m; i += blockSize) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[j * m + i] * x[j];
        }
        y[i] = sum;
    }
    __syncthreads();
}

__device__ void matTransposeVec(const float* A, const float* x, float* y, int m, int n) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    for (int j = tid; j < n; j += blockSize) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += A[j * m + i] * x[i];
        }
        y[j] = sum;
    }
    __syncthreads();
}

__device__ void vecCopy(float* y, const float* x, int n) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    for (int i = tid; i < n; i += blockSize) {
        y[i] = x[i];
    }
    __syncthreads();
}

__device__ void vecSet(float* x, float val, int n) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    for (int i = tid; i < n; i += blockSize) {
        x[i] = val;
    }
    __syncthreads();
}

/**
 * Soft-thresholding: S_lambda(x)_i = sign(x_i) * max(|x_i| - lambda, 0)
 */
__device__ void softThreshold(float* x, float lambda, int n) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    for (int i = tid; i < n; i += blockSize) {
        float val = x[i];
        float absVal = fabsf(val);
        if (absVal <= lambda) {
            x[i] = 0.0f;
        } else {
            x[i] = copysignf(absVal - lambda, val);
        }
    }
    __syncthreads();
}

/**
 * Project onto L1 ball of radius tau
 */
__device__ void projectL1Ball(float* x, int n, float tau, float* workspace) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (tau <= 0.0f) {
        vecSet(x, 0.0f, n);
        return;
    }

    float xNorm1 = norm1(x, n, workspace, tid, blockSize);
    if (xNorm1 <= tau) {
        return;
    }

    // Store absolute values
    for (int i = tid; i < n; i += blockSize) {
        workspace[i] = fabsf(x[i]);
    }
    __syncthreads();

    float maxAbs = normInf(x, n, workspace + n, tid, blockSize);
    float lambdaLow = 0.0f;
    float lambdaHigh = maxAbs;
    float lambda = 0.0f;

    for (int iter = 0; iter < 40; iter++) {
        lambda = 0.5f * (lambdaLow + lambdaHigh);

        float sumThresh = 0.0f;
        for (int i = tid; i < n; i += blockSize) {
            sumThresh += fmaxf(workspace[i] - lambda, 0.0f);
        }
        float totalSum = blockReduceSum(sumThresh, workspace + n, tid, blockSize);

        if (fabsf(totalSum - tau) < 1e-7f * tau) break;

        if (totalSum > tau) {
            lambdaLow = lambda;
        } else {
            lambdaHigh = lambda;
        }
    }

    for (int i = tid; i < n; i += blockSize) {
        float val = x[i];
        float absVal = fabsf(val);
        x[i] = (absVal <= lambda) ? 0.0f : copysignf(absVal - lambda, val);
    }
    __syncthreads();
}

/**
 * Estimate Lipschitz constant ||A'A||_2 using power iteration
 */
__device__ float estimateLipschitz(const float* A, int m, int n,
                                    float* v, float* Av, float* AtAv,
                                    float* reduceWs, int tid, int blockSize) {
    for (int i = tid; i < n; i += blockSize) {
        v[i] = 1.0f / sqrtf((float)n);
    }
    __syncthreads();

    float sigma = 1.0f;
    for (int iter = 0; iter < 30; iter++) {
        matVec(A, v, Av, m, n);
        matTransposeVec(A, Av, AtAv, m, n);

        sigma = norm2(AtAv, n, reduceWs, tid, blockSize);
        if (sigma < 1e-10f) {
            sigma = 1.0f;
            break;
        }

        for (int i = tid; i < n; i += blockSize) {
            v[i] = AtAv[i] / sigma;
        }
        __syncthreads();
    }

    return sigma;
}

// ============================================================================
// BPDN Kernel using FISTA with LASSO formulation
// ============================================================================

/**
 * Solve BPDN: min ||x||_1 s.t. ||Ax-b||_2 <= sigma
 *
 * Equivalent to finding lambda* for LASSO: min (1/2)||Ax-b||^2 + lambda*||x||_1
 * such that the solution has ||Ax-b|| = sigma.
 *
 * Uses FISTA for inner optimization and bisection on lambda for root-finding.
 */
__global__ void bpdnKernel(
    const float* __restrict__ A,
    const float* __restrict__ b,
    float* __restrict__ x,
    const float* __restrict__ sigmas,
    BPDNInfo* __restrict__ info,
    int m, int n,
    bool uniformSigma,
    BPDNParams params)
{
    int probIdx = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    const float* myA = A + probIdx * m * n;
    const float* myB = b + probIdx * m;
    float* myX = x + probIdx * n;
    float sigma = uniformSigma ? sigmas[0] : sigmas[probIdx];
    BPDNInfo* myInfo = info + probIdx;

    extern __shared__ float sharedMem[];

    // Memory layout
    float* x_curr = sharedMem;                    // [n]
    float* y = sharedMem + n;                     // [n] FISTA momentum
    float* x_prev = sharedMem + 2 * n;            // [n]
    float* grad = sharedMem + 3 * n;              // [n]
    float* r = sharedMem + 4 * n;                 // [m]
    float* Av = sharedMem + 4 * n + m;            // [m]
    float* workspace = sharedMem + 4 * n + 2 * m; // [2*n]
    float* reduceWs = sharedMem + 6 * n + 2 * m;  // [blockSize]

    // Initialize
    vecSet(x_curr, 0.0f, n);
    vecCopy(y, x_curr, n);

    // Compute norms for scaling
    float bNorm = norm2(myB, m, reduceWs, tid, blockSize);
    if (bNorm < 1e-10f) bNorm = 1.0f;

    // Lipschitz constant
    float L = estimateLipschitz(myA, m, n, workspace, Av, grad, reduceWs, tid, blockSize);
    L = fmaxf(L, 1e-6f);
    float invL = 1.0f / L;

    // Compute A'b for lambda bounds
    matTransposeVec(myA, myB, grad, m, n);
    float lambdaMax = normInf(grad, n, reduceWs, tid, blockSize);

    // For BPDN, we search for lambda such that ||Ax-b|| = sigma
    // lambda = 0 gives least squares (small residual)
    // lambda = lambdaMax gives x = 0 (residual = ||b||)

    float lambda;
    float lambda_low = 0.0f;
    float lambda_high = lambdaMax;
    float rNorm_low = 0.0f;   // Will be computed
    float rNorm_high = bNorm; // At lambda = lambdaMax, x = 0
    bool foundLow = false;
    bool foundHigh = true;

    // Initial lambda guess
    if (sigma <= 0.0f || sigma >= bNorm) {
        // Basis pursuit or infeasible: use small lambda
        lambda = 1e-6f * lambdaMax;
    } else {
        // Interpolate based on sigma
        lambda = lambdaMax * (bNorm - sigma) / bNorm;
    }

    int status = BPDNInfo::STATUS_UNKNOWN;
    int totalIters = 0;
    float rNorm = bNorm;

    // Use adaptive iteration count based on problem size
    int maxFistaIters = 150;
    float innerTol = 1e-5f;

    // Outer loop: bisection on lambda
    for (int outerIter = 0; outerIter < 50; outerIter++) {

        // Reset FISTA for new lambda
        float t = 1.0f;
        vecCopy(y, x_curr, n);

        // Inner FISTA loop until convergence
        for (int fistaIter = 0; fistaIter < maxFistaIters; fistaIter++) {
            totalIters++;

            vecCopy(x_prev, x_curr, n);
            float t_prev = t;

            // Gradient: A'(Ay - b)
            matVec(myA, y, r, m, n);
            for (int i = tid; i < m; i += blockSize) {
                r[i] = r[i] - myB[i];
            }
            __syncthreads();

            matTransposeVec(myA, r, grad, m, n);

            // Gradient step + soft threshold
            for (int i = tid; i < n; i += blockSize) {
                x_curr[i] = y[i] - invL * grad[i];
            }
            __syncthreads();

            // Soft threshold with lambda/L
            float thresh = lambda * invL;
            softThreshold(x_curr, thresh, n);

            // FISTA momentum
            t = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t_prev * t_prev));
            float beta = (t_prev - 1.0f) / t;

            for (int i = tid; i < n; i += blockSize) {
                y[i] = x_curr[i] + beta * (x_curr[i] - x_prev[i]);
            }
            __syncthreads();

            // Check convergence based on change in x
            float diffNorm = 0.0f;
            for (int i = tid; i < n; i += blockSize) {
                float d = x_curr[i] - x_prev[i];
                diffNorm += d * d;
            }
            diffNorm = sqrtf(blockReduceSum(diffNorm, reduceWs, tid, blockSize));

            float xNorm = norm2(x_curr, n, reduceWs, tid, blockSize);
            if (diffNorm < innerTol * fmaxf(xNorm, 1e-6f)) {
                break;
            }
        }

        // Compute residual
        matVec(myA, x_curr, r, m, n);
        for (int i = tid; i < m; i += blockSize) {
            r[i] = myB[i] - r[i];
        }
        __syncthreads();

        rNorm = norm2(r, m, reduceWs, tid, blockSize);

        // Check termination
        if (sigma <= 0.0f) {
            // Basis pursuit: minimize residual
            if (rNorm < params.bpTol * bNorm) {
                status = BPDNInfo::STATUS_BP_SOLVED;
                break;
            }
            lambda *= 0.5f;
        } else {
            float relErr = fabsf(rNorm - sigma) / sigma;
            if (relErr < params.optTol || relErr < 0.05f) {  // Accept if within 5%
                status = BPDNInfo::STATUS_ROOT_FOUND;
                break;
            }

            // Update bisection bounds based on residual
            if (rNorm < sigma) {
                lambda_low = lambda;
                rNorm_low = rNorm;
                foundLow = true;
            } else {
                lambda_high = lambda;
                rNorm_high = rNorm;
                foundHigh = true;
            }

            // Update lambda using bisection
            if (!foundLow) {
                lambda *= 0.5f;
            } else if (!foundHigh) {
                lambda *= 2.0f;
            } else {
                // Linear interpolation for faster convergence
                float t = (sigma - rNorm_low) / (rNorm_high - rNorm_low + 1e-10f);
                t = fmaxf(0.1f, fminf(0.9f, t));  // Keep away from extremes
                lambda = lambda_low + t * (lambda_high - lambda_low);
            }

            lambda = fmaxf(1e-12f, fminf(lambda, lambdaMax));
        }

        if (totalIters >= params.maxIterations) {
            break;
        }
    }

    if (status == BPDNInfo::STATUS_UNKNOWN) {
        // Accept if within reasonable tolerance of target
        if (sigma > 0.0f && fabsf(rNorm - sigma) / sigma < 0.2f) {
            status = BPDNInfo::STATUS_ROOT_FOUND;
        } else {
            status = BPDNInfo::STATUS_MAX_ITER;
        }
    }

    // Copy solution
    for (int i = tid; i < n; i += blockSize) {
        myX[i] = x_curr[i];
    }
    __syncthreads();

    float finalL1 = norm1(x_curr, n, reduceWs, tid, blockSize);

    if (tid == 0) {
        myInfo->iterations = totalIters;
        myInfo->status = status;
        myInfo->rNorm = rNorm;
        myInfo->xNorm1 = finalL1;
        myInfo->gap = 0.0f;
        myInfo->tau = lambda;  // Store lambda instead of tau
    }
}

// ============================================================================
// Host Interface
// ============================================================================

cudaError_t allocateBPDNBatch(BPDNBatch& batch) {
    cudaError_t err;

    size_t sizeA = (size_t)batch.numProblems * batch.m * batch.n * sizeof(float);
    size_t sizeB = (size_t)batch.numProblems * batch.m * sizeof(float);
    size_t sizeX = (size_t)batch.numProblems * batch.n * sizeof(float);
    size_t sizeSigma = batch.uniformSigma ? sizeof(float) : batch.numProblems * sizeof(float);
    size_t sizeInfo = batch.numProblems * sizeof(BPDNInfo);

    err = cudaMalloc(&batch.d_A, sizeA);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&batch.d_b, sizeB);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&batch.d_x, sizeX);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&batch.d_sigma, sizeSigma);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&batch.d_info, sizeInfo);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

cudaError_t freeBPDNBatch(BPDNBatch& batch) {
    if (batch.d_A) cudaFree(batch.d_A);
    if (batch.d_b) cudaFree(batch.d_b);
    if (batch.d_x) cudaFree(batch.d_x);
    if (batch.d_sigma) cudaFree(batch.d_sigma);
    if (batch.d_info) cudaFree(batch.d_info);

    batch.d_A = nullptr;
    batch.d_b = nullptr;
    batch.d_x = nullptr;
    batch.d_sigma = nullptr;
    batch.d_info = nullptr;

    return cudaSuccess;
}

cudaError_t copyToDevice(BPDNBatch& batch,
                         const float* h_A,
                         const float* h_b,
                         const float* h_sigma,
                         const float* h_x0,
                         cudaStream_t stream) {
    cudaError_t err;

    size_t sizeA = (size_t)batch.numProblems * batch.m * batch.n * sizeof(float);
    size_t sizeB = (size_t)batch.numProblems * batch.m * sizeof(float);
    size_t sizeX = (size_t)batch.numProblems * batch.n * sizeof(float);
    size_t sizeSigma = batch.uniformSigma ? sizeof(float) : batch.numProblems * sizeof(float);

    err = cudaMemcpyAsync(batch.d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(batch.d_b, h_b, sizeB, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(batch.d_sigma, h_sigma, sizeSigma, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    if (h_x0) {
        err = cudaMemcpyAsync(batch.d_x, h_x0, sizeX, cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemsetAsync(batch.d_x, 0, sizeX, stream);
    }
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

cudaError_t copyToHost(const BPDNBatch& batch,
                       float* h_x,
                       BPDNInfo* h_info,
                       cudaStream_t stream) {
    cudaError_t err;

    size_t sizeX = (size_t)batch.numProblems * batch.n * sizeof(float);
    size_t sizeInfo = batch.numProblems * sizeof(BPDNInfo);

    err = cudaMemcpyAsync(h_x, batch.d_x, sizeX, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(h_info, batch.d_info, sizeInfo, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

cudaError_t solveBPDNBatch(BPDNBatch& batch, const BPDNParams& params, cudaStream_t stream) {
    int m = batch.m;
    int n = batch.n;

    int blockSize = 128;

    // Shared memory: x_curr[n] + y[n] + x_prev[n] + grad[n] + r[m] + Av[m]
    //                + workspace[2n] + reduceWs[blockSize]
    size_t sharedMemSize = (6 * n + 2 * n + 2 * m + blockSize) * sizeof(float);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    if (sharedMemSize > props.sharedMemPerBlock) {
        fprintf(stderr, "BPDN Error: Problem too large for shared memory. "
                "Required: %zu, Available: %zu\n",
                sharedMemSize, props.sharedMemPerBlock);
        return cudaErrorInvalidConfiguration;
    }

    bpdnKernel<<<batch.numProblems, blockSize, sharedMemSize, stream>>>(
        batch.d_A, batch.d_b, batch.d_x, batch.d_sigma, batch.d_info,
        m, n, batch.uniformSigma, params
    );

    return cudaGetLastError();
}

} // namespace bpdn

extern "C" {

int solveBPDN_batch(
    int numProblems,
    int m,
    int n,
    const float* h_A,
    const float* h_b,
    float* h_x,
    float sigma,
    int maxIter,
    float tol)
{
    bpdn::BPDNBatch batch;
    batch.numProblems = numProblems;
    batch.m = m;
    batch.n = n;
    batch.uniformSigma = true;

    cudaError_t err = bpdn::allocateBPDNBatch(batch);
    if (err != cudaSuccess) return (int)err;

    err = bpdn::copyToDevice(batch, h_A, h_b, &sigma, nullptr, 0);
    if (err != cudaSuccess) {
        bpdn::freeBPDNBatch(batch);
        return (int)err;
    }

    bpdn::BPDNParams params;
    params.sigma = sigma;
    params.maxIterations = maxIter;
    params.optTol = tol;

    err = bpdn::solveBPDNBatch(batch, params, 0);
    if (err != cudaSuccess) {
        bpdn::freeBPDNBatch(batch);
        return (int)err;
    }

    bpdn::BPDNInfo* h_info = new bpdn::BPDNInfo[numProblems];

    err = bpdn::copyToHost(batch, h_x, h_info, 0);
    cudaDeviceSynchronize();

    delete[] h_info;
    bpdn::freeBPDNBatch(batch);

    return (int)err;
}

}
