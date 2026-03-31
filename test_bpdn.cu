#include "bpdn_cuda.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

/**
 * Test program for CUDA BPDN implementation
 *
 * Tests:
 * 1. Single small problem (known solution)
 * 2. Batch of random problems
 * 3. Performance test with ~1500 problems
 */

// Generate a sparse signal
void generateSparseSignal(float* x, int n, int sparsity, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> idxDist(0, n - 1);

    std::fill(x, x + n, 0.0f);

    for (int i = 0; i < sparsity; i++) {
        int idx = idxDist(gen);
        x[idx] = dist(gen);
    }
}

// Generate random matrix
void generateRandomMatrix(float* A, int m, int n, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)m));

    for (int i = 0; i < m * n; i++) {
        A[i] = dist(gen);
    }
}

// Compute Ax
void matVecHost(const float* A, const float* x, float* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            y[i] += A[j * m + i] * x[j];
        }
    }
}

// Compute L1 norm
float norm1Host(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(x[i]);
    }
    return sum;
}

// Compute L2 norm
float norm2Host(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrtf(sum);
}

// Test 1: Single problem with known sparse solution
bool testSingleProblem() {
    std::cout << "Test 1: Single problem with known sparse solution\n";

    const int m = 50;   // measurements
    const int n = 100;  // unknowns
    const int k = 5;    // sparsity

    std::mt19937 gen(42);

    // Generate sparse true signal
    std::vector<float> x_true(n, 0.0f);
    generateSparseSignal(x_true.data(), n, k, gen);

    // Generate random measurement matrix
    std::vector<float> A(m * n);
    generateRandomMatrix(A.data(), m, n, gen);

    // Generate measurements with small noise
    std::vector<float> b(m);
    matVecHost(A.data(), x_true.data(), b.data(), m, n);

    float noiseLevel = 0.01f * norm2Host(b.data(), m);
    std::normal_distribution<float> noiseDist(0.0f, noiseLevel / sqrtf((float)m));
    for (int i = 0; i < m; i++) {
        b[i] += noiseDist(gen);
    }

    // Set up BPDN batch
    bpdn::BPDNBatch batch;
    batch.numProblems = 1;
    batch.m = m;
    batch.n = n;
    batch.uniformSigma = true;

    cudaError_t err = bpdn::allocateBPDNBatch(batch);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate batch: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    float sigma = noiseLevel;  // Set sigma to noise level
    err = bpdn::copyToDevice(batch, A.data(), b.data(), &sigma, nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy to device: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    bpdn::BPDNParams params;
    params.sigma = sigma;
    params.maxIterations = 500;
    params.optTol = 1e-5f;

    err = bpdn::solveBPDNBatch(batch, params, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to solve: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    // Copy results back
    std::vector<float> x_sol(n);
    bpdn::BPDNInfo info;
    err = bpdn::copyToHost(batch, x_sol.data(), &info, 0);
    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "Failed to copy results: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    bpdn::freeBPDNBatch(batch);

    // Compute actual residual
    std::vector<float> r(m);
    matVecHost(A.data(), x_sol.data(), r.data(), m, n);
    for (int i = 0; i < m; i++) {
        r[i] = b[i] - r[i];
    }
    float actualRNorm = norm2Host(r.data(), m);

    // Compute recovery error
    float recoveryError = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x_sol[i] - x_true[i];
        recoveryError += diff * diff;
    }
    recoveryError = sqrtf(recoveryError) / norm2Host(x_true.data(), n);

    std::cout << "  Iterations: " << info.iterations << "\n";
    std::cout << "  Status: " << info.status << "\n";
    std::cout << "  Final residual norm: " << actualRNorm << " (target: " << sigma << ")\n";
    std::cout << "  Solution L1 norm: " << norm1Host(x_sol.data(), n) << "\n";
    std::cout << "  True signal L1 norm: " << norm1Host(x_true.data(), n) << "\n";
    std::cout << "  Relative recovery error: " << recoveryError << "\n";

    bool pass = (actualRNorm <= 1.5f * sigma) && (recoveryError < 0.5f);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";

    return pass;
}

// Test 2: Batch of problems
bool testBatch() {
    std::cout << "Test 2: Batch of problems\n";

    const int numProblems = 100;
    const int m = 60;
    const int n = 120;
    const int k = 8;

    std::mt19937 gen(123);

    // Generate batch data
    std::vector<float> A(numProblems * m * n);
    std::vector<float> b(numProblems * m);
    std::vector<float> x_true(numProblems * n);
    std::vector<float> sigmas(numProblems);

    for (int p = 0; p < numProblems; p++) {
        generateSparseSignal(x_true.data() + p * n, n, k, gen);
        generateRandomMatrix(A.data() + p * m * n, m, n, gen);
        matVecHost(A.data() + p * m * n, x_true.data() + p * n, b.data() + p * m, m, n);

        // Add noise and compute sigma for this problem
        float bNorm = norm2Host(b.data() + p * m, m);
        float noiseStd = 0.01f * bNorm / sqrtf((float)m);
        std::normal_distribution<float> noiseDist(0.0f, noiseStd);
        float noiseNormSq = 0.0f;
        for (int i = 0; i < m; i++) {
            float noise = noiseDist(gen);
            b[p * m + i] += noise;
            noiseNormSq += noise * noise;
        }
        // Set sigma to slightly above actual noise level
        sigmas[p] = 1.2f * sqrtf(noiseNormSq);
    }

    // Set up batch with per-problem sigma
    bpdn::BPDNBatch batch;
    batch.numProblems = numProblems;
    batch.m = m;
    batch.n = n;
    batch.uniformSigma = false;

    cudaError_t err = bpdn::allocateBPDNBatch(batch);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate batch: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    err = bpdn::copyToDevice(batch, A.data(), b.data(), sigmas.data(), nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy to device: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    bpdn::BPDNParams params;
    params.sigma = 0.0f;  // Per-problem sigma is used
    params.maxIterations = 1000;
    params.optTol = 1e-3f;

    auto start = std::chrono::high_resolution_clock::now();
    err = bpdn::solveBPDNBatch(batch, params, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (err != cudaSuccess) {
        std::cerr << "Failed to solve: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Copy results
    std::vector<float> x_sol(numProblems * n);
    std::vector<bpdn::BPDNInfo> infos(numProblems);
    err = bpdn::copyToHost(batch, x_sol.data(), infos.data(), 0);
    cudaDeviceSynchronize();

    bpdn::freeBPDNBatch(batch);

    // Compute statistics
    int numConverged = 0;
    float avgIter = 0.0f;
    float avgRecoveryError = 0.0f;

    for (int p = 0; p < numProblems; p++) {
        if (infos[p].status == bpdn::BPDNInfo::STATUS_ROOT_FOUND ||
            infos[p].status == bpdn::BPDNInfo::STATUS_OPTIMAL) {
            numConverged++;
        }
        avgIter += infos[p].iterations;

        float perr = 0.0f;
        float trueNorm = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = x_sol[p * n + i] - x_true[p * n + i];
            perr += diff * diff;
            trueNorm += x_true[p * n + i] * x_true[p * n + i];
        }
        avgRecoveryError += sqrtf(perr) / sqrtf(trueNorm + 1e-10f);
    }
    avgIter /= numProblems;
    avgRecoveryError /= numProblems;

    std::cout << "  Problems: " << numProblems << "\n";
    std::cout << "  Converged: " << numConverged << " (" << (100.0f * numConverged / numProblems) << "%)\n";
    std::cout << "  Average iterations: " << avgIter << "\n";
    std::cout << "  Average recovery error: " << avgRecoveryError << "\n";
    std::cout << "  Total time: " << elapsed << " ms\n";
    std::cout << "  Time per problem: " << (elapsed / numProblems) << " ms\n";

    // Pass if most problems converge and recovery error is reasonable
    bool pass = (numConverged >= 0.8 * numProblems) && (avgRecoveryError < 0.1f);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";

    return pass;
}

// Test 3: Performance test with ~1500 problems
bool testPerformance() {
    std::cout << "Test 3: Performance test (1500 problems)\n";

    const int numProblems = 1500;
    const int m = 150;   // More measurements for better conditioning
    const int n = 250;
    const int k = 12;    // Slightly sparser

    std::mt19937 gen(456);

    // Generate batch data
    std::vector<float> A(numProblems * m * n);
    std::vector<float> b(numProblems * m);
    std::vector<float> sigmas(numProblems);

    for (int p = 0; p < numProblems; p++) {
        // Generate sparse signal
        std::vector<float> x_temp(n, 0.0f);
        generateSparseSignal(x_temp.data(), n, k, gen);

        // Generate matrix and measurements
        generateRandomMatrix(A.data() + p * m * n, m, n, gen);
        matVecHost(A.data() + p * m * n, x_temp.data(), b.data() + p * m, m, n);

        // Add noise and compute per-problem sigma
        float bNorm = norm2Host(b.data() + p * m, m);
        float noiseStd = 0.02f * bNorm / sqrtf((float)m);
        std::normal_distribution<float> noiseDist(0.0f, noiseStd);
        float noiseNormSq = 0.0f;
        for (int i = 0; i < m; i++) {
            float noise = noiseDist(gen);
            b[p * m + i] += noise;
            noiseNormSq += noise * noise;
        }
        sigmas[p] = 1.2f * sqrtf(noiseNormSq);
    }

    // Set up batch with per-problem sigma
    bpdn::BPDNBatch batch;
    batch.numProblems = numProblems;
    batch.m = m;
    batch.n = n;
    batch.uniformSigma = false;

    cudaError_t err = bpdn::allocateBPDNBatch(batch);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate batch: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    err = bpdn::copyToDevice(batch, A.data(), b.data(), sigmas.data(), nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy to device: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    bpdn::BPDNParams params;
    params.sigma = 0.0f;  // Per-problem sigma
    params.maxIterations = 3000;  // More iterations for larger problems
    params.optTol = 5e-3f;        // Relaxed tolerance

    // Warmup
    err = bpdn::solveBPDNBatch(batch, params, 0);
    cudaDeviceSynchronize();

    // Reset x to zero for timing run
    cudaMemset(batch.d_x, 0, numProblems * n * sizeof(float));

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    err = bpdn::solveBPDNBatch(batch, params, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (err != cudaSuccess) {
        std::cerr << "Failed to solve: " << cudaGetErrorString(err) << "\n";
        bpdn::freeBPDNBatch(batch);
        return false;
    }

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Copy results
    std::vector<float> x_sol(numProblems * n);
    std::vector<bpdn::BPDNInfo> infos(numProblems);
    err = bpdn::copyToHost(batch, x_sol.data(), infos.data(), 0);
    cudaDeviceSynchronize();

    bpdn::freeBPDNBatch(batch);

    // Compute statistics
    int numConverged = 0;
    float avgIter = 0.0f;
    int maxIter = 0;
    int minIter = params.maxIterations;

    for (int p = 0; p < numProblems; p++) {
        if (infos[p].status == bpdn::BPDNInfo::STATUS_ROOT_FOUND ||
            infos[p].status == bpdn::BPDNInfo::STATUS_OPTIMAL) {
            numConverged++;
        }
        avgIter += infos[p].iterations;
        maxIter = std::max(maxIter, infos[p].iterations);
        minIter = std::min(minIter, infos[p].iterations);
    }
    avgIter /= numProblems;

    std::cout << "  Problem size: " << m << " x " << n << "\n";
    std::cout << "  Problems: " << numProblems << "\n";
    std::cout << "  Converged: " << numConverged << " (" << (100.0f * numConverged / numProblems) << "%)\n";
    std::cout << "  Iterations (avg/min/max): " << avgIter << "/" << minIter << "/" << maxIter << "\n";
    std::cout << "  Total time: " << elapsed << " ms\n";
    std::cout << "  Throughput: " << (numProblems / (elapsed / 1000.0)) << " problems/sec\n";
    std::cout << "  Time per problem: " << (elapsed / numProblems) << " ms\n";

    // Pass if most problems converge within reasonable time
    bool pass = (numConverged >= 0.7 * numProblems) && (elapsed < 120000);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";

    return pass;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "CUDA BPDN Implementation Tests\n";
    std::cout << "========================================\n\n";

    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "Using GPU: " << props.name << "\n";
    std::cout << "Compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Shared memory per block: " << props.sharedMemPerBlock << " bytes\n\n";

    int passed = 0;
    int total = 0;

    total++;
    if (testSingleProblem()) passed++;

    total++;
    if (testBatch()) passed++;

    total++;
    if (testPerformance()) passed++;

    std::cout << "========================================\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
