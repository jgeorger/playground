// Benchmark: Cell-Averaging CFAR vs Order-Statistic CFAR, 1D per beam.
//
// Input layout: d_in[num_beams][num_samples] (row-major, float, power domain).
//
// Stencil for cell under test (CUT) at sample x, symmetric about the CUT:
//
//   [ N ref ][ G guard ][ CUT ][ G guard ][ N ref ]
//
//   left  reference cells: x-G-N .. x-G-1
//   right reference cells: x+G+1 .. x+G+N
//
// Reference cells that fall outside [0, num_samples) are excluded, so the
// estimate near the row edges is built from fewer (one-sided) cells.
//
//   CA-CFAR noise  = mean   of the valid reference cells
//   OS-CFAR noise  = k-th smallest of the valid reference cells
//
// OS rank selection: by default k is chosen so the OS-CFAR false-alarm rate
// matches CA-CFAR for the same alpha in exponential (square-law) noise:
//
//   Pfa_CA = (1 + alpha/M)^-M
//   Pfa_OS = prod_{i=0}^{k-1} (M - i) / (M - i + alpha),     M = 2N
//
// and k (1-based, out of M) is the rank minimising |log Pfa_OS - log Pfa_CA|.
// It can be overridden with --os-rank. Near the row edges, where only V < M
// cells are valid, the 0-based rank is scaled to floor((k-1) * V / M).
//
// Both kernels write threshold[b][x] = alpha * noise and a uint8 detection
// mask det[b][x] = (x_cut > threshold), so their memory traffic is identical
// and the timing difference reflects the cost of the noise estimator.
//
// Usage:
//   benchmark_cfar [--beams B] [--samples S] [--ref N] [--guard G]
//                  [--iters I] [--alpha A] [--os-rank K] [--no-verify]

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int BLOCK   = 256;  // threads (= output samples) per block
static constexpr int MAX_REF = 32;   // max reference cells per side for OS-CFAR

// ---------------------------------------------------------------------------
// Shared-memory tile load: BLOCK outputs plus a halo of (G+N) on each side.
// Out-of-range samples are stored as 0 (CA) – validity is tracked separately
// by index, so the fill value is irrelevant to correctness.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void load_tile(float* tile, const float* __restrict__ row,
               int tile_start, int tile_len, int num_samples) {
    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        const int g = tile_start + i;
        tile[i] = (g >= 0 && g < num_samples) ? row[g] : 0.0f;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// CA-CFAR: runtime N and G. Each thread sums its 2N reference cells from
// shared memory (out-of-range halo cells were loaded as 0, so they add
// nothing; we only need the count of valid cells for the divisor).
// ---------------------------------------------------------------------------
__global__
void ca_cfar_kernel(const float* __restrict__ d_in,
                    float* __restrict__ d_thresh,
                    uint8_t* __restrict__ d_det,
                    int num_samples, int num_ref, int num_guard, float alpha) {
    extern __shared__ float tile[];

    const int beam = blockIdx.y;
    const size_t row_off = (size_t)beam * num_samples;
    const float* row = d_in + row_off;

    const int halo       = num_ref + num_guard;
    const int block_x0   = blockIdx.x * blockDim.x;
    const int tile_start = block_x0 - halo;
    const int tile_len   = blockDim.x + 2 * halo;

    load_tile(tile, row, tile_start, tile_len, num_samples);

    const int x = block_x0 + threadIdx.x;
    if (x >= num_samples) return;

    const int c = threadIdx.x + halo;  // CUT position in tile
    float sum = 0.0f;
    for (int k = 1; k <= num_ref; ++k) {
        sum += tile[c - num_guard - k];
        sum += tile[c + num_guard + k];
    }

    // Count of in-range reference cells.
    const int left_lo  = max(x - num_guard - num_ref, 0);
    const int left_hi  = x - num_guard - 1;                   // inclusive
    const int right_lo = x + num_guard + 1;
    const int right_hi = min(x + num_guard + num_ref, num_samples - 1);
    const int valid = max(left_hi - left_lo + 1, 0) + max(right_hi - right_lo + 1, 0);

    const float noise = valid > 0 ? sum / (float)valid : 0.0f;
    const float thr   = alpha * noise;
    d_thresh[row_off + x] = thr;
    d_det[row_off + x]    = tile[c] > thr ? 1 : 0;
}

// ---------------------------------------------------------------------------
// OS-CFAR (k-th order statistic). N is a template parameter so the 2N-cell window lives
// in registers. Selection is by rank counting with index tie-breaking:
//
//   rank_i = #{ j < i : w[j] <= w[i] } + #{ j > i : w[j] < w[i] }
//
// which gives every element a unique rank 0..W-1. Each unordered pair
// (i, j) is compared once, so the cost is W(W-1)/2 compares, fully
// unrolled and branch-free. Missing (out-of-range) cells are +INF, which
// pushes them to the top ranks; with V valid cells we select the element
// of rank floor(rank0 * V / W), where rank0 is the 0-based full-window rank.
// ---------------------------------------------------------------------------
template <int N>
__global__
void os_cfar_kernel(const float* __restrict__ d_in,
                    float* __restrict__ d_thresh,
                    uint8_t* __restrict__ d_det,
                    int num_samples, int num_guard, int rank0, float alpha) {
    extern __shared__ float tile[];
    constexpr int W = 2 * N;

    const int beam = blockIdx.y;
    const size_t row_off = (size_t)beam * num_samples;
    const float* row = d_in + row_off;

    const int halo       = N + num_guard;
    const int block_x0   = blockIdx.x * blockDim.x;
    const int tile_start = block_x0 - halo;
    const int tile_len   = blockDim.x + 2 * halo;

    load_tile(tile, row, tile_start, tile_len, num_samples);

    const int x = block_x0 + threadIdx.x;
    if (x >= num_samples) return;

    const int c = threadIdx.x + halo;
    float w[W];
    int valid = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        const int gl = x - num_guard - N + k;    // left cells, ascending
        const int gr = x + num_guard + 1 + k;    // right cells, ascending
        const bool vl = gl >= 0;
        const bool vr = gr < num_samples;
        w[k]     = vl ? tile[c - num_guard - N + k] : INFINITY;
        w[N + k] = vr ? tile[c + num_guard + 1 + k] : INFINITY;
        valid += (int)vl + (int)vr;
    }

    int rank[W];
#pragma unroll
    for (int i = 0; i < W; ++i) rank[i] = 0;
#pragma unroll
    for (int i = 0; i < W; ++i) {
#pragma unroll
        for (int j = i + 1; j < W; ++j) {
            const int lt = w[j] < w[i];   // j > i: counts toward rank_i if strictly less
            rank[i] += lt;
            rank[j] += 1 - lt;            // i < j: w[i] <= w[j] counts toward rank_j
        }
    }

    const int target = (rank0 * valid) / W;
    float kth = 0.0f;
#pragma unroll
    for (int i = 0; i < W; ++i) kth = (rank[i] == target) ? w[i] : kth;

    const float noise = valid > 0 ? kth : 0.0f;
    const float thr   = alpha * noise;
    d_thresh[row_off + x] = thr;
    d_det[row_off + x]    = tile[c] > thr ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
struct Params {
    int   beams   = 10;
    int   samples = 1 << 20;
    int   ref     = 11;
    int   guard   = 3;
    int   iters   = 11;   // first is warm-up and is discarded
    float alpha   = 5.0f;
    int   os_rank = 0;    // 1-based rank out of 2*ref; 0 = auto (match CA Pfa)
    bool  verify  = true;
};

static void launch_ca(const Params& p, const float* in, float* thr, uint8_t* det) {
    dim3 block(BLOCK);
    dim3 grid((p.samples + BLOCK - 1) / BLOCK, p.beams);
    size_t smem = (BLOCK + 2 * (p.ref + p.guard)) * sizeof(float);
    ca_cfar_kernel<<<grid, block, smem>>>(in, thr, det, p.samples, p.ref, p.guard, p.alpha);
}

template <int N>
static void launch_os_n(const Params& p, const float* in, float* thr, uint8_t* det) {
    dim3 block(BLOCK);
    dim3 grid((p.samples + BLOCK - 1) / BLOCK, p.beams);
    size_t smem = (BLOCK + 2 * (N + p.guard)) * sizeof(float);
    os_cfar_kernel<N><<<grid, block, smem>>>(in, thr, det, p.samples, p.guard,
                                              p.os_rank - 1, p.alpha);
}

using OsLauncher = void (*)(const Params&, const float*, float*, uint8_t*);

// Table of OS-CFAR launchers indexed by N-1, instantiated for N = 1..MAX_REF.
template <int... Is>
static OsLauncher os_launcher_impl(int n, std::integer_sequence<int, Is...>) {
    static const OsLauncher table[] = {&launch_os_n<Is + 1>...};
    return table[n - 1];
}
static OsLauncher os_launcher(int n) {
    return os_launcher_impl(n, std::make_integer_sequence<int, MAX_REF>{});
}

// Theoretical false-alarm probabilities in exponential noise (M cells).
static double pfa_ca(int M, double alpha) { return std::pow(1.0 + alpha / M, -M); }
static double pfa_os(int M, int k, double alpha) {
    double p = 1.0;
    for (int i = 0; i < k; ++i) p *= (double)(M - i) / (M - i + alpha);
    return p;
}
// 1-based OS rank whose Pfa is closest (in log) to CA-CFAR's at the same alpha.
static int match_os_rank(int M, double alpha) {
    const double target = std::log(pfa_ca(M, alpha));
    int best = 1;
    double best_err = INFINITY;
    for (int k = 1; k <= M; ++k) {
        const double err = std::fabs(std::log(pfa_os(M, k, alpha)) - target);
        if (err < best_err) { best_err = err; best = k; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------
static void gather_ref(const float* row, int S, int x, int N, int G, std::vector<float>& out) {
    out.clear();
    for (int k = 1; k <= N; ++k) {
        int l = x - G - k, r = x + G + k;
        if (l >= 0) out.push_back(row[l]);
        if (r < S)  out.push_back(row[r]);
    }
}

static bool verify(const Params& p, const std::vector<float>& h_in,
                   const std::vector<float>& ca_thr, const std::vector<float>& os_thr) {
    std::vector<float> cells;
    cells.reserve(2 * p.ref);
    size_t ca_bad = 0, os_bad = 0;
    double ca_max_rel = 0.0;
    for (int b = 0; b < p.beams; ++b) {
        const float* row = h_in.data() + (size_t)b * p.samples;
        for (int x = 0; x < p.samples; ++x) {
            const size_t idx = (size_t)b * p.samples + x;
            gather_ref(row, p.samples, x, p.ref, p.guard, cells);
            const int V = (int)cells.size();

            double sum = 0.0;
            for (float v : cells) sum += v;
            const double ca_ref = V ? p.alpha * sum / V : 0.0;
            const double rel = std::fabs(ca_thr[idx] - ca_ref) / std::max(std::fabs(ca_ref), 1e-12);
            ca_max_rel = std::max(ca_max_rel, rel);
            if (rel > 1e-4) ++ca_bad;

            float os_ref = 0.0f;
            if (V) {
                const int t = ((p.os_rank - 1) * V) / (2 * p.ref);
                std::nth_element(cells.begin(), cells.begin() + t, cells.end());
                os_ref = p.alpha * cells[t];
            }
            if (os_thr[idx] != os_ref) ++os_bad;
        }
    }
    std::printf("Verify: CA mismatches %zu (max rel err %.2e), OS mismatches %zu\n",
                ca_bad, ca_max_rel, os_bad);
    return ca_bad == 0 && os_bad == 0;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
template <typename F>
static float time_kernel(int iters, F&& launch) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        if (i > 0) total += ms;  // discard first (warm-up) iteration
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return (float)(total / (iters - 1));
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--beams B] [--samples S] [--ref N] [--guard G]\n"
        "          [--iters I] [--alpha A] [--os-rank K] [--no-verify]\n"
        "  --beams    number of beams (rows)              [default 10]\n"
        "  --samples  samples per beam (columns)          [default 1048576]\n"
        "  --ref      reference cells per side (1..%d)    [default 11]\n"
        "  --guard    guard cells per side (>= 0)         [default 3]\n"
        "  --iters    timed iterations incl. warm-up (>=2)[default 11]\n"
        "  --alpha    threshold scale factor              [default 5.0]\n"
        "  --os-rank  OS rank k, 1..2*ref (1 = smallest)  [default: auto, matches CA Pfa]\n"
        "  --no-verify  skip CPU reference check\n",
        prog, MAX_REF);
}

static Params parse_args(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", name); std::exit(1); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--beams"))     p.beams   = std::atoi(need("--beams"));
        else if (!std::strcmp(argv[i], "--samples"))   p.samples = std::atoi(need("--samples"));
        else if (!std::strcmp(argv[i], "--ref"))       p.ref     = std::atoi(need("--ref"));
        else if (!std::strcmp(argv[i], "--guard"))     p.guard   = std::atoi(need("--guard"));
        else if (!std::strcmp(argv[i], "--iters"))     p.iters   = std::atoi(need("--iters"));
        else if (!std::strcmp(argv[i], "--alpha"))     p.alpha   = (float)std::atof(need("--alpha"));
        else if (!std::strcmp(argv[i], "--os-rank"))   p.os_rank = std::atoi(need("--os-rank"));
        else if (!std::strcmp(argv[i], "--no-verify")) p.verify  = false;
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown argument: %s\n", argv[i]); usage(argv[0]); std::exit(1); }
    }
    if (p.beams < 1 || p.samples < 1 || p.ref < 1 || p.ref > MAX_REF ||
        p.guard < 0 || p.iters < 2 || p.os_rank < 0 || p.os_rank > 2 * p.ref) {
        std::fprintf(stderr, "Invalid parameters.\n");
        usage(argv[0]);
        std::exit(1);
    }
    if (p.beams > 65535) { std::fprintf(stderr, "beams must be <= 65535\n"); std::exit(1); }
    if (p.os_rank == 0) p.os_rank = match_os_rank(2 * p.ref, p.alpha);
    return p;
}

int main(int argc, char** argv) {
    const Params p = parse_args(argc, argv);
    const size_t n = (size_t)p.beams * p.samples;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Beams %d x samples %d (%.1f M cells), ref %d/side, guard %d/side, "
                "alpha %.2f, %d iters (first discarded)\n",
                p.beams, p.samples, n / 1e6, p.ref, p.guard, p.alpha, p.iters);
    std::printf("OS rank k = %d of %d; theoretical Pfa: CA %.3e, OS %.3e\n",
                p.os_rank, 2 * p.ref, pfa_ca(2 * p.ref, p.alpha),
                pfa_os(2 * p.ref, p.os_rank, p.alpha));

    // Square-law detected noise: exponential(1), plus sparse targets.
    std::vector<float> h_in(n);
    {
        std::mt19937 rng(12345);
        std::exponential_distribution<float> noise(1.0f);
        for (auto& v : h_in) v = noise(rng);
        std::uniform_int_distribution<size_t> pos(0, n - 1);
        for (int t = 0; t < p.beams * 100; ++t) h_in[pos(rng)] += 50.0f;
    }

    float *d_in, *d_thr_ca, *d_thr_os;
    uint8_t *d_det_ca, *d_det_os;
    CHECK_CUDA(cudaMalloc(&d_in,     n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_thr_ca, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_thr_os, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_det_ca, n));
    CHECK_CUDA(cudaMalloc(&d_det_os, n));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const OsLauncher launch_os = os_launcher(p.ref);

    const float ca_ms = time_kernel(p.iters, [&] { launch_ca(p, d_in, d_thr_ca, d_det_ca); });
    const float os_ms = time_kernel(p.iters, [&] { launch_os(p, d_in, d_thr_os, d_det_os); });

    // Effective bandwidth: read input once, write float threshold + uint8 mask.
    const double bytes = (double)n * (sizeof(float) + sizeof(float) + sizeof(uint8_t));
    std::printf("\n%-22s %12s %14s\n", "Kernel", "Avg ms", "Eff. GB/s");
    std::printf("%-22s %12.4f %14.1f\n", "CA-CFAR (mean)",   ca_ms, bytes / (ca_ms * 1e6));
    std::printf("%-22s %12.4f %14.1f\n", "OS-CFAR (k-th)", os_ms, bytes / (os_ms * 1e6));
    std::printf("OS / CA latency ratio: %.2fx\n\n", os_ms / ca_ms);

    std::vector<float> ca_thr(n), os_thr(n);
    std::vector<uint8_t> ca_det(n), os_det(n);
    CHECK_CUDA(cudaMemcpy(ca_thr.data(), d_thr_ca, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(os_thr.data(), d_thr_os, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ca_det.data(), d_det_ca, n, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(os_det.data(), d_det_os, n, cudaMemcpyDeviceToHost));
    size_t ca_hits = 0, os_hits = 0;
    for (size_t i = 0; i < n; ++i) { ca_hits += ca_det[i]; os_hits += os_det[i]; }
    std::printf("Detections: CA %zu, OS %zu (%d targets injected)\n",
                ca_hits, os_hits, p.beams * 100);

    int rc = 0;
    if (p.verify && !verify(p, h_in, ca_thr, os_thr)) rc = 1;

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_thr_ca));
    CHECK_CUDA(cudaFree(d_thr_os));
    CHECK_CUDA(cudaFree(d_det_ca));
    CHECK_CUDA(cudaFree(d_det_os));
    return rc;
}
