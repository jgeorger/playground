#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
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

static constexpr int NUM_ITERS = 11;   // first is warm-up, average the rest
static constexpr int MIN_TAPS  = 3;
static constexpr int MAX_TAPS  = 20;
static constexpr int COEFFS_PER_ROW  = MAX_TAPS;         // room for up to 20 taps
static constexpr int NUM_COEFF_ROWS  = MAX_TAPS + 1;     // indexed by num_taps

// Coefficient table for the binomial MTI canceller, indexed by num_taps.
// Row n (for n = 3..20) holds coefficients c_k = (-1)^k * C(n-1, k), k = 0..n-1.
// Rows 0..2 are unused (padded with zeros).
//   n=3 : [1, -2, 1]
//   n=4 : [1, -3, 3, -1]
//   ...
__constant__ float c_coeffs[NUM_COEFF_ROWS * COEFFS_PER_ROW];

// Multi-pulse canceller MTI filter, "same-mode" (output length == input length).
// Input:  d_in  [num_pulses × num_range_bins]  (pulse-major, per channel)
// Output: d_out [num_pulses × num_range_bins]  (pulse-major, per channel)
//
// Centered convolution:
//   output[p, r] = sum_{k=0..TAPS-1} coeff[k] * input[p + k - HALF_TAPS, r]
// with input treated as zero outside [0, num_pulses). TAPS is required
// to be odd, so HALF_TAPS = (TAPS - 1) / 2 taps hang off each side and
// the output timing lines up with the input timing.
//
// Optimizations:
//   * TAPS is a template parameter → inner loop is fully unrolled and
//     coefficients are broadcast from constant memory into registers.
//   * Each thread produces PPT consecutive output pulses using a shared
//     sliding window of PPT + TAPS - 1 loads (vs PPT × TAPS naively),
//     approaching the theoretical minimum of one global load per input.
//   * Outer accumulation loop is over taps, inner over outputs — the
//     window value at position i+k is broadcast into PPT MACs while the
//     coefficient sits in a register.
template <int TAPS, int PPT>
__global__
void mti_kernel_opt(const cuComplex* __restrict__ d_in,
                    cuComplex* __restrict__ d_out,
                    int num_range_bins,
                    int num_pulses,
                    size_t in_channel_stride,
                    size_t out_channel_stride) {
    const int channel = blockIdx.z;
    const cuComplex* in_ch  = d_in  + channel * in_channel_stride;
    cuComplex*       out_ch = d_out + channel * out_channel_stride;

    const int r     = blockIdx.x * blockDim.x + threadIdx.x;
    const int p_out = (blockIdx.y * blockDim.y + threadIdx.y) * PPT;
    if (r >= num_range_bins || p_out >= num_pulses) return;

    // Index arithmetic within a channel uses size_t so a single channel
    // may exceed 2 GiB of elements without wrapping around.
    const size_t rb64 = (size_t)num_range_bins;

    constexpr int HALF_TAPS = (TAPS - 1) / 2;   // TAPS is guaranteed odd
    constexpr int W = PPT + TAPS - 1;
    const int load_base = p_out - HALF_TAPS;    // first input index needed

    float win_re[W];
    float win_im[W];

    // Fast path when the entire window fits in bounds (the common case:
    // any output away from the leading HALF_TAPS or trailing HALF_TAPS
    // pulses of the volume).
    if (load_base >= 0 && load_base + W <= num_pulses) {
        #pragma unroll
        for (int i = 0; i < W; ++i) {
            cuComplex v = in_ch[(size_t)(load_base + i) * rb64 + r];
            win_re[i] = v.x;
            win_im[i] = v.y;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < W; ++i) {
            int p = load_base + i;
            if ((unsigned)p < (unsigned)num_pulses) {   // handles both p<0 and p>=np
                cuComplex v = in_ch[(size_t)p * rb64 + r];
                win_re[i] = v.x;
                win_im[i] = v.y;
            } else {
                win_re[i] = 0.0f;
                win_im[i] = 0.0f;
            }
        }
    }

    float acc_re[PPT];
    float acc_im[PPT];
    #pragma unroll
    for (int i = 0; i < PPT; ++i) { acc_re[i] = 0.0f; acc_im[i] = 0.0f; }

    // Outer over taps, inner over outputs: window register used PPT times
    // per tap iteration; coefficient stays in a single register.
    #pragma unroll
    for (int k = 0; k < TAPS; ++k) {
        float c = c_coeffs[TAPS * COEFFS_PER_ROW + k];
        #pragma unroll
        for (int i = 0; i < PPT; ++i) {
            acc_re[i] += c * win_re[i + k];
            acc_im[i] += c * win_im[i + k];
        }
    }

    // Fast path when all PPT writes are in bounds.
    if (p_out + PPT <= num_pulses) {
        #pragma unroll
        for (int i = 0; i < PPT; ++i) {
            out_ch[(size_t)(p_out + i) * rb64 + r] =
                make_cuFloatComplex(acc_re[i], acc_im[i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < PPT; ++i) {
            int p = p_out + i;
            if (p < num_pulses) {
                out_ch[(size_t)p * rb64 + r] =
                    make_cuFloatComplex(acc_re[i], acc_im[i]);
            }
        }
    }
}

static constexpr int MTI_PPT = 8;

static void launch_mti(const cuComplex* d_in, cuComplex* d_out,
                       int num_channels, int num_range_bins, int num_pulses,
                       int num_taps) {
    // Same-mode convolution: one output pulse per input pulse.
    const size_t stride = (size_t)num_pulses * num_range_bins;
    dim3 block(32, 8);
    dim3 grid((num_range_bins + block.x - 1) / block.x,
              (num_pulses + block.y * MTI_PPT - 1) / (block.y * MTI_PPT),
              (unsigned)num_channels);

    #define MTI_DISPATCH(N)                                                    \
        case N:                                                                \
            mti_kernel_opt<N, MTI_PPT><<<grid, block>>>(                       \
                d_in, d_out, num_range_bins, num_pulses,                       \
                stride, stride);                                               \
            break

    switch (num_taps) {
        MTI_DISPATCH(3);  MTI_DISPATCH(5);  MTI_DISPATCH(7);  MTI_DISPATCH(9);
        MTI_DISPATCH(11); MTI_DISPATCH(13); MTI_DISPATCH(15); MTI_DISPATCH(17);
        MTI_DISPATCH(19);
        default: /* validated at caller */ break;
    }
    #undef MTI_DISPATCH
}

static void upload_coeff_table() {
    // Populate only odd tap counts. Even rows stay zero-filled and are
    // rejected before dispatch, so they can never be selected at runtime.
    std::vector<float> host_table(NUM_COEFF_ROWS * COEFFS_PER_ROW, 0.0f);
    for (int n = MIN_TAPS; n <= MAX_TAPS; n += 2) {
        int order = n - 1;   // polynomial order = n - 1
        long long binom = 1; // C(order, 0)
        for (int k = 0; k <= order; ++k) {
            float sign = (k & 1) ? -1.0f : 1.0f;
            host_table[n * COEFFS_PER_ROW + k] = sign * (float)binom;
            binom = binom * (order - k) / (k + 1);
        }
    }
    CHECK_CUDA(cudaMemcpyToSymbol(c_coeffs, host_table.data(),
                                  host_table.size() * sizeof(float)));
}

struct SweepParams {
    std::vector<int> num_channels;
    std::vector<int> num_range_bins;
    std::vector<int> num_pulses;
    std::vector<int> num_taps;
};

// Minimal JSON reader for objects of the shape:
//   { "num_range_bins": [1024, 2048], "num_pulses": [16, 32], "num_taps": [3, 4, 5] }
// Extracts one integer array per named key. Not a full JSON parser; ignores
// nested objects and non-integer values.
static bool extract_int_array(const std::string& text, const std::string& key,
                              std::vector<int>& out) {
    std::string needle = "\"" + key + "\"";
    size_t p = text.find(needle);
    if (p == std::string::npos) return false;
    p += needle.size();
    while (p < text.size() && std::isspace((unsigned char)text[p])) ++p;
    if (p >= text.size() || text[p] != ':') return false;
    ++p;
    while (p < text.size() && std::isspace((unsigned char)text[p])) ++p;
    if (p >= text.size() || text[p] != '[') return false;
    ++p;
    while (p < text.size() && text[p] != ']') {
        while (p < text.size() && (std::isspace((unsigned char)text[p]) || text[p] == ',')) ++p;
        if (p >= text.size() || text[p] == ']') break;
        size_t start = p;
        while (p < text.size() && (std::isdigit((unsigned char)text[p]) ||
                                   text[p] == '-' || text[p] == '+')) ++p;
        if (p == start) { ++p; continue; }
        out.push_back(std::atoi(text.substr(start, p - start).c_str()));
    }
    return true;
}

static bool load_sweep_json(const std::string& path, SweepParams& sp) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Could not open sweep file: %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    extract_int_array(text, "num_channels", sp.num_channels);
    bool have_r = extract_int_array(text, "num_range_bins", sp.num_range_bins);
    bool have_p = extract_int_array(text, "num_pulses",     sp.num_pulses);
    bool have_t = extract_int_array(text, "num_taps",       sp.num_taps);

    if (sp.num_channels.empty()) sp.num_channels.push_back(1);  // default

    if (!have_r || sp.num_range_bins.empty() ||
        !have_p || sp.num_pulses.empty() ||
        !have_t || sp.num_taps.empty()) {
        std::fprintf(stderr,
                     "Sweep JSON must define non-empty arrays "
                     "\"num_range_bins\", \"num_pulses\", \"num_taps\" "
                     "(and optionally \"num_channels\").\n");
        return false;
    }
    return true;
}

static std::string sanitize(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) out.push_back(c);
        else if (c == ' ' || c == '-' || c == '_' || c == '.') out.push_back('_');
    }
    if (out.empty()) out = "unknown_gpu";
    return out;
}

static std::string make_output_filename(const std::string& base) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::string gpu = sanitize(prop.name);
    std::time_t now = std::time(nullptr);
    std::tm tm_local = *std::localtime(&now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_local);
    return base + "_" + ts + "_" + gpu + ".csv";
}

struct RunContext {
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;
    FILE* outFile;
};

static void emit_line(RunContext& ctx, const char* line, int len) {
    std::fwrite(line, 1, len, stdout);
    std::fflush(stdout);
    if (ctx.outFile) {
        std::fwrite(line, 1, len, ctx.outFile);
        std::fflush(ctx.outFile);
    }
}

static void emit_header(RunContext& ctx) {
    const char* hdr = "num_channels,num_range_bins,num_pulses,num_taps,avg_runtime_ms\n";
    emit_line(ctx, hdr, (int)std::strlen(hdr));
}

static void run_point(RunContext& ctx, int num_channels, int num_range_bins,
                      int num_pulses, int num_taps) {
    if (num_taps < MIN_TAPS || num_taps > MAX_TAPS) {
        std::fprintf(stderr,
                     "  skip: num_taps=%d outside supported range [%d, %d]\n",
                     num_taps, MIN_TAPS, MAX_TAPS);
        return;
    }
    if ((num_taps & 1) == 0) {
        std::fprintf(stderr,
                     "  skip: num_taps=%d must be odd\n", num_taps);
        return;
    }
    // grid.z is limited to 65535 on all current CUDA GPUs.
    if (num_channels < 1 || num_channels > 65535) {
        std::fprintf(stderr,
                     "  skip: num_channels=%d outside supported range [1, 65535]\n",
                     num_channels);
        return;
    }

    // Same-mode convolution — output shape matches input.
    size_t per_channel = (size_t)num_pulses * num_range_bins;
    size_t in_elems  = (size_t)num_channels * per_channel;
    size_t out_elems = in_elems;

    std::fprintf(stderr,
                 "Point: num_channels=%d num_range_bins=%d num_pulses=%d "
                 "num_taps=%d | mem in=%.2f out=%.2f MB\n",
                 num_channels, num_range_bins, num_pulses, num_taps,
                 (double)(in_elems  * sizeof(cuComplex)) / (1024.0 * 1024.0),
                 (double)(out_elems * sizeof(cuComplex)) / (1024.0 * 1024.0));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<cuComplex> h_in(in_elems);
    for (size_t i = 0; i < in_elems; ++i)
        h_in[i] = make_cuFloatComplex(dist(rng), dist(rng));

    cuComplex *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  in_elems  * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(cuComplex)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), in_elems * sizeof(cuComplex),
                          cudaMemcpyHostToDevice));

    double sum_ms = 0.0;
    int counted = 0;
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ctx.ev_start, 0));
        launch_mti(d_in, d_out, num_channels, num_range_bins, num_pulses,
                   num_taps);
        CHECK_CUDA(cudaEventRecord(ctx.ev_stop, 0));
        CHECK_CUDA(cudaEventSynchronize(ctx.ev_stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ctx.ev_start, ctx.ev_stop));
        if (iter > 0) {           // skip the first (warm-up) run
            sum_ms += ms;
            ++counted;
        }
    }
    float avg_ms = (float)(sum_ms / counted);

    char line[160];
    int len = std::snprintf(line, sizeof(line), "%d,%d,%d,%d,%.6f\n",
                            num_channels, num_range_bins, num_pulses,
                            num_taps, avg_ms);
    emit_line(ctx, line, len);

    cudaFree(d_in);
    cudaFree(d_out);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s <num_channels> <num_range_bins> <num_pulses> <num_taps>\n"
                 "  %s --sweep <file.json> [--out <base>]\n"
                 "\n"
                 "num_taps must be an odd integer in [%d, %d] (binomial\n"
                 "coefficients precomputed into constant memory at startup).\n"
                 "num_channels acts as a batch dimension: the input volume is\n"
                 "num_channels * num_pulses * num_range_bins complex samples,\n"
                 "and the same MTI filter is applied independently per channel.\n"
                 "Output has the same shape as the input; centered same-mode\n"
                 "convolution is used, with zero padding at the pulse edges.\n"
                 "\n"
                 "Sweep file JSON schema:\n"
                 "  {\n"
                 "    \"num_channels\":   [1, 4, 16],\n"
                 "    \"num_range_bins\": [1024, 2048, 4096],\n"
                 "    \"num_pulses\":     [16, 32, 64],\n"
                 "    \"num_taps\":       [3, 5, 7, 9]\n"
                 "  }\n"
                 "\"num_channels\" is optional (defaults to [1]).\n"
                 "When --sweep is given, positional arguments are ignored.\n"
                 "\n"
                 "Output CSV (also written to stdout):\n"
                 "  num_channels,num_range_bins,num_pulses,num_taps,avg_runtime_ms\n",
                 prog, prog, MIN_TAPS, MAX_TAPS);
}

int main(int argc, char** argv) {
    std::string sweep_path;
    std::string out_base = "mti_benchmark";
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--sweep" || a == "-s") && i + 1 < argc) {
            sweep_path = argv[++i];
        } else if ((a == "--out" || a == "-o") && i + 1 < argc) {
            out_base = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            positional.push_back(a);
        }
    }

    SweepParams sp;
    if (!sweep_path.empty()) {
        if (!load_sweep_json(sweep_path, sp)) return 1;
    } else {
        if (positional.size() != 4) {
            print_usage(argv[0]);
            return 1;
        }
        sp.num_channels.push_back(std::atoi(positional[0].c_str()));
        sp.num_range_bins.push_back(std::atoi(positional[1].c_str()));
        sp.num_pulses.push_back(std::atoi(positional[2].c_str()));
        sp.num_taps.push_back(std::atoi(positional[3].c_str()));
    }

    for (int v : sp.num_channels) if (v <= 0) {
        std::fprintf(stderr, "num_channels values must be positive\n"); return 1;
    }
    for (int v : sp.num_range_bins) if (v <= 0) {
        std::fprintf(stderr, "num_range_bins values must be positive\n"); return 1;
    }
    for (int v : sp.num_pulses) if (v <= 0) {
        std::fprintf(stderr, "num_pulses values must be positive\n"); return 1;
    }

    upload_coeff_table();

    std::string out_path = make_output_filename(out_base);
    FILE* outFile = std::fopen(out_path.c_str(), "w");
    if (!outFile) {
        std::fprintf(stderr, "Could not open output file: %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "Writing results to: %s\n", out_path.c_str());

    RunContext ctx;
    ctx.outFile = outFile;
    CHECK_CUDA(cudaEventCreate(&ctx.ev_start));
    CHECK_CUDA(cudaEventCreate(&ctx.ev_stop));

    emit_header(ctx);

    size_t total = sp.num_channels.size() * sp.num_range_bins.size() *
                   sp.num_pulses.size() * sp.num_taps.size();
    std::fprintf(stderr,
                 "Sweep: %zu combinations (num_channels=%zu, num_range_bins=%zu, "
                 "num_pulses=%zu, num_taps=%zu)\n",
                 total, sp.num_channels.size(), sp.num_range_bins.size(),
                 sp.num_pulses.size(), sp.num_taps.size());

    size_t idx = 0;
    for (int c : sp.num_channels)
        for (int r : sp.num_range_bins)
            for (int p : sp.num_pulses)
                for (int t : sp.num_taps) {
                    ++idx;
                    std::fprintf(stderr, "[%zu/%zu] ", idx, total);
                    run_point(ctx, c, r, p, t);
                }

    cudaEventDestroy(ctx.ev_start);
    cudaEventDestroy(ctx.ev_stop);
    std::fclose(outFile);
    return 0;
}
