#include <cstdio>
#include <cuda_runtime.h>
#include <unistd.h>
#include <iostream>
#include <random>
#include <iterator>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

// ---------------------------------------------------------------------------
// Kernel: find all runs of 1's using warp ballot intrinsics
//
// Each run is stored as a single int2: {.x = start, .y = end} (inclusive).
// Only start-detecting threads claim output slots and write both start and
// end in a single coalesced STG.64 store.
// ---------------------------------------------------------------------------
__global__ void find_runs_of_ones(const int* __restrict__ data,
                                  int    N,
                                  int2*  __restrict__ runs,        // output: (start, end) pairs
                                  int*   __restrict__ d_num_runs)  // output: total count (init to 0)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // --- Step 1: Detect rising edge (start of a run of 1's) ----------------
    bool is_start = false;
    int  my_end   = -1;

    // Ballot MUST be outside conditionals — all 32 lanes must participate
    unsigned warp_mask = __ballot_sync(0xFFFFFFFF, (tid < N) ? data[tid] : 0);

    if (tid < N && data[tid] == 1) {
        if (tid == 0 || data[tid - 1] == 0) {
            is_start = true;

            // --- Step 2: Find end of run, fast path via ballot -------------
            unsigned zeros_above = ~warp_mask & ~((1u << (lane + 1)) - 1);

            if (zeros_above) {
                int first_zero_lane = __ffs(zeros_above) - 1;
                my_end = tid - lane + first_zero_lane - 1;
            } else {
                // Run extends past warp boundary — scan global memory
                int e = (tid | 31);
                while (e + 1 < N && data[e + 1] == 1) {
                    e++;
                }
                my_end = e;
            }
        }
    }

    // --- Step 3: Intra-warp compaction via ballot --------------------------
    unsigned start_ballot = __ballot_sync(0xFFFFFFFF, is_start);
    int my_rank    = __popc(start_ballot & ((1u << lane) - 1));
    int warp_count = __popc(start_ballot);

    // --- Step 4: Block-level prefix sum across warps -----------------------
    __shared__ int s_warp_offsets[WARPS_PER_BLOCK];
    __shared__ int s_block_base;

    if (lane == 0) {
        s_warp_offsets[warp_id] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            int c = s_warp_offsets[w];
            s_warp_offsets[w] = total;
            total += c;
        }
        if (total > 0) {
            s_block_base = atomicAdd(d_num_runs, total);
        } else {
            s_block_base = 0;
        }
    }
    __syncthreads();

    // --- Step 5: Single coalesced int2 store -------------------------------
    if (is_start) {
        int out_idx = s_block_base + s_warp_offsets[warp_id] + my_rank;
        runs[out_idx] = make_int2(tid, my_end);
    }
}

// ---------------------------------------------------------------------------
// Sort kernel (single-thread insertion sort — fine for small output)
// ---------------------------------------------------------------------------
__global__ void sort_runs(int2* __restrict__ runs, int n)
{
    for (int i = 1; i < n; i++) {
        int2 tmp = runs[i];
        int j = i - 1;
        while (j >= 0 && runs[j].x > tmp.x) {
            runs[j + 1] = runs[j];
            j--;
        }
        runs[j + 1] = tmp;
    }
}

void generate_clumpy(int* data, int N, float p_start, float p_stay) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    data[0] = (dist(rng) < p_start) ? 1 : 0;
    for (int i = 1; i < N; i++) {
        if (data[i - 1] == 0)
            data[i] = (dist(rng) < p_start) ? 1 : 0;
        else
            data[i] = (dist(rng) < p_stay) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int N(200);
    float p_start(0.02f);
    float p_stay(0.95f);
    bool verbose(false);

    int opt;
    while ((opt = getopt(argc, argv, "n:v")) != -1)
    {
        switch (opt)
        {
            case 'n':
                N = atoi(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << std::endl
                          << " [-n <Num segments>]\n";
                return EXIT_FAILURE;
        }
    }
    //                 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
    //int h_data[] = {   0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 };
    //  Expected runs:  (1,3)          (6,7)   (9,9)  (11,14)
    //const int N = sizeof(h_data) / sizeof(h_data[0]);
    int *h_data = new int[N];
    generate_clumpy(h_data, N, p_start, p_stay);
    if (verbose) {
        std::copy(h_data, h_data + N - 1, std::ostream_iterator<int>(std::cout, ""));
        std::cout << std::endl;
    }

    // --- Device allocations ------------------------------------------------
    int  *d_data, *d_num_runs;
    int2 *d_runs;
    cudaMalloc(&d_data,     N * sizeof(int));
    cudaMalloc(&d_runs,     (N / 2 + 1) * sizeof(int2));   // worst case: N/2 runs
    cudaMalloc(&d_num_runs, sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_num_runs, 0, sizeof(int));

    // --- Launch ------------------------------------------------------------
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_runs_of_ones<<<grid, BLOCK_SIZE>>>(d_data, N, d_runs, d_num_runs);

    int h_num_runs = 0;
    cudaMemcpy(&h_num_runs, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost);

    if (grid > 1 && h_num_runs > 0) {
        sort_runs<<<1, 1>>>(d_runs, h_num_runs);
    }

    // --- Retrieve results --------------------------------------------------
    int2* h_runs = new int2[h_num_runs];
    cudaMemcpy(h_runs, d_runs, h_num_runs * sizeof(int2), cudaMemcpyDeviceToHost);

    // --- Print -------------------------------------------------------------
    printf("Found %d runs of 1's:\n", h_num_runs);
    for (int i = 0; i < h_num_runs; i++) {
        printf("  Run %d: [%d, %d]  (length %d)\n",
               i, h_runs[i].x, h_runs[i].y, h_runs[i].y - h_runs[i].x + 1);
    }

    // --- Cleanup -----------------------------------------------------------
    delete[] h_runs;
    cudaFree(d_data);
    cudaFree(d_runs);
    cudaFree(d_num_runs);

    return 0;
}
