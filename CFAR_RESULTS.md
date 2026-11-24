# CA-CFAR vs OS-CFAR Benchmark Results

Benchmark: `benchmark_cfar.cu` (`./build/Release/benchmark_cfar`)

## Setup

- **GPU:** NVIDIA GB10 (SM 12.1), CUDA 13.0, Release build
- **Data:** 10 beams × 1,048,576 samples (float), exponential noise plus 100 injected targets per beam
- **Stencil:** 3 guard cells per side; reference cells per side swept from 7 to 31 in steps of 4
- **Threshold:** `alpha = 5.0` for both detectors
- **OS rank:** chosen automatically so the OS-CFAR theoretical false-alarm rate matches CA-CFAR's
- **Timing:** 11 iterations per kernel, first discarded; values are the average of the remaining 10
- **Correctness:** checked against a CPU reference at 8 and 11 reference cells per side (0 mismatches); the sweep itself ran with `--no-verify`

Command used:

```bash
for r in 7 11 15 19 23 27 31; do ./build/Release/benchmark_cfar --ref $r --no-verify; done
```

## Results

| Ref/side (total) | OS rank k | CA ms | OS ms | OS/CA | CA detections | OS detections |
|---|---|---|---|---|---|---|
| 7 (14)  | 10 | 0.401 | 0.722  | 1.8×  | 146,311 | 114,322 |
| 11 (22) | 15 | 0.403 | 1.514  | 3.8×  | 116,141 | 103,424 |
| 15 (30) | 20 | 0.404 | 2.691  | 6.7×  | 103,038 | 97,777  |
| 19 (38) | 25 | 0.412 | 4.308  | 10.4× | 95,967  | 94,212  |
| 23 (46) | 30 | 0.422 | 6.291  | 14.9× | 91,161  | 91,560  |
| 27 (54) | 35 | 0.429 | 8.530  | 19.9× | 88,104  | 89,833  |
| 31 (62) | 40 | 0.442 | 12.799 | 29.0× | 85,647  | 88,266  |

## Observations

- **CA-CFAR barely changes** (0.40 to 0.44 ms across the range). It is limited by memory bandwidth (214–236 GB/s effective); extra reference cells are read from shared memory and add almost nothing.
- **OS-CFAR slows roughly with the square of the window size.** Each output compares every pair of reference cells, so going from 14 to 62 cells is about 20× more work and took about 18× longer. Beyond about 7 cells per side it is compute-bound.
- **The step from 27 to 31 cells per side is larger than the trend predicts** (1.5× vs. about 1.3× expected). Register spilling at a 62-cell window is the likely cause, but this has not been confirmed.
- **Detection counts stay close.** The automatic rank keeps OS detections within a few percent of CA from 15 cells per side upward. At 7 per side the gap is about 22% because, with only 14 cells, the available ranks are spaced too far apart to match closely.

## Possible next step

For large windows, a selection method whose cost grows more slowly than all-pairs comparison (for example, an incremental sliding-window order statistic) would reduce the OS-CFAR cost.
