#include <cuda_runtime.h>
#include <iostream>
#include "segment_merger.h"

// Assuming all segments are sorted, we can easily calculate the distance between them, and mark
// them for merging.
__global__ void get_segment_distances(const Segment* __restrict__ segments,
                                      int segment_count,
                                      float distance_threshold,
                                      int* below_threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < segment_count - 1)
    {
        if (segments[tid + 1].start - segments[tid].end < distance_threshold)
        {
            below_threshold[tid] = 1;
        }
        else
        {
            below_threshold[tid] = 0;
        }
    }
}
// So each element of the above indicates whether or not to merge with segment to right.
// Length is N-1...
// 11100101 // N=9
//     Considering segments i, i+1, if merge[i]==1, set seg[i].end = seg[i+1].end
// 1 1 0 0
//     Considering segments i, i+2, if merge[i]==1, set seg[i].end = seg[i+2].end
/*
__global__ void merge_kernel_1(Segment* segments, int* segment_count, int threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int count(0);
    if(tid < segment_count)
    {
        if((tid == 0) || ((tid > 0) && (segments[tid].start - segments[tid-1].end > threshold)))
        {
            // This is the left side of a segment that will be kept
            Segment mrgSeg = segments[tidx];
        }
    }
}
*/
__global__ void parallel_merge_kernel(Segment* segments, int* segment_count, int threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid != 0) return; // Only one thread processes the entire array
    
    int n = *segment_count;
    if (n <= 1) return;
    
    int write_idx = 0;
    
    // Process all segments sequentially in one thread
    for (int read_idx = 0; read_idx < n; read_idx++) {
        if (read_idx == 0) {
            // Always keep the first segment
            segments[write_idx] = segments[read_idx];
        } else {
            // Check if current segment can be merged with the last written segment
            if (segments[read_idx].start - segments[write_idx].end <= threshold) {
                // Merge: extend the end of the current merged segment
                segments[write_idx].end = segments[read_idx].end;
            } else {
                // Cannot merge: start a new segment
                write_idx++;
                segments[write_idx] = segments[read_idx];
            }
        }
    }
    
    *segment_count = write_idx + 1;
}

__global__ void parallel_merge_with_streams_kernel(Segment* segments, int* segment_count, 
                                                  int threshold, int* temp_segments_data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid != 0) return; // Only one thread processes
    
    int n = *segment_count;
    if (n <= 1) return;
    
    // Use temporary space for intermediate results
    Segment* temp_segments = (Segment*)temp_segments_data;
    
    // Multiple passes approach for better parallelization potential
    bool changed = true;
    int current_n = n;
    int pass = 0;
    const int max_passes = n; // Safety limit
    
    while (changed && pass < max_passes) {
        changed = false;
        int write_idx = 0;
        
        // Copy first segment
        temp_segments[0] = segments[0];
        
        for (int i = 1; i < current_n; i++) {
            // Check if current segment can be merged with the last written segment
            if (segments[i].start - temp_segments[write_idx].end <= threshold) {
                // Merge: extend the end
                temp_segments[write_idx].end = segments[i].end;
                changed = true;
            } else {
                // Cannot merge: start new segment
                write_idx++;
                temp_segments[write_idx] = segments[i];
            }
        }
        
        current_n = write_idx + 1;
        
        // Copy result back to main array
        for (int i = 0; i < current_n; i++) {
            segments[i] = temp_segments[i];
        }
        
        pass++;
    }
    
    *segment_count = current_n;
}

extern "C" void merge_segments(Segment* segments, int* segment_count, int threshold) {
    if (*segment_count <= 1) return;
    
    int n = *segment_count;
    Segment* d_segments;
    int* d_segment_count;
    int* d_temp_segments_data;
    int* d_to_merge;
    int* to_merge;
    int in_segment_count(*segment_count);

    // Allocate device memory
    cudaMalloc(&d_segments, n * sizeof(Segment));
    cudaMalloc(&d_segment_count, sizeof(int));
    cudaMalloc(&d_temp_segments_data, n * sizeof(Segment)); // Temporary space
    cudaMalloc(&d_to_merge, n * sizeof(int));
    cudaMallocHost(&to_merge, n * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_segments, segments, n * sizeof(Segment), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segment_count, segment_count, sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 threads(256);
    dim3 grid((in_segment_count + threads.x - 1)/threads.x);
    get_segment_distances<<<grid, threads>>>(d_segments, in_segment_count, threshold, d_to_merge);

    // Launch kernel - single thread approach that's CUDA-compatible
    parallel_merge_with_streams_kernel<<<1, 1>>>(d_segments, d_segment_count, threshold, d_temp_segments_data);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(segment_count, d_segment_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(segments, d_segments, *segment_count * sizeof(Segment), cudaMemcpyDeviceToHost);
    cudaMemcpy(to_merge, d_to_merge, in_segment_count * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i < in_segment_count-1; i++) {
        std::cout << to_merge[i];
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_segments);
    cudaFree(d_segment_count);
    cudaFree(d_temp_segments_data);
    cudaFree(d_to_merge);
    cudaFreeHost(to_merge);
}