#include <cuda_runtime.h>
#include <stdio.h>
#include "segment_merger.h"

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
    
    // Allocate device memory
    cudaMalloc(&d_segments, n * sizeof(Segment));
    cudaMalloc(&d_segment_count, sizeof(int));
    cudaMalloc(&d_temp_segments_data, n * sizeof(Segment)); // Temporary space
    
    // Copy input to device
    cudaMemcpy(d_segments, segments, n * sizeof(Segment), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segment_count, segment_count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel - single thread approach that's CUDA-compatible
    parallel_merge_with_streams_kernel<<<1, 1>>>(d_segments, d_segment_count, threshold, d_temp_segments_data);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(segment_count, d_segment_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(segments, d_segments, (*segment_count) * sizeof(Segment), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_segments);
    cudaFree(d_segment_count);
    cudaFree(d_temp_segments_data);
}