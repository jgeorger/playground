#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "segment_merger.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_segments> <expected_output_segments>" << std::endl;
    std::cout << "  input_segments: Number of input line segments" << std::endl;
    std::cout << "  expected_output_segments: Expected number of output segments after merging" << std::endl;
    std::cout << std::endl;
    std::cout << "The program generates uniformly-distributed segments and measures merge performance." << std::endl;
}

std::vector<Segment> generate_uniform_segments(int count, int expected_output, int seed = 42) {
    std::mt19937 gen(seed);
    std::vector<Segment> segments;
    
    if (count == 0) return segments;
    if (expected_output <= 0 || expected_output > count) {
        expected_output = count; // No merging case
    }
    
    // Strategy: Create segments in clusters where each cluster will merge into one segment
    // Each cluster will have (count / expected_output) segments on average
    int segments_per_cluster = std::max(1, count / expected_output);
    int cluster_spread = 100; // Distance between clusters
    int intra_cluster_gap = 2; // Gap within cluster (small enough to merge with threshold 3)
    int segment_length = 5; // Length of each segment
    
    int current_pos = 0;
    int segments_created = 0;
    int clusters_created = 0;
    
    while (segments_created < count) {
        // Determine cluster size for this cluster
        int remaining_segments = count - segments_created;
        int remaining_clusters = expected_output - clusters_created;
        int cluster_size;
        
        if (remaining_clusters <= 1) {
            // Last cluster gets all remaining segments
            cluster_size = remaining_segments;
        } else {
            // Distribute segments evenly among remaining clusters
            cluster_size = std::max(1, remaining_segments / remaining_clusters);
        }
        
        // Create segments in this cluster
        for (int i = 0; i < cluster_size && segments_created < count; i++) {
            int start = current_pos;
            int end = start + segment_length;
            segments.push_back({start, end});
            segments_created++;
            
            // Move to next segment position within cluster
            current_pos = end + intra_cluster_gap; // Small gap within cluster
        }
        
        clusters_created++;
        
        // Move to next cluster position
        if (segments_created < count) {
            current_pos += cluster_spread;
        }
    }
    
    // Sort segments by start position (requirement of the algorithm)
    std::sort(segments.begin(), segments.end(), 
              [](const Segment& a, const Segment& b) { return a.start < b.start; });
    
    return segments;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    int input_count = std::atoi(argv[1]);
    int expected_output = std::atoi(argv[2]);
    
    if (input_count <= 0) {
        std::cerr << "Error: input_segments must be positive" << std::endl;
        return 1;
    }
    
    if (expected_output <= 0) {
        std::cerr << "Error: expected_output_segments must be positive" << std::endl;
        return 1;
    }
    
    if (expected_output > input_count) {
        std::cerr << "Error: expected_output_segments cannot exceed input_segments" << std::endl;
        return 1;
    }
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }
    
    // Generate test data
    std::cout << "Generating " << input_count << " uniformly-distributed segments..." << std::endl;
    auto segments = generate_uniform_segments(input_count, expected_output);
    
    std::cout << "Input segments: " << segments.size() << std::endl;
    std::cout << "Expected output segments: ~" << expected_output << std::endl;
    
    // Show first few segments for verification
    std::cout << "First 5 segments: ";
    for (int i = 0; i < std::min(5, (int)segments.size()); i++) {
        std::cout << "[" << segments[i].start << "," << segments[i].end << "] ";
    }
    std::cout << std::endl;
    
    // Warm up CUDA
    int warmup_count = segments.size();
    merge_segments(segments.data(), &warmup_count, 3);
    
    // Reset segments for actual benchmark
    segments = generate_uniform_segments(input_count, expected_output);
    int segment_count = segments.size();
    
    // Benchmark the merge operation
    std::cout << "Running benchmark..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use threshold 3 to allow merging (intra_cluster_gap is 2, so 2 <= 3 allows merging)
    merge_segments(segments.data(), &segment_count, 3);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double milliseconds = duration.count() / 1000.0;
    
    // Output results
    std::cout << std::endl;
    std::cout << "=== BENCHMARK RESULTS ===" << std::endl;
    std::cout << "Input segments: " << input_count << std::endl;
    std::cout << "Output segments: " << segment_count << std::endl;
    std::cout << "Runtime: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (input_count / milliseconds * 1000.0) << " segments/second" << std::endl;
    
    // Show first few merged segments
    std::cout << "First 5 merged segments: ";
    for (int i = 0; i < std::min(5, segment_count); i++) {
        std::cout << "[" << segments[i].start << "," << segments[i].end << "] ";
    }
    std::cout << std::endl;
    
    return 0;
}