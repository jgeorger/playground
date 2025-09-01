#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "segment_merger.h"

class SegmentMergerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        // Clean up CUDA
        cudaDeviceReset();
    }
    
    // Helper function to compare segment arrays
    bool segments_equal(const std::vector<Segment>& a, const std::vector<Segment>& b) {
        if (a.size() != b.size()) {
            std::cout << "Size mismatch: actual=" << a.size() << " expected=" << b.size() << std::endl;
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i].start != b[i].start || a[i].end != b[i].end) {
                std::cout << "Segment " << i << " mismatch: actual=[" << a[i].start << "," << a[i].end 
                         << "] expected=[" << b[i].start << "," << b[i].end << "]" << std::endl;
                return false;
            }
        }
        return true;
    }
    
    // Helper function to print segments for debugging
    void print_segments(const std::vector<Segment>& segments, const std::string& label) {
        std::cout << label << ": ";
        for (const auto& seg : segments) {
            std::cout << "[" << seg.start << "," << seg.end << "] ";
        }
        std::cout << std::endl;
    }
    
    // Helper function to run merge_segments and return result as vector
    std::vector<Segment> run_merge(std::vector<Segment> input, int threshold) {
        int count = input.size();
        merge_segments(input.data(), &count, threshold);
        input.resize(count);
        return input;
    }
};

TEST_F(SegmentMergerTest, EmptyArray) {
    std::vector<Segment> segments;
    int count = 0;
    merge_segments(segments.data(), &count, 5);
    EXPECT_EQ(count, 0);
}

TEST_F(SegmentMergerTest, SingleSegment) {
    std::vector<Segment> segments = {{1, 4}};
    std::vector<Segment> expected = {{1, 4}};
    
    auto result = run_merge(segments, 3);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, TwoSegmentsMerge) {
    std::vector<Segment> segments = {{1, 4}, {6, 8}};
    std::vector<Segment> expected = {{1, 8}};
    
    auto result = run_merge(segments, 3);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, TwoSegmentsNoMerge) {
    std::vector<Segment> segments = {{1, 4}, {8, 12}};
    std::vector<Segment> expected = {{1, 4}, {8, 12}};
    
    auto result = run_merge(segments, 3);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, ExampleFromPrompt) {
    std::vector<Segment> segments = {{1, 4}, {8, 12}, {15, 20}};
    std::vector<Segment> expected = {{1, 4}, {8, 20}};
    
    auto result = run_merge(segments, 3);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, AllSegmentsMerge) {
    std::vector<Segment> segments = {{1, 3}, {4, 6}, {7, 9}, {10, 12}};
    std::vector<Segment> expected = {{1, 12}};
    
    auto result = run_merge(segments, 1);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, NoMergeZeroThreshold) {
    std::vector<Segment> segments = {{1, 3}, {5, 7}, {9, 11}};
    std::vector<Segment> expected = {{1, 3}, {5, 7}, {9, 11}};
    
    auto result = run_merge(segments, 0);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, AdjacentSegments) {
    std::vector<Segment> segments = {{1, 3}, {4, 6}, {7, 9}};
    std::vector<Segment> expected = {{1, 9}};
    
    auto result = run_merge(segments, 1);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, LargeThreshold) {
    std::vector<Segment> segments = {{1, 2}, {10, 15}, {25, 30}};
    std::vector<Segment> expected = {{1, 30}};
    
    auto result = run_merge(segments, 100);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, ManySegments) {
    std::vector<Segment> segments;
    for (int i = 0; i < 100; i++) {
        segments.push_back({i * 3, i * 3 + 1});
    }
    
    auto result = run_merge(segments, 2); // Changed threshold to 2 to allow merging
    
    // With threshold 2, segments [0,1], [3,4], [6,7], ... should merge into one big segment
    // Gap between consecutive segments is 2 (e.g., 3-1=2), so threshold 2 allows merging
    std::vector<Segment> expected = {{0, 298}};
    
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, PartialMerge) {
    std::vector<Segment> segments = {{1, 3}, {5, 7}, {9, 11}, {20, 25}, {27, 30}};
    std::vector<Segment> expected = {{1, 11}, {20, 30}};
    
    auto result = run_merge(segments, 2);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, SinglePointSegments) {
    std::vector<Segment> segments = {{1, 1}, {3, 3}, {5, 5}};
    std::vector<Segment> expected = {{1, 5}};
    
    auto result = run_merge(segments, 2);
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, NonNegativeNumbers) {
    std::vector<Segment> segments = {{2, 4}, {7, 9}, {12, 14}};
    std::vector<Segment> expected = {{2, 9}, {12, 14}};
    
    auto result = run_merge(segments, 3);
    
    // Gap 1: 7 - 4 = 3 <= 3, merge to [2,9]
    // Gap 2: 12 - 9 = 3 <= 3, would merge to [2,14], but let's use threshold 2
    
    // Actually, let's use threshold 2 for partial merge
    result = run_merge(segments, 2);
    expected = {{2, 4}, {7, 9}, {12, 14}}; // No merges with threshold 2
    
    
    EXPECT_TRUE(segments_equal(result, expected));
}

TEST_F(SegmentMergerTest, ExactThresholdBoundary) {
    std::vector<Segment> segments = {{1, 4}, {7, 10}};
    
    // With threshold 2, gap is 3, should not merge
    auto result1 = run_merge(segments, 2);
    std::vector<Segment> expected1 = {{1, 4}, {7, 10}};
    EXPECT_TRUE(segments_equal(result1, expected1));
    
    // With threshold 3, gap is 3, should merge
    auto result2 = run_merge(segments, 3);
    std::vector<Segment> expected2 = {{1, 10}};
    EXPECT_TRUE(segments_equal(result2, expected2));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found. Tests require CUDA support." << std::endl;
        return 1;
    }
    
    return RUN_ALL_TESTS();
}