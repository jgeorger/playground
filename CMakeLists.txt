cmake_minimum_required(VERSION 3.28)
project(SegmentMerger LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 121) # 61) is this my GTX1080

# Suppress deprecated GPU target warnings and add line info for profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -lineinfo")

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Create library for segment merger
add_library(segment_merger_lib segment_merger.cu)
target_link_libraries(segment_merger_lib CUDA::cudart)

# Create executable for tests
add_executable(test_segment_merger test_segment_merger.cu)
target_link_libraries(test_segment_merger 
    segment_merger_lib
    gtest
    gtest_main
    CUDA::cudart
)

# Create benchmark executable
add_executable(benchmark_segment_merger benchmark_segment_merger.cu)
target_link_libraries(benchmark_segment_merger 
    segment_merger_lib
    CUDA::cudart
)

add_executable(find_runs find_runs.cu)
target_link_libraries(find_runs
    CUDA::cudart
)

# BPDN (Basis Pursuit Denoise) library and test
add_library(bpdn_lib bpdn_cuda.cu)
target_link_libraries(bpdn_lib CUDA::cudart)
set_property(TARGET bpdn_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_bpdn test_bpdn.cu)
target_link_libraries(test_bpdn
    bpdn_lib
    CUDA::cudart
)
set_property(TARGET test_bpdn PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Enable testing
enable_testing()
add_test(NAME SegmentMergerTests COMMAND test_segment_merger)

# Set CUDA separable compilation for device linking
set_property(TARGET segment_merger_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_segment_merger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET benchmark_segment_merger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
