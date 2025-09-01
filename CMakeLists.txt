cmake_minimum_required(VERSION 3.28)
project(SegmentMerger LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61)

# Suppress deprecated GPU target warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

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

# Enable testing
enable_testing()
add_test(NAME SegmentMergerTests COMMAND test_segment_merger)

# Set CUDA separable compilation for device linking
set_property(TARGET segment_merger_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_segment_merger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET benchmark_segment_merger PROPERTY CUDA_SEPARABLE_COMPILATION ON)