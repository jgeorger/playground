# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a playground repository for experimenting with CUDA code, specifically focused on line segment merging algorithms.

## Build Commands
```bash
# Configure (Ninja Multi-Config: one build tree for Release, Debug, RelWithDebInfo)
cmake --preset default

# Build all three configurations
cmake --build --preset all

# Build a single configuration (release | debug | relwithdebinfo)
cmake --build --preset release

# Run tests for a configuration via CTest
ctest --preset release

# Run tests directly (binaries land in build/<Config>/)
./build/Release/test_segment_merger

# Run benchmark (example: 1000 input segments, expect 100 output segments)
./build/Release/benchmark_segment_merger 1000 100

# Clean build artifacts
rm -rf build/
```

## Architecture
- `segment_merger.cu`: Main CUDA implementation for merging line segments
- `segment_merger.h`: Header file with function declarations
- `test_segment_merger.cu`: GoogleTest-based unit test suite
- `benchmark_segment_merger.cu`: Performance benchmark program
- `CMakeLists.txt`: CMake build configuration
- `CMakePresets.json`: Ninja Multi-Config preset (`default`) with build/test presets for release, debug, relwithdebinfo (plus `all` build preset)
- `CMakeUserPresets.json`: User-specific build configurations
- Line segments are represented as structs with `start` and `end` fields
- Algorithm assumes input segments are sorted and non-overlapping
- Merges segments within a configurable distance threshold

## Development Notes
- CMake 3.28+ required
- Ninja build system required (uses the Ninja Multi-Config generator)
- CUDA architecture 6.1+ required
- Uses GoogleTest framework for unit testing
- Uses explicit device memory management with cudaMalloc/cudaMemcpy
- Tests cover edge cases including empty arrays, single segments, and various threshold values