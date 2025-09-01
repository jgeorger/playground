# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a playground repository for experimenting with CUDA code, specifically focused on line segment merging algorithms.

## Build Commands
```bash
# Configure for release build
cmake --preset release

# Configure for debug build
cmake --preset debug

# Build project
cmake --build --preset release

# Run tests
./build/release/test_segment_merger

# Clean build artifacts  
rm -rf build/
```

## Architecture
- `segment_merger.cu`: Main CUDA implementation for merging line segments
- `test_segment_merger.cu`: GoogleTest-based unit test suite
- `CMakeLists.txt`: CMake build configuration
- `CMakePresets.json`: Common build presets (release/debug)
- `CMakeUserPresets.json`: User-specific build configurations
- Line segments are represented as structs with `start` and `end` fields
- Algorithm assumes input segments are sorted and non-overlapping
- Merges segments within a configurable distance threshold

## Development Notes
- CMake 3.28+ required
- Ninja build system required
- CUDA architecture 6.1+ required
- Uses GoogleTest framework for unit testing
- Uses explicit device memory management with cudaMalloc/cudaMemcpy
- Tests cover edge cases including empty arrays, single segments, and various threshold values