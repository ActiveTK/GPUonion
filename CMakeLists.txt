cmake_minimum_required(VERSION 3.18)

# Must be set before project() enables the CUDA language, otherwise CMake
# fills in its own default (a single old arch).
# Real binaries for Pascal .. Hopper, plus PTX (compute_90) so newer GPUs
# (Blackwell sm_100/sm_120, ...) run via JIT. Override with e.g.
#   -DCMAKE_CUDA_ARCHITECTURES=89        (build machine only, fast compile)
#   -DCMAKE_CUDA_ARCHITECTURES=90        (H100 only)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "60-real;70-real;75-real;80-real;86-real;89-real;90-real;90-virtual")
endif()

# ccache, when installed, caches nvcc/host compiler outputs so a rebuild
# whose inputs did not change (same source, same flags, same arch) is served
# from the cache instead of taking minutes. Override with
#   -DCMAKE_CUDA_COMPILER_LAUNCHER= (empty)  to disable
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND NOT DEFINED CMAKE_CUDA_COMPILER_LAUNCHER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "using ccache: ${CCACHE_PROGRAM}")
endif()

project(gpuonion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Every -B batch size is a separate instantiation of each of the three search
# kernels (22 in total), and nvcc spends most of its time in ptxas on them.
# The default build compiles only -B 512, which is what the default run uses;
# turn this on to be able to pass any other -B value.
option(GPUONION_ALL_BATCH_SIZES "compile every -B batch size (much slower build)" OFF)

add_executable(gpuonion src/main.cu)
if(GPUONION_ALL_BATCH_SIZES)
    target_compile_definitions(gpuonion PRIVATE GPUONION_ALL_BATCH_SIZES=1)
endif()

if(NOT MSVC)
    target_compile_options(gpuonion PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

if(WIN32)
    target_link_libraries(gpuonion PRIVATE bcrypt)
endif()
