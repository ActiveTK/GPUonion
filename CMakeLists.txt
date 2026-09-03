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

project(gpuonion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(gpuonion src/main.cu)

if(NOT MSVC)
    target_compile_options(gpuonion PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

if(WIN32)
    target_link_libraries(gpuonion PRIVATE bcrypt)
endif()
