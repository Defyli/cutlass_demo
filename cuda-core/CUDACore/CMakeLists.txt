cmake_minimum_required(VERSION 3.0)

project(CUDACore LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -O3 --use_fast_math -std=c++17 -Xptxas=-v -Xcompiler -fopenmp")

find_package(CUDA REQUIRED)

file(GLOB KERNEL_SOURCE "./*.cu")

cuda_add_library(CK SHARED ${KERNEL_SOURCE})
target_include_directories(CK PUBLIC "../include")
target_link_libraries(CK ${CUDA_LIBRARIES} -lcublas -lcusparse)