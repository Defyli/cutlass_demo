cmake_minimum_required(VERSION 3.0)

project(TT)

enable_language(CUDA)

add_subdirectory(CUDACore)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -O3 --use_fast_math -std=c++17 -Xptxas=-v -Xcompiler -fopenmp")

find_package(CUDA REQUIRED)

cuda_add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main PUBLIC "./Tensor" PUBLIC "./include")
target_link_libraries(main ${CUDA_LIBRARIES} -lcublas -lcusparse CK)
# target_compile_options(main PUBLIC "-gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} -O3 --use_fast_math -std=c++17 -Xptxas=-v -Xcompiler -fopenmp")