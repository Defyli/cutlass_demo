cmake_minimum_required(VERSION 3.18)
project(cutlass_simple_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 添加 cutlass 子目录，确保 cutlass 仓库已克隆到项目根目录

find_package(CUDA REQUIRED)
# 设置你的 GPU 计算架构。
# 例如:
# 75 for Turing (RTX 20xx)
# 80 for Ampere (A100)
# 86 for Ampere (RTX 30xx)
# 89 for Ada Lovelace (RTX 40xx)
# 请根据你的 GPU 型号修改此值
set(CMAKE_CUDA_ARCHITECTURES 89)
set(COMMON_NVCC_FLAGS "-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda --ptxas-options=-v --generate-line-info -Xcompiler=-fPIC -Xcompiler=-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_NVCC_FLAGS}")



add_executable(cutlass_simple_demo main.cu)


# 显式地将 cutlass/include 添加到头文件搜索路径
# 这会告诉编译器去 cutlass/include 目录中查找头文件
target_include_directories(cutlass_simple_demo PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
        ${CUDA_INCLUDE_DIRS})


# 链接 CUTLASS 和 CUDA 运行时库
target_link_libraries(cutlass_simple_demo PRIVATE cudart CUDA::cublas)