cmake_minimum_required(VERSION 3.18)
project(cutlass_gemm_fp8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

# 关键：设置 SM89 架构 (RTX 4090, L40, Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 编译选项
# --expt-relaxed-constexpr: 允许在 constexpr 中调用 host 函数 (Cutlass 需要)
# --use_fast_math: 性能优化
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info")

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cutlass/tools/util/include

)

# 源文件 (只编译 FP8 的 main)
add_executable(gemm_fp8_demo ./main.cu)

# 链接库
# 注意：必须链接 cublasLt 才能运行 FP8 基准测试
target_link_libraries(gemm_fp8_demo PRIVATE cudart cublas cublasLt)