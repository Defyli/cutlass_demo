cmake_minimum_required(VERSION 3.18)
project(cutlass_gemm_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

# 设置 GPU 架构 (例如 RTX 3090/A100 = 86/80)
# sm89 for 4090
set(CMAKE_CUDA_ARCHITECTURES 89)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info")

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/SM89
)

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cu")

# 可执行文件
add_executable(gemm_demo ${SOURCES})

# 链接库
target_link_libraries(gemm_demo PRIVATE cudart cublas)