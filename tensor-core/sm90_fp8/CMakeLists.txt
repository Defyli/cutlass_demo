cmake_minimum_required(VERSION 3.18)
project(cutlass_gemm_fp8_sm90 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

# ============================================================================
# SM90 (Hopper H100/H800)
# 必须使用 sm_90a (带 sm90a 特性), 普通 sm_90 不支持 wgmma 指令
# ============================================================================
set(CMAKE_CUDA_ARCHITECTURES "90a")

# ============================================================================
# 编译选项
#   --expt-relaxed-constexpr : CuTe 大量使用 constexpr, 必须开启
#   --expt-extended-lambda   : CUDA lambda 支持
#   -O3 --use_fast_math      : 性能优化
#   --generate-line-info     : Nsight Systems/Compute profiling 支持
# ============================================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    --use_fast_math \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    --generate-line-info \
    -lineinfo \
    --ptxas-options=-v")

# ============================================================================
# 包含目录
#   - 当前目录 (gemm_fp8_sm90.cuh)
#   - utils (CUDA_CHECK 等工具)
#   - cutlass/include (CuTe + Cutlass headers)
#   - cutlass/tools/util/include (辅助工具)
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cutlass/tools/util/include
)

# ============================================================================
# 可执行文件
# ============================================================================
add_executable(gemm_fp8_sm90_demo main_fp8.cu)

# ============================================================================
# 链接库
# ============================================================================
target_link_libraries(gemm_fp8_sm90_demo PRIVATE cudart cublas cublasLt)

# ============================================================================
# 打印配置信息 (可选)
# ============================================================================
message(STATUS "=== SM90 FP8 GEMM Demo Build ===")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CuTe/Cutlass include: ${CMAKE_CURRENT_SOURCE_DIR}/../../cutlass/include")
message(STATUS "Target: gemm_fp8_sm90_demo (FP8 E4M3)")
