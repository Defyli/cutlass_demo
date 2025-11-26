cmake_minimum_required(VERSION 3.18)
project(hstu_test LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(NCCL_DIR "/usr/local/lib/python3.9/site-packages/nvidia/nccl")
# 查找必要的包
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#find_library(NCCL_LIBRARY "${NCCL_DIR}/lib")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")
# 定义一个选项来控制 Pybind11 模块的构建
option(CMAKE_PYBIND11_ENABLE "Enable pybind11 module build" OFF)

# 强制设置CUDA架构，只编译sm80
# set(CMAKE_CUDA_ARCHITECTURES "80")
# unset(TORCH_CUDA_ARCH_LIST)

# 设置全局编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --threads 0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
endif()

# 根据构建目标添加子目录
if(CMAKE_PYBIND11_ENABLE)
    message(STATUS "Configuring for Python module build.")
    add_subdirectory(pybind11)
endif()

# 无条件添加 include 子目录
add_subdirectory(include)

# 仅在不构建 Python 模块时，才配置和构建 C++ 测试程序
if(NOT CMAKE_PYBIND11_ENABLE)
    message(STATUS "Configuring for C++ test build.")
    # ===================================================================
    # 编译 mfalcon_test 测试程序
    # ===================================================================
    add_executable(mfalcon_test test/mfalcon_test.cpp)
    # 明确设置测试代码由 nvcc 编译
    set_source_files_properties(test/mfalcon_test.cpp PROPERTIES LANGUAGE CUDA)
    # 设置测试程序属性
    set_target_properties(mfalcon_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        INSTALL_RPATH "$ORIGIN/include"
    )

    # 为测试程序链接库
    target_link_libraries(mfalcon_test PRIVATE
        hstu_attn_lib
        # ${TORCH_LIBRARIES} # 不再需要，由 hstu_attn_lib 传递
        #${NCCL_DIR}/lib/libnccl.so.2
        cublas
    )
    target_compile_definitions(mfalcon_test PRIVATE HSTU_CXX_TEST)
    # 为测试程序添加包含目录
    target_include_directories(mfalcon_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
        #${NCCL_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    )
endif()

# 打印配置信息
message(STATUS "Building HSTU as SHARED library")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Torch Version: ${Torch_VERSION}")