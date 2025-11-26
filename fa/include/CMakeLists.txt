# cmake_minimum_required(VERSION 3.18)
# # 设置标准
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)

# # 强制设置CUDA架构
# set(CMAKE_CUDA_ARCHITECTURES "80")

# # 编译选项
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
#     add_definitions(-DHSTU_MINIMAL_INSTANTIATION)
# else()
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
# endif()


# 1. 创建一个 OBJECT 库来编译所有源文件
set(HSTU_SOURCES
    hstu_api.cpp
    hstu_instantiations.cu
)
file(GLOB_RECURSE HSTU_HEADERS "*.h" "*.hpp" "*.cuh")

add_library(hstu_objects OBJECT ${HSTU_SOURCES} ${HSTU_HEADERS})

# 2. 将所有编译和链接设置应用于 OBJECT 库
set_source_files_properties(${HSTU_SOURCES} ${HSTU_HEADERS} PROPERTIES LANGUAGE CUDA)

set_target_properties(hstu_objects PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "80"
)

target_compile_options(hstu_objects PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --ptxas-options=-v
        --generate-line-info
        -Xcompiler=-fPIC
        -Xcompiler=-fopenmp
    >
)

target_include_directories(hstu_objects PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/examples/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_compile_definitions(hstu_objects PUBLIC
    CUTE_STATIC_ASSERT_ON_FAIL
    HSTU_DISABLE_BACKWARD
    HSTU_DISABLE_86OR89
    HSTU_DISABLE_HDIM32
    HSTU_DISABLE_HDIM64
    HSTU_DISABLE_HDIM128
    # HSTU_DISABLE_RAB
    HSTU_DISABLE_DRAB
    HSTU_DISABLE_CONTEXT
    HSTU_DISABLE_LOCAL
    HSTU_DISABLE_DELTA_Q
    HSTU_DISABLE_PAGED_KV
    HSTU_DISABLE_TARGET
    HSTU_DISABLE_CAUSAL
    # PYBIND11_DISABLE
    # MFALCON_DEBUG
    HSTU_CXX_TEST
)
if(NOT CMAKE_PYBIND11_ENABLE)
    target_compile_definitions(hstu_objects PUBLIC PYBIND11_DISABLE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(hstu_objects PRIVATE HSTU_MINIMAL_INSTANTIATION)
endif()


if(TARGET Torch::torch)
    target_link_libraries(hstu_objects PUBLIC Torch::torch)
else()
    target_include_directories(hstu_objects PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(hstu_objects PUBLIC ${TORCH_LIBRARIES})
endif()
target_link_libraries(hstu_objects PUBLIC Python3::Python)

target_compile_definitions(hstu_objects PRIVATE
    TORCH_EXTENSION_NAME=hstu_attn
    TORCH_API_INCLUDE_EXTENSION_H
)



# 4. 如果启用，则创建 Python MODULE 库
if(CMAKE_PYBIND11_ENABLE)
    pybind11_add_module(hstu_attn MODULE $<TARGET_OBJECTS:hstu_objects>)
    set_target_properties(hstu_attn PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(hstu_attn PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Python3_LIBRARIES})
else()
    # 3. 创建一个可链接的 SHARED 库供 C++ 测试使用
    add_library(hstu_attn_lib SHARED $<TARGET_OBJECTS:hstu_objects>)
    set_target_properties(hstu_attn_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(hstu_attn_lib PUBLIC hstu_objects)
endif()

# 设置导出变量供父级使用
set(HSTU_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)