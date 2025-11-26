#!/bin/bash

# 该脚本允许你选择性地编译 Python 模块或 C++ 测试程序。
# 用法:
#   ./compile.sh python  (编译 Python 模块)
#   ./compile.sh cpp     (编译并运行 C++ 测试)
# 默认编译 Python 模块。

set -e # 如果任何命令失败，立即退出

# --- 配置 ---
BUILD_TARGET=${1:-python} # 默认为 'python'
TARGET_ARCH="80" # cuda arch
TORCH_ARCH="8.0;8.9" #torch arch
# --- 脚本执行 ---
echo "--- 开始构建流程 (目标: ${BUILD_TARGET}) ---"

# 设置环境变量
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')}

# 清理并创建构建目录
BUILD_DIR="build"
echo "--- 清理并创建构建目录: ${BUILD_DIR} ---"
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 根据目标配置 CMake
if [ "$BUILD_TARGET" = "python" ]; then
    echo "--- 正在为 [Python 模块] 配置 CMake ---"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
        -DCMAKE_PYBIND11_ENABLE=ON \
        -GNinja
elif [ "$BUILD_TARGET" = "cpp" ]; then
    echo "--- 正在为 [C++ 测试] 配置 CMake ---"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
        -DCMAKE_PYBIND11_ENABLE=OFF \
        -GNinja
elif [ "$BUILD_TARGET" = "debug" ]; then
    echo "--- 正在为 [Debug 测试] 配置 CMake ---"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DTORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
        -DCMAKE_PYBIND11_ENABLE=OFF \
        -GNinja
else
    echo "错误: 无效的构建目标 '${BUILD_TARGET}'。请使用 'python' 或 'cpp'。"
    exit 1
fi

# 编译
echo "--- 正在使用 Ninja 进行编译 ---"
ninja -j$(nproc)
echo "--- 编译成功 ---"

# 后续步骤
if [ "$BUILD_TARGET" = "python" ]; then
    echo "--- 正在拷贝 Python 模块到 mf_experimental/ ---"
    cp include/hstu_attn*.so ../python_lib/
    echo "Python 模块 'hstu_attn' 已准备就绪。"
elif [ "$BUILD_TARGET" = "cpp" ]; then
    echo "--- 正在运行 C++ 性能测试 ---"
    # 由于 CMakeLists.txt 中设置了 RPATH，可执行文件应该能找到 .so 库
    ./mfalcon_test
elif [ "$BUILD_TARGET" = "debug" ]; then
    echo "--- 正在运行 C++ debug ---"
    compute-sanitizer --tool memcheck ./mfalcon_test > mfalcon_test_output.log 2>&1
fi

echo "--- 脚本执行完毕 ---"# 如果需要用compute-sanitizer进行正确性检查,需要重新用Debug模式编译
# echo "--- Running M-Falcon test with Compute Sanitizer (memcheck) ---"
# cmake .. -DCMAKE_BUILD_TYPE=Debug && ninja
# compute-sanitizer --tool memcheck ./mfalcon_test > mfalcon_test_output.log 2>&1
# echo "--- Test finished. Compute Sanitizer output has been redirected to build/mfalcon_test_output.log ---"