#!/bin/bash
# SM90 (Hopper) GEMM Demo 构建脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building SM90 GEMM Demo ==="
echo "Source dir: ${SCRIPT_DIR}"
echo "Build dir:  ${BUILD_DIR}"

# 清理旧 build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# CMake 配置
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc) VERBOSE=0

echo ""
echo "=== Build Complete ==="
echo ""

# 运行 (可选指定 M N K)
if [ "$1" = "run" ]; then
    echo "=== Running gemm_sm90_demo ==="
    ./gemm_sm90_demo ${2:-4096} ${3:-4096} ${4:-2048}
fi
