#!/bin/bash
# 如果 build 目录不存在则创建
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ..
make -j$(nproc)
./main