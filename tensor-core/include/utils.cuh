#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <typename T>
void gen_rand_data(T *data, int n) {
    for (int i = 0; i < n; ++i) {
        float v = (rand() % 200 - 100) * 0.01f;
        data[i] = static_cast<T>(v);
    }
}

template <typename T>
void check_result(const T* host_ref, const T* device_res, int N, const char* label) {

    CUDA_CHECK(cudaDeviceSynchronize());
    T* host_res = (T*)malloc(N * sizeof(T));

    CUDA_CHECK(cudaMemcpy(host_res, device_res, N * sizeof(T), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    float threshold = 0.1f;
    int err_cnt = 0;

    for (int i = 0; i < N; ++i) {
        float v_ref = static_cast<float>(host_ref[i]);
        float v_res = static_cast<float>(host_res[i]);
        float diff = std::fabs(v_res - v_ref);
        if (diff > max_diff) max_diff = diff;
        if (diff > threshold && err_cnt < 5) {
            printf("[%s] Error at %d: ref=%f, res=%f\n", label, i, v_ref, v_res);
            err_cnt++;
        }
    }
    
    if (err_cnt > 0) {
        printf("[%s] FAILED. Max diff: %f\n", label, max_diff);
    } else {
        printf("[%s] PASSED. Max diff: %f\n", label, max_diff);
    }
    
    free(host_res);
}