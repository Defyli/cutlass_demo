#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "hstu_api.h"


template <typename T>
struct DtypeMap;

template <>
struct DtypeMap<at::Half> {
    static constexpr torch::Dtype dtype = torch::kFloat16;
};

template <>
struct DtypeMap<at::BFloat16> {
    static constexpr torch::Dtype dtype = torch::kBFloat16;
};

// 测试配置结构体
struct TestConfig {
    int batch_size;
    int user_length;
    int target_length;
    int num_heads;
    int head_dim;
    int micro_bs;
    int num_iterations;
    bool is_rab;
    bool has_mask;
    
    TestConfig(int b, int u, int t, int h, int d, int mbs, int iter = 10,bool rab=false,bool mask=false) 
        : batch_size(b), user_length(u), target_length(t), 
          num_heads(h), head_dim(d), micro_bs(mbs), num_iterations(iter),is_rab(rab),has_mask(mask) {}
};


// 性能测试结果
struct PerformanceResult {
    double mfalcon_time_ms;
    double regular_time_ms;
    double mask_time_ms;
    double speedup;
    double mask_speedup = 0.0;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3)
                  << "M-Falcon: " << mfalcon_time_ms << "ms, "
                  << "Regular: " << regular_time_ms << "ms, "
                  << "Mask: "<< mask_time_ms <<"ms, "
                  << "Speedup: " << speedup << "x" <<" "
                  << "Mask Speedup: " << mask_speedup << "x"<<std::endl;
    }
};

// 精度测试结果
struct AccuracyResult {
    double max_abs_error;
    double mean_abs_error;
    double relative_error;
    bool passed;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(6)
                  << "Max Error: " << max_abs_error << ", "
                  << "Mean Error: " << mean_abs_error << ", "
                  << "Relative Error: " << relative_error << ", "
                  << "Status: " << (passed ? "PASS" : "FAIL") << std::endl;
    }
};

template <typename T>
class MFalconTester {
private:
    cudaStream_t stream_;
    
public:
    MFalconTester() {
        cudaStreamCreate(&stream_);
    }
    
    ~MFalconTester() {
        cudaStreamDestroy(stream_);
    }
    
    // 创建测试数据
    std::vector<at::Tensor> createTestData(const TestConfig& config) {
        auto options = torch::TensorOptions().device(torch::kCUDA).dtype(DtypeMap<T>::dtype);
        
        // 计算总序列长度
        std::vector<int> cu_seqlens_q_vec = {0};
        std::vector<int> cu_seqlens_k_vec = {0};
        
        int total_q = 0, total_k = 0;
        
        for (int i = 0; i < config.batch_size; ++i) {
            int seq_len = config.user_length + config.target_length;
            total_q += seq_len;
            total_k += seq_len;
            
            cu_seqlens_q_vec.push_back(total_q);
            cu_seqlens_k_vec.push_back(total_k);
        }
        
        // 创建张量
        auto q = torch::randn({total_q, config.num_heads, config.head_dim}, options);
        auto k = torch::randn({total_k, config.num_heads, config.head_dim}, options);
        auto v = torch::randn({total_k, config.num_heads, config.head_dim}, options);
        
        auto cu_seqlens_q = torch::from_blob(cu_seqlens_q_vec.data(), 
                                           {config.batch_size + 1}, 
                                           torch::TensorOptions().dtype(torch::kInt32))
                                           .to(torch::kCUDA).clone();
        auto cu_seqlens_k = torch::from_blob(cu_seqlens_k_vec.data(), 
                                           {config.batch_size + 1}, 
                                           torch::TensorOptions().dtype(torch::kInt32))
                                           .to(torch::kCUDA).clone();
        // M-Falcon 参数
        auto user_length = torch::full({config.batch_size}, config.user_length, 
                                     torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto target_length = torch::full({config.batch_size}, config.target_length, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto micro_bs = torch::full({config.batch_size}, config.micro_bs, 
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        std::vector<at::Tensor> tensors = {q, k, v, cu_seqlens_q, cu_seqlens_k,
                                           user_length, target_length, micro_bs};

        if (config.has_mask) {
            int seq_len = config.user_length + config.target_length;
            tensors.push_back(torch::ones({config.batch_size, seq_len, seq_len}, options));
        }

        return tensors;
    }
    
    // 运行 M-Falcon 版本
    at::Tensor runMFalcon(const std::vector<at::Tensor>& inputs, const TestConfig& config) {
        auto q = inputs[0], k = inputs[1], v = inputs[2];
        auto cu_seqlens_q = inputs[3], cu_seqlens_k = inputs[4];
        auto user_length = inputs[5], target_length = inputs[6], micro_bs = inputs[7];
        
        // 使用 set_params_fprop 设置参数
        int max_seqlen_q = config.user_length + config.target_length;
        int max_seqlen_k = config.user_length + config.target_length;

        std::optional<at::Tensor> attn_mask_opt;
        if (config.has_mask) {
             attn_mask_opt = inputs[8];
        }

        std::optional<const at::Tensor> num_contexts_opt, num_targets_opt, kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt;
        std::optional<at::Tensor> rab_opt;
        std::optional<const bool> is_mfalcon_opt = true;
        std::optional<const at::Tensor> user_length_opt = user_length;
        std::optional<const at::Tensor> micro_bs_opt = micro_bs;
        std::optional<const at::Tensor> target_length_opt = target_length;
        
        
        // 调用内核
        auto result_vec = hstu_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            num_contexts_opt, num_targets_opt,
            1, -1, -1,
            1.0f / std::sqrt(config.head_dim),
            rab_opt, false,
            kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt,
            attn_mask_opt,
            is_mfalcon_opt,
            user_length_opt,
            micro_bs_opt,
            target_length_opt
        );
       
        return result_vec[0];
    }
    
    // 运行常规版本
    at::Tensor runRegular(const std::vector<at::Tensor>& inputs, const TestConfig& config) {
        auto q = inputs[0], k = inputs[1], v = inputs[2];
        auto cu_seqlens_q = inputs[3], cu_seqlens_k = inputs[4];
        
        
        // 使用 set_params_fprop 设置参数
        int max_seqlen_q = config.user_length + config.target_length;
        int max_seqlen_k = config.user_length + config.target_length;

        std::optional<const at::Tensor> num_contexts_opt, num_targets_opt, kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt;
        std::optional<at::Tensor> rab_opt;
        std::optional<const bool> is_mfalcon_opt = false;
        std::optional<const at::Tensor> cu_seqlens_cache_opt, user_length_opt, micro_bs_opt, target_length_opt, mfalcon_k_cache_opt, mfalcon_v_cache_opt, rab_cache_opt;
        std::optional<at::Tensor> attn_mask_opt;
        auto result_vec = hstu_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            num_contexts_opt, num_targets_opt,
            1, -1, -1,
            1.0f / std::sqrt(config.head_dim),
            rab_opt, false,
            kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt,
            attn_mask_opt,
            is_mfalcon_opt,
            user_length_opt,
            micro_bs_opt,
            target_length_opt
        );
        
        return result_vec[0];
    }

    at::Tensor runRegularCausalWithRab(const std::vector<at::Tensor>& inputs, const TestConfig& config) {
        auto q = inputs[0], k = inputs[1], v = inputs[2];
        auto cu_seqlens_q = inputs[3], cu_seqlens_k = inputs[4];
        
        int max_seqlen_q = config.user_length + config.target_length;
        int max_seqlen_k = config.user_length + config.target_length;

        // 启用因果掩码
        int window_size_left = max_seqlen_k;
        int window_size_right = 0;

        // 启用RAB
        auto options = torch::TensorOptions().device(torch::kCUDA).dtype(DtypeMap<T>::dtype);
        auto rab = torch::zeros({config.batch_size, config.num_heads, max_seqlen_q, max_seqlen_k}, options);
        std::optional<at::Tensor> rab_opt = rab;

        std::optional<const at::Tensor> num_contexts_opt, num_targets_opt, kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt;
        std::optional<const bool> is_mfalcon_opt = false;
        std::optional<const at::Tensor> user_length_opt, micro_bs_opt, target_length_opt;
        std::optional<at::Tensor> attn_mask_opt;

        auto result_vec = hstu_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            num_contexts_opt, num_targets_opt,
            1, -1, -1,
            1.0f / std::sqrt(config.head_dim),
            rab_opt, false,
            kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt,
            attn_mask_opt,
            is_mfalcon_opt,
            user_length_opt,
            micro_bs_opt,
            target_length_opt
        );
        
        return result_vec[0];
    }

    at::Tensor runMask(const std::vector<at::Tensor>& inputs, const TestConfig& config)
    {
        auto q = inputs[0], k = inputs[1], v = inputs[2];
        auto cu_seqlens_q = inputs[3], cu_seqlens_k = inputs[4];
        
        // 使用 set_params_fprop 设置参数
        int max_seqlen_q = config.user_length + config.target_length;
        int max_seqlen_k = config.user_length + config.target_length;

        std::optional<at::Tensor> attn_mask_opt;
        if (config.has_mask) {
             attn_mask_opt = inputs[8];
        }

        std::optional<const at::Tensor> num_contexts_opt, num_targets_opt, kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt;
        std::optional<at::Tensor> rab_opt;
        std::optional<const bool> is_mfalcon_opt = false;
        std::optional<const at::Tensor> user_length_opt;
        std::optional<const at::Tensor> micro_bs_opt;
        std::optional<const at::Tensor> target_length_opt;
        
        
        // 调用内核
        auto result_vec = hstu_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            num_contexts_opt, num_targets_opt,
            1, -1, -1,
            1.0f / std::sqrt(config.head_dim),
            rab_opt, false,
            kv_cache_opt, page_offsets_opt, page_ids_opt, last_page_lens_opt, cu_seqlens_t_opt,
            attn_mask_opt,
            is_mfalcon_opt,
            user_length_opt,
            micro_bs_opt,
            target_length_opt
        );
       
        return result_vec[0];

    }
    

     PerformanceResult benchmarkPerformanceWithMask(const TestConfig& config) {
        // 为M-Falcon模式创建带掩码的数据
        TestConfig mask_config = config;
        mask_config.has_mask = true;
        auto inputs = createTestData(mask_config);
        
        // 预热
        for (int i = 0; i < 3; ++i) {
            this->runMFalcon(inputs, mask_config);
            this->runRegularCausalWithRab(inputs, config);
            this->runMask(inputs, config);
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        float mfalcon_total_time = 0.0f;
        float regular_total_time = 0.0f;
        float mask_total_time = 0.0f;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        // 测试 M-Falcon (带掩码)
        cudaEventRecord(start_event, stream);
        for (int i = 0; i < config.num_iterations; ++i) {
            this->runMFalcon(inputs, mask_config);
        }
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&mfalcon_total_time, start_event, stop_event);
        
        // 测试常规 HSTU (带因果掩码和RAB)
        cudaEventRecord(start_event, stream);
        for (int i = 0; i < config.num_iterations; ++i) {
            this->runRegularCausalWithRab(inputs, config);
        }
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&regular_total_time, start_event, stop_event);

        //测试mask版本(仅有mask)
        cudaEventRecord(start_event, stream);
        for(int i = 0; i < config.num_iterations; ++i)
        {
            this->runMask(inputs, config);
        }
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&mask_total_time, start_event, stop_event);

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        PerformanceResult result;
        result.mfalcon_time_ms = mfalcon_total_time / config.num_iterations;
        result.regular_time_ms = regular_total_time / config.num_iterations;
        result.mask_time_ms = mask_total_time / config.num_iterations;
        result.speedup = result.regular_time_ms / result.mfalcon_time_ms;
        result.mask_speedup =  result.regular_time_ms / result.mask_time_ms;
        
        return result;
    }
    
    // 性能测试
    PerformanceResult benchmarkPerformance(const TestConfig& config) {
        auto inputs = createTestData(config);
        
        // 预热
        for (int i = 0; i < 3; ++i) {
            this->runMFalcon(inputs, config);
            this->runRegular(inputs, config);
        }
        cudaDeviceSynchronize();
        // 测试 M-Falcon
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        float mfalcon_total_time = 0.0f;
        float regular_total_time = 0.0f;

        // 获取 PyTorch 当前使用的 CUDA stream
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        // 测试 M-Falcon
        cudaEventRecord(start_event, stream);
        for (int i = 0; i < config.num_iterations; ++i) {
            this->runMFalcon(inputs, config);
        }
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&mfalcon_total_time, start_event, stop_event);
        
        // 测试常规版本
        cudaEventRecord(start_event, stream);
        for (int i = 0; i < config.num_iterations; ++i) {
            this->runRegular(inputs, config);
        }
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&regular_total_time, start_event, stop_event);
        
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        PerformanceResult result;
        result.mfalcon_time_ms = mfalcon_total_time / config.num_iterations;
        result.regular_time_ms = regular_total_time / config.num_iterations;
        result.speedup = result.regular_time_ms / result.mfalcon_time_ms;
        
        return result;
    }
    
    // 精度测试
    AccuracyResult testAccuracy(const TestConfig& config) {
        auto inputs = this->createTestData(config);
        
        auto mfalcon_out = this->runMFalcon(inputs, config);
        auto regular_out = this->runRegular(inputs, config);
        
        // 转换为 float 进行比较
        auto mfalcon_float = mfalcon_out.to(torch::kFloat32);
        auto regular_float = regular_out.to(torch::kFloat32);
        
        auto diff = torch::abs(mfalcon_float - regular_float);
        auto max_error = torch::max(diff).item<double>();
        auto mean_error = torch::mean(diff).item<double>();
        
        auto regular_norm = torch::norm(regular_float).item<double>();
        auto diff_norm = torch::norm(diff).item<double>();
        auto relative_error = diff_norm / regular_norm;
        
        AccuracyResult result;
        result.max_abs_error = max_error;
        result.mean_abs_error = mean_error;
        result.relative_error = relative_error;
        result.passed = (relative_error < 1e-2); // 1% 相对误差阈值
        
        return result;
    }
    
    // 运行完整测试套件
void runTestSuite(const std::string& dtype_name) {
        std::vector<TestConfig> configs;
        const int user_length = 1376;
        const int num_heads = 3;
        const int default_iterations = 10;


        const int head_dim_tests[] = {256};
        const int batch_size_tests[] = {1};
        const int target_length[] = {256,512,1024,2048,4096};

        for (int head_dim : head_dim_tests) {
            for (int batch_size : batch_size_tests) {
                for (int target_length : target_length) {
                    for (int micro_bs = 128; micro_bs < target_length; micro_bs *= 2) {
                        configs.emplace_back(
                            batch_size,
                            user_length,
                            target_length,
                            num_heads,
                            head_dim,
                            micro_bs,
                            default_iterations
                        );
                    }
                }
            }
        }
        
        std::cout << "=== M-Falcon vs Regular HSTU Performance & Accuracy Test (" << dtype_name << ") ===" << std::endl;
        std::cout << std::endl;
        
        for (const auto& config : configs) {
            std::cout << "Testing Configuration:" << std::endl;
            std::cout << "  Batch: " << config.batch_size 
                      << ", User: " << config.user_length 
                      << ", Target: " << config.target_length
                      << ", Heads: " << config.num_heads 
                      << ", HeadDim: " << config.head_dim 
                      << ", MicroBS: " << config.micro_bs << std::endl;
            
            try {
                // // 性能测试
                // std::cout << "  Performance: ";
                // auto perf_result = this->benchmarkPerformance(config);
                // perf_result.print();
                
                // // 精度测试
                // std::cout << "  Accuracy: ";
                // auto acc_result = this->testAccuracy(config);
                // acc_result.print();

                std::cout << "  Comparison Test (M-Falcon+Mask vs Regular+RAB vs Mask): "<<std::endl;
                auto perf_comp_result = this->benchmarkPerformanceWithMask(config);
                perf_comp_result.print();
                
            } catch (const std::exception& e) {
                std::cout << "  ERROR: " << e.what() << std::endl;
            }
            
            std::cout << std::endl;
        }
    }
};

int main() {
    try {
        // 初始化 CUDA
        cudaSetDevice(0);
        
        // 检查 GPU 架构
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
        
        if (prop.major < 8) {
            std::cerr << "Error: This test requires Ampere (SM 8.0) or newer GPU" << std::endl;
            return -1;
        }
        
        // 运行测试
        // 运行 FP16 测试
        std::cout << "\n--- Running tests for FP16 ---\n" << std::endl;
        MFalconTester<at::Half> tester_fp16;
        tester_fp16.runTestSuite("FP16");

        // 运行 BF16 测试
        std::cout << "\n--- Running tests for BF16 ---\n" << std::endl;
        MFalconTester<at::BFloat16> tester_bf16;
        tester_bf16.runTestSuite("BF16");
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}