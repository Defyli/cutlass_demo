// #include "hstu_fwd.h"

// // 为动态库提供完整的模板实例化
// // 这样可以避免链接时的undefined reference错误

// // ============================================================================
// // A100 (SM80) 架构的实例化
// // ============================================================================
// #include "hstu_fwd.h"

// // 为动态库提供完整的模板实例化
// // 这样可以避免链接时的undefined reference错误

// // ============================================================================
// // A100 (SM80) 架构的实例化
// // ============================================================================

// // ============================================================================
// // 默认路径实例化 (Is_mfalcon = false)
// // ============================================================================

// // FP16 数据类型 - 256维度
// template void run_hstu_fwd_<80, cutlass::half_t, 256, false, false, false, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, false, false, false, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);

// // FP16 + 128维
// #ifndef HSTU_DISABLE_HDIM128
// template void run_hstu_fwd_<80, cutlass::half_t, 128, false, false, false, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// template void run_hstu_fwd_<80, cutlass::bfloat16_t, 128, false, false, false, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// #endif

// #ifndef HSTU_DISABLE_RAB
// template void run_hstu_fwd_<80, cutlass::half_t, 256, true, false, false, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, true, false, false, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// #endif 

// #ifndef HSTU_DISABLE_CAUSAL
// template void run_hstu_fwd_<80, cutlass::half_t, 256, false, false, true, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, false, false, true, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
// #endif

// #if !defined(HSTU_DISABLE_RAB) && !defined(HSTU_DISABLE_CAUSAL)
//     template void run_hstu_fwd_<80, cutlass::half_t, 256, true, false, true, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);
//     template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, true, false, true, false, false, false, false>(Hstu_fwd_params&, cudaStream_t);

// #endif

// #ifndef HSTU_DISABLE_MASK



// #endif
// // ============================================================================
// // M-Falcon 路径实例化 (Is_mfalcon = true)
// // ============================================================================
// #ifndef HSTU_MFALCON_DISABLE
//     // Has_rab = false, Has_mask = false
//     // FP16 数据类型 - 256维度
//     template void run_hstu_fwd_<80, cutlass::half_t, 256, false, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//     template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, false, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//     // FP16 + 128维
//     #ifndef HSTU_DISABLE_HDIM128
//     template void run_hstu_fwd_<80, cutlass::half_t, 128, false, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//     template void run_hstu_fwd_<80, cutlass::bfloat16_t, 128, false, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//     #endif

//     // Has_rab = true, Has_mask = false
//     #ifndef HSTU_DISABLE_RAB
//         template void run_hstu_fwd_<80, cutlass::half_t, 256, true, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//         template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, true, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//         // FP16 + 128维
//         #ifndef HSTU_DISABLE_HDIM128
//         template void run_hstu_fwd_<80, cutlass::half_t, 128, true, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//         template void run_hstu_fwd_<80, cutlass::bfloat16_t, 128, true, false, false, false, false, false, true, false>(Hstu_fwd_params&, cudaStream_t);
//         #endif
//     #endif

//     // Has_rab = false, Has_mask = true
//     #ifndef HSTU_DISABLE_MASK
//         template void run_hstu_fwd_<80, cutlass::half_t, 256, false, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, false, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         #ifndef HSTU_DISABLE_HDIM128
//         template void run_hstu_fwd_<80, cutlass::half_t, 128, false, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         template void run_hstu_fwd_<80, cutlass::bfloat16_t, 128, false, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         #endif
//     #endif

//     // Has_rab = true, Has_mask = true
//     #if !defined(HSTU_DISABLE_RAB) && !defined(HSTU_DISABLE_MASK)
//         template void run_hstu_fwd_<80, cutlass::half_t, 256, true, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         template void run_hstu_fwd_<80, cutlass::bfloat16_t, 256, true, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         #ifndef HSTU_DISABLE_HDIM128
//         template void run_hstu_fwd_<80, cutlass::half_t, 128, true, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         template void run_hstu_fwd_<80, cutlass::bfloat16_t, 128, true, false, false, false, false, false, true, true>(Hstu_fwd_params&, cudaStream_t);
//         #endif
//     #endif

// #endif

#include "hstu_fwd.h"

// 本文件为 hstu_fwd 内核提供显式模板实例化。
// 使用宏可以更轻松地管理大量的模板组合，
// 并遵循构建配置中的特性禁用标志。

// 基础宏，用于实例化一个特定的内核变体。
// 除了 Has_rab, Is_mfalcon, 和 Has_mask 之外，所有布尔标志都设置为 false，
// 以匹配当前禁用其他特性的构建配置。
#define INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, HAS_RAB, IS_MFALCON, HAS_MASK) \
    template void run_hstu_fwd_<ARCH, DTYPE, KHEADDIM, HAS_RAB, \
                                /*Is_local=*/false, /*Is_causal=*/false, /*Is_context=*/false, \
                                /*Is_target=*/false, /*Is_delta_q=*/false, \
                                IS_MFALCON, HAS_MASK>(Hstu_fwd_params&, cudaStream_t);

// --- 用于条件实例化的辅助宏 ---
#ifdef HSTU_DISABLE_RAB
    #define INSTANTIATE_IF_RAB_ENABLED(...)
#else
    #define INSTANTIATE_IF_RAB_ENABLED(...) __VA_ARGS__
#endif

#ifdef HSTU_DISABLE_MASK
    #define INSTANTIATE_IF_MASK_ENABLED(...)
#else
    #define INSTANTIATE_IF_MASK_ENABLED(...) __VA_ARGS__
#endif

#ifdef HSTU_MFALCON_DISABLE
    #define INSTANTIATE_IF_MFALCON_ENABLED(...)
#else
    #define INSTANTIATE_IF_MFALCON_ENABLED(...) __VA_ARGS__
#endif

#ifdef HSTU_DISABLE_HDIM256
    #define INSTANTIATE_HDIM256(...)
#else
    #define INSTANTIATE_HDIM256(...) __VA_ARGS__
#endif

#ifdef HSTU_DISABLE_HDIM128
    #define INSTANTIATE_HDIM128(...)
#else
    #define INSTANTIATE_HDIM128(...) __VA_ARGS__
#endif

#ifdef HSTU_DISABLE_HDIM64
    #define INSTANTIATE_HDIM64(...)
#else
    #define INSTANTIATE_HDIM64(...) __VA_ARGS__
#endif

#ifdef HSTU_DISABLE_HDIM32
    #define INSTANTIATE_HDIM32(...)
#else
    #define INSTANTIATE_HDIM32(...) __VA_ARGS__
#endif


// 宏，为给定的头维度和数据类型实例化所有布尔组合。
#define INSTANTIATE_FWD_COMBOS_FOR_HEAD_DIM(ARCH, DTYPE, KHEADDIM) \
    /* 1. 标准 HSTU (Is_mfalcon = false, Has_mask = false) */ \
    INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/false, /*Is_mfalcon=*/false, /*Has_mask=*/false); \
    INSTANTIATE_IF_RAB_ENABLED( \
        INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/true,  /*Is_mfalcon=*/false, /*Has_mask=*/false); \
    ) \
    \
    /* 2. 独立 Mask (Is_mfalcon = false, Has_mask = true) */ \
    INSTANTIATE_IF_MASK_ENABLED( \
        INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/false, /*Is_mfalcon=*/false, /*Has_mask=*/true); \
        INSTANTIATE_IF_RAB_ENABLED( \
            INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/true,  /*Is_mfalcon=*/false, /*Has_mask=*/true); \
        ) \
    ) \
    \
    /* 3. M-Falcon (Is_mfalcon = true) */ \
    INSTANTIATE_IF_MFALCON_ENABLED( \
        /* 3a. M-Falcon 无掩码 */ \
        INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/false, /*Is_mfalcon=*/true,  /*Has_mask=*/false); \
        INSTANTIATE_IF_RAB_ENABLED( \
            INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/true,  /*Is_mfalcon=*/true,  /*Has_mask=*/false); \
        ) \
        /* 3b. M-Falcon 带掩码 */ \
        INSTANTIATE_IF_MASK_ENABLED( \
            INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/false, /*Is_mfalcon=*/true,  /*Has_mask=*/true); \
            INSTANTIATE_IF_RAB_ENABLED( \
                INSTANTIATE_FWD_KERNEL(ARCH, DTYPE, KHEADDIM, /*Has_rab=*/true,  /*Is_mfalcon=*/true,  /*Has_mask=*/true); \
            ) \
        ) \
    )

// 宏，为给定的数据类型实例化所有支持的头维度。
#define INSTANTIATE_FWD_FOR_DTYPE(ARCH, DTYPE) \
    INSTANTIATE_HDIM256(INSTANTIATE_FWD_COMBOS_FOR_HEAD_DIM(ARCH, DTYPE, 256);) \
    INSTANTIATE_HDIM128(INSTANTIATE_FWD_COMBOS_FOR_HEAD_DIM(ARCH, DTYPE, 128);) \
    INSTANTIATE_HDIM64(INSTANTIATE_FWD_COMBOS_FOR_HEAD_DIM(ARCH, DTYPE, 64);) \
    INSTANTIATE_HDIM32(INSTANTIATE_FWD_COMBOS_FOR_HEAD_DIM(ARCH, DTYPE, 32);)

// 主宏，为给定的架构实例化所有组合。
#define INSTANTIATE_FWD_FOR_ARCH(ARCH) \
    INSTANTIATE_FWD_FOR_DTYPE(ARCH, cutlass::half_t); \
    INSTANTIATE_FWD_FOR_DTYPE(ARCH, cutlass::bfloat16_t);

// ============================================================================
// A100 (SM80) 架构的实例化
// ============================================================================
INSTANTIATE_FWD_FOR_ARCH(80);