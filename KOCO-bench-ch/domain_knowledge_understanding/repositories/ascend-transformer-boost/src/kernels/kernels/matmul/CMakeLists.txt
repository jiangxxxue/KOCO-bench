set(matmul_srcs
    ${CMAKE_CURRENT_LIST_DIR}/matmul_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/batch_matmul_kernel/batch_matmul_nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/batch_matmul_kernel/batch_matmul_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matmul_kernel/matmul_nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matmul_kernel/matmul_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_bf16_kernel/pp_matmul_bf16_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_bf16_nd_nz_nd_kernel/pp_matmul_bf16_nd_nz_nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_f16_mix_kernel/pp_matmul_f16_mix_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_f16_kernel/pp_matmul_f16_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_f16_opt_kernel/pp_matmul_f16_opt_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_i8_kernel/pp_matmul_i8_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_f16_m300_kernel/pp_matmul_f16_m300_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_i8_nz_kernel/pp_matmul_i8_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_i8_nz_compress_kernel/pp_matmul_i8_nz_compress_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_kernel/pp_matmul_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_accum_kernel/pp_matmul_accum_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_with_bias_kernel/pp_matmul_with_bias_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_ein_sum_kernel/pp_matmul_ein_sum_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/matmul_nd_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/matmul_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_mix_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_i8_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_nd_tiling.cpp
)

add_operation(MatMulOperation "${matmul_srcs}")

add_kernel(pp_matmul_add
    ascend910b mix pp_matmul_f16_mix_kernel/op_kernel/pp_matmul_mix.cce PpMatMulF16MixKernel)

add_kernel(pp_matmul
    ascend910b cube pp_matmul_f16_kernel/op_kernel/pp_matmul.cce PpMatMulF16Kernel)

add_kernel(pp_matmul_int8
    ascend910b cube pp_matmul_i8_kernel/op_kernel/pp_matmul.cce PpMatMulI8Kernel)

add_kernel(pp_matmul_f16_nz
    ascend310p cube pp_matmul_kernel/op_kernel/pp_matmul_f16_nz.cce PpMatMulNzF16Kernel)

add_kernel(pp_matmul_f16_nz
    ascend910 cube pp_matmul_kernel/op_kernel/pp_matmul_f16_nz.cce PpMatMulNzF16Kernel)

add_kernel(pp_matmul_int8_nz
    ascend310p cube pp_matmul_i8_nz_kernel/op_kernel/pp_matmul_310p.cce PpMatMulI8NzKernel)

add_kernel(pp_matmul_int8_nz
    ascend910 cube pp_matmul_i8_nz_kernel/op_kernel/pp_matmul_310p.cce PpMatMulI8NzKernel)

add_kernel(pp_matmul_int8_nz_compress ascend310p cube
    pp_matmul_i8_nz_compress_kernel/op_kernel/pp_matmul_i8_nz_compress.cce PpMatMulI8NzCompressKernel)

add_kernel(pp_matmul_f16_opt
    ascend910b cube pp_matmul_f16_opt_kernel/op_kernel/pp_matmul.cce PpMatMulF16OptKernel)

add_kernel(pp_matmul_bf16
    ascend910b cube pp_matmul_bf16_kernel/op_kernel/pp_matmul_bf16.cce PpMatMulBf16Kernel)

add_kernel(pp_matmul_bf16_nd_nz_nd
    ascend910b cube pp_matmul_bf16_nd_nz_nd_kernel/op_kernel/pp_matmul_bf16_nd_nz_nd.cce PpMatMulBf16NdNzNdKernel)

add_kernel(pp_matmul_int8_bf16
    ascend910b mix pp_matmul_i8_kernel/op_kernel/pp_matmul_bf16.cce PpMatMulI8Bf16Kernel)

add_kernel(pp_matmul_accum_atomic
    ascend910b cube pp_matmul_accum_kernel/op_kernel/pp_matmul_accum_atomic.cce PpMatmulAccumAtomicKernel)

add_kernel(pp_matmul_with_bias
    ascend910b cube pp_matmul_with_bias_kernel/op_kernel/pp_matmul_with_bias.cce PpMatmulWithBiasKernel)

add_kernel(pp_matmul_f16_nd
    ascend310b cube pp_matmul_f16_m300_kernel/op_kernel/pp_matmul_f16_nd.cce PpMatmulF16NdM300Kernel)

add_kernel(pp_matmul_f16_nz_m300
    ascend310b cube pp_matmul_f16_m300_kernel/op_kernel/pp_matmul_f16_nz.cce PpMatmulF16NzM300Kernel)

add_kernel(pp_matmul_int8
    ascend310b cube pp_matmul_i8_kernel/op_kernel/pp_matmul.cce PpMatMulI8Kernel)

add_kernel(pp_matmul_int8_nd_nz
    ascend310b cube pp_matmul_i8_kernel/op_kernel/pp_matmul_nz_m300.cce PpMatMulI8NdNzKernel)

add_kernel(pp_matmul_ein_sum
    ascend910b cube pp_matmul_ein_sum_kernel/op_kernel/pp_matmul_ein_sum.cce PpMatmulEinSumKernel)

add_kernel(pp_matmul_i8_weight_nz
    ascend910b cube pp_matmul_i8_kernel/op_kernel/pp_matmul_i8_weight_nz.cce PpMatMulI8WeightNzKernel)
