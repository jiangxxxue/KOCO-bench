set(moe_gate_corr_srcs
    ${CMAKE_CURRENT_LIST_DIR}/moe_gate_corr_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_gate_corr_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/moe_gate_corr_tiling.cpp
)

add_operation(MoeGateCorrOperation "${moe_gate_corr_srcs}")

add_kernel(moe_gate_corr_m16_n16_k2048 ascend910b cube
    op_kernel/moe_gate_corr_m16_n16_k2048.cpp
    MoeGateCorrM16N16K2048Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)

add_kernel(moe_gate_corr_m16_n32_k1024 ascend910b cube
    op_kernel/moe_gate_corr_m16_n32_k1024.cpp
    MoeGateCorrM16N32K1024Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)

add_kernel(moe_gate_corr_m32_n32_k1024 ascend910b cube
    op_kernel/moe_gate_corr_m32_n32_k1024.cpp
    MoeGateCorrM32N32K1024Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)

add_kernel(moe_gate_corr_m32_n48_k640 ascend910b cube
    op_kernel/moe_gate_corr_m32_n48_k640.cpp
    MoeGateCorrM32N48K640Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)

add_kernel(moe_gate_corr_m48_n48_k640 ascend910b cube
    op_kernel/moe_gate_corr_m48_n48_k640.cpp
    MoeGateCorrM48N48K640Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)

add_kernel(moe_gate_corr_m48_n64_k512 ascend910b cube
    op_kernel/moe_gate_corr_m48_n64_k512.cpp
    MoeGateCorrM48N64K512Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)

add_kernel(moe_gate_corr_m64_n64_k512 ascend910b cube
    op_kernel/moe_gate_corr_m64_n64_k512.cpp
    MoeGateCorrM64N64K512Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
)