set(elewise_srcs
    ${CMAKE_CURRENT_LIST_DIR}/add/add_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cast/cast_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cos/cos_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/equal/equal_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/elewise_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/greater/greater_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/less/less_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/logical_and/logical_and_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/logical_not/logical_not_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/logical_or/logical_or_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mul/mul_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/muls/muls_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quant/quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quant/quant_tiling/quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_quant/dynamic_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_quant/dynamic_quant_tiling/dynamic_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/realdiv/realdiv_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/simple_broadcast/tiling/simple_broadcast_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/simple_broadcast/simple_broadcast_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sin/sin_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/elewise_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/neg/neg_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sub/sub_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tanh/tanh_kernel.cpp
)

add_operation(ElewiseOperation "${elewise_srcs}")

add_kernel(dynamic_quant ascend910b vector
    dynamic_quant/op_kernel/dynamic_quant.cpp
    DynamicQuantKernel)

add_kernel(dynamic_quant ascend310p vector
    dynamic_quant/op_kernel/dynamic_quant.cpp
    DynamicQuantKernel)

add_kernel(quant_per_channel ascend910b vector
    simple_broadcast/kernel/quant_per_channel.cpp
    AtbQuantPerChannelKernel)

add_kernel(dequant_per_channel ascend910b vector
    simple_broadcast/kernel/dequant_per_channel.cpp
    AtbDequantPerChannelKernel)

add_kernel(quant_per_channel ascend310p vector
    simple_broadcast/kernel/quant_per_channel.cpp
    AtbQuantPerChannelKernel)

add_kernel(quant_per_channel ascend910 vector
    simple_broadcast/kernel/quant_per_channel.cpp
    AtbQuantPerChannelKernel)

add_kernel(quant_per_channel ascend310b vector
    simple_broadcast/kernel/quant_per_channel.cpp
    AtbQuantPerChannelKernel)

add_kernel(quant ascend910b vector
        quant/op_kernel/quant.cpp
        QuantF16Kernel)

add_kernel(quant ascend310p vector
        quant/op_kernel/quant.cpp
        QuantF16Kernel)

add_kernel(quant ascend910 vector
        quant/op_kernel/quant.cpp
        QuantF16Kernel)