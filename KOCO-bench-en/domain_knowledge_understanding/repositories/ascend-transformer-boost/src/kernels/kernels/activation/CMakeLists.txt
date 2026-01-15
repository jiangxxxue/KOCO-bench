set(activation_srcs
    ${CMAKE_CURRENT_LIST_DIR}/activation_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fast_gelu/fast_gelu_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gelu/gelu_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log/log_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/relu/relu_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/swish/swish_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/swiglu_backward/swiglu_backward_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/swiglu_backward/tiling/swiglu_backward_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/swiglu_forward/swiglu_forward_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/swiglu_forward/tiling/swiglu_forward_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/faster_gelu_forward/faster_gelu_forward_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/faster_gelu_forward/tiling/faster_gelu_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sigmoid/sigmoid_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/activation_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gelu_forward/tiling/gelu_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gelu_forward/gelu_forward_kernel.cpp
)

add_operation(ActivationOperation "${activation_srcs}")

add_kernel(swiglu_forward ascend910b vector
    swiglu_forward/kernel/swiglu_forward.cpp
    SwiGluForwardKernel)

add_kernel(swiglu_forward ascend310p vector
    swiglu_forward/kernel/swiglu_forward.cpp
    SwiGluForwardKernel)

add_kernel(swiglu_forward ascend910 vector
    swiglu_forward/kernel/swiglu_forward.cpp
    SwiGluForwardKernel)

add_kernel(swiglu_backward ascend910b vector
    swiglu_backward/kernel/swiglu_backward.cpp
    SwiGluBackwardKernel)

add_kernel(faster_gelu_forward ascend310p vector
    faster_gelu_forward/kernel/faster_gelu_forward.cpp
    FasterGeluForwardKernel)

add_kernel(faster_gelu_forward ascend910 vector
    faster_gelu_forward/kernel/faster_gelu_forward.cpp
    FasterGeluForwardKernel)

add_kernel(faster_gelu_forward ascend910b vector
    faster_gelu_forward/kernel/faster_gelu_forward.cpp
    FasterGeluForwardKernel)

add_kernel(gelu_forward ascend310p vector
        gelu_forward/kernel/gelu_forward.cpp
        GeluForwardKernel)

add_kernel(gelu_forward ascend910b vector
        gelu_forward/kernel/gelu_forward.cpp
        GeluForwardKernel)

add_kernel(gelu_forward ascend910 vector
        gelu_forward/kernel/gelu_forward.cpp
        GeluForwardKernel)
