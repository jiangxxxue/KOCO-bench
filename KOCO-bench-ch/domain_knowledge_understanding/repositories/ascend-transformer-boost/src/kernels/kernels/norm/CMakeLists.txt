set(norm_srcs
    ${CMAKE_CURRENT_LIST_DIR}/norm_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/coherelayernorm/cohere_layer_norm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/coherelayernorm/tiling/cohere_layer_norm_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernorm/layernorm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernorm/tiling/layernorm_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernormstride/layernorm_stride_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernormstride/tiling/layer_norm_stride_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernormquant/layernorm_f16_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernormquant/tiling/layer_norm_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/adalayernorm/adalayernorm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/adalayernorm/tiling/adalayer_norm_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/postlayernorm/postlayernorm_f16_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/postlayernorm/postlayernorm_f16_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/postlayernorm/tiling/post_layer_norm_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/postrmsnorm/tiling/post_rms_norm_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/postrmsnorm/post_rms_norm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prelayernorm/pre_layer_norm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prermsnorm/pre_rms_norm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnorm/rms_norm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnorm/tiling/rms_norm_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnormbackward/rms_norm_backward_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnormbackward/tiling/rms_norm_grad_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnormforward/rms_norm_forward_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnormforward/tiling/rms_norm_forward_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnormquant/rms_norm_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsnormquant/tiling/rms_norm_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsprenormquant/rmsprenorm_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmsprenormquant/tiling/rms_pre_norm_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/normdynamicquant/tiling/norm_dynamic_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/normdynamicquant/norm_dynamic_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmspostnormquant/rmspostnorm_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rmspostnormquant/tiling/rms_post_norm_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gatherprermsnorm/gather_pre_rms_norm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gatherprermsnorm/tiling/gather_pre_rms_norm_tiling.cpp
)

add_operation(NormOperation "${norm_srcs}")

add_kernel(cohere_layer_norm
    ascend910b vector coherelayernorm/op_kernel/cohere_layer_norm.cpp CohereLayernormKernel)

add_kernel(cohere_layer_norm
    ascend310p vector coherelayernorm/op_kernel/cohere_layer_norm.cpp CohereLayernormKernel)

add_kernel(layer_norm_quant
    ascend910b vector layernormquant/op_kernel/layer_norm_quant.cpp LayerNormF16QuantKernel)

add_kernel(layer_norm_quant
    ascend310p vector layernormquant/op_kernel/layer_norm_quant.cpp LayerNormF16QuantKernel)

add_kernel(layer_norm_stride
    ascend910b vector layernormstride/op_kernel/layer_norm_stride.cpp LayerNormStrideKernel)

add_kernel(layer_norm_stride
    ascend310p vector layernormstride/op_kernel/layer_norm_stride.cpp LayerNormStrideKernel)

add_kernel(ada_layer_norm
    ascend910b vector adalayernorm/op_kernel/ada_layernorm.cpp AdaLayerNormKernel)

add_kernel(ada_layer_norm
    ascend310p vector adalayernorm/op_kernel/ada_layernorm.cpp AdaLayerNormKernel)

add_kernel(post_layer_norm
    ascend910b vector postlayernorm/op_kernel/post_layer_norm.cpp PostLayerNormF16Kernel)

add_kernel(post_layer_norm
    ascend310p vector postlayernorm/op_kernel/post_layer_norm.cpp PostLayerNormF16Kernel)

add_kernel(post_layer_norm_quant
    ascend910b vector postlayernorm/op_kernel/post_layer_norm_quant.cpp
    PostLayerNormF16QuantKernel)

add_kernel(post_layer_norm_quant
    ascend310p vector postlayernorm/op_kernel/post_layer_norm_quant.cpp
    PostLayerNormF16QuantKernel)

add_kernel(post_rms_norm
    ascend310p vector postrmsnorm/op_kernel/post_rms_norm.cpp
    PostRmsNormKernel)

add_kernel(post_rms_norm
    ascend910b vector postrmsnorm/op_kernel/post_rms_norm.cpp
    PostRmsNormKernel)

add_kernel(pre_rms_norm
    ascend310p vector prermsnorm/op_kernel/pre_rms_norm.cpp
    PreRmsNormKernel)

add_kernel(pre_rms_norm
    ascend910b vector prermsnorm/op_kernel/pre_rms_norm.cpp
    PreRmsNormKernel)

add_kernel(rms_norm
    ascend310p vector rmsnorm/op_kernel/rms_norm.cpp
    RmsNormKernel)

add_kernel(rms_norm
    ascend910 vector rmsnorm/op_kernel/rms_norm.cpp
    RmsNormKernel)

add_kernel(pre_layer_norm
    ascend910b vector prelayernorm/op_kernel/pre_layer_norm.cpp
    PreLayerNormKernel)

add_kernel(pre_layer_norm
    ascend310p vector prelayernorm/op_kernel/pre_layer_norm.cpp
    PreLayerNormKernel)

add_kernel(rms_norm
    ascend910b vector rmsnorm/op_kernel/rms_norm.cpp
    RmsNormKernel)

add_kernel(rms_norm_quant
    ascend910b vector rmsnormquant/op_kernel/rms_norm_quant.cpp
    RmsNormQuantKernel)

add_kernel(rms_norm_quant
    ascend910 vector rmsnormquant/op_kernel/rms_norm_quant.cpp
    RmsNormQuantKernel)

add_kernel(rms_norm_quant
    ascend310p vector rmsnormquant/op_kernel/rms_norm_quant.cpp
    RmsNormQuantKernel)

add_kernel(rms_pre_norm_quant
    ascend910b vector rmsprenormquant/op_kernel/rms_pre_norm_quant.cpp
    RmsPreNormQuantKernel)

add_kernel(rms_pre_norm_quant
    ascend310p vector rmsprenormquant/op_kernel/rms_pre_norm_quant.cpp
    RmsPreNormQuantKernel)

add_kernel(rms_pre_norm_quant
    ascend910 vector rmsprenormquant/op_kernel/rms_pre_norm_quant.cpp
    RmsPreNormQuantKernel)

add_kernel(rms_norm_forward ascend910b vector
    rmsnormforward/op_kernel/rms_norm.cpp
    RmsNormForwardKernel)

add_kernel(rms_norm_backward ascend910b vector
    rmsnormbackward/op_kernel/rms_norm_grad.cpp
    RmsNormBackwardKernel)

add_kernel(rms_norm_dynamic_quant
    ascend910b vector normdynamicquant/op_kernel/rmsnorm_dynamic_quant.cpp
    RmsNormDynamicQuantKernel)

add_kernel(rms_norm_dynamic_quant
    ascend310p vector normdynamicquant/op_kernel/rmsnorm_dynamic_quant.cpp
    RmsNormDynamicQuantKernel)

add_kernel(layer_norm_dynamic_quant
    ascend910b vector normdynamicquant/op_kernel/layernorm_dynamic_quant.cpp
    LayerNormDynamicQuantKernel)

add_kernel(layer_norm_dynamic_quant
    ascend310p vector normdynamicquant/op_kernel/layernorm_dynamic_quant.cpp
    LayerNormDynamicQuantKernel)

add_kernel(rms_post_norm_quant
    ascend910b vector rmspostnormquant/op_kernel/rms_post_norm_quant.cpp
    RmsPostNormQuantKernel)

add_kernel(rms_post_norm_quant
    ascend310p vector rmspostnormquant/op_kernel/rms_post_norm_quant.cpp
    RmsPostNormQuantKernel)

add_kernel(rms_norm
        ascend310b vector rmsnorm/op_kernel/rms_norm.cpp
        RmsNormKernel)

add_kernel(rms_norm_quant
        ascend310b vector rmsnormquant/op_kernel/rms_norm_quant.cpp
        RmsNormQuantKernel)

add_kernel(gather_pre_rms_norm
    ascend910b vector gatherprermsnorm/op_kernel/gather_pre_rms_norm.cpp
    GatherPreRmsNormKernel)