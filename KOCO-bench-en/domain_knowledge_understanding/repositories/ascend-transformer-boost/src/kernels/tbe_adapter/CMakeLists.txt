#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
include(${MKI_PACKAGE_DIR}/cmake/host_config.cmake)
remove_definitions(-DOpSpace=Mki)
file(GLOB_RECURSE SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/stubs/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/platform/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling_runner/*.cpp
)

set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
set(METADEF_DIR ${PROJECT_SOURCE_DIR}/3rdparty/metadef)
set(CANN_OPS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cann-ops-adv/src/transformer)
set(TBE_TILING_DIR ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling)

if("${PACKAGE_COMPILE}" STREQUAL "ON" AND NOT "${LOCAL_RELEASE_COMPILE}" STREQUAL "ON")
    set(ASDOPS_METADEF_FILES
        ${METADEF_DIR}/base/any_value.cc
        ${METADEF_DIR}/base/type/ascend_string_impl.cc
        ${METADEF_DIR}/base/utils/aligned_ptr.cc
        ${METADEF_DIR}/base/utils/type_utils_impl.cc
        ${METADEF_DIR}/base/runtime/compute_node_info.cc
        ${METADEF_DIR}/base/runtime/runtime_attrs.cc
        ${METADEF_DIR}/base/runtime/tiling_data.cc
        ${METADEF_DIR}/base/registry/op_impl_registry.cc
        ${METADEF_DIR}/exe_graph/lowering/getcdim.cc
        ${METADEF_DIR}/exe_graph/lowering/shape_utils.cc
        ${METADEF_DIR}/graph/attr/ge_attr_define.cc
        ${METADEF_DIR}/graph/normal_graph/tensor.cc
        ${METADEF_DIR}/graph/type/types.cc
        ${METADEF_DIR}/register/op_tiling/op_tiling_attr_utils.cc
        ${METADEF_DIR}/register/op_tiling/op_tiling_info.cc
        ${METADEF_DIR}/register/op_binary_resource_manager.cc
        ${METADEF_DIR}/register/tuning_bank_key_registry.cc
        ${METADEF_DIR}/register/tuning_tiling_registry.cc
        ${METADEF_DIR}/register/ascendc/tilingdata_base.cc
        ${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc
        ${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc
        ${METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc
        ${METADEF_DIR}/third_party/transformer/src/axis_util.cc
    )

    # Add tbe_tiling.cc here
    set(ASDOPS_TILING_FILES
        ${TBE_TILING_DIR}/auto_tiling_context.cc
        ${TBE_TILING_DIR}/auto_tiling_rt2.cc
        ${TBE_TILING_DIR}/broadcast_v3.cc
        ${TBE_TILING_DIR}/cache_tiling.cc
        ${TBE_TILING_DIR}/compress_dequant_cache_tiling.cc
        ${TBE_TILING_DIR}/concat_dsl.cc
        ${TBE_TILING_DIR}/cube_tiling_runtime.cc
        ${TBE_TILING_DIR}/elewise_v3.cc
        ${TBE_TILING_DIR}/fusion.cc
        ${TBE_TILING_DIR}/gather_dsl.cc
        ${TBE_TILING_DIR}/gemm_ub_cache_tiling.cc
        ${TBE_TILING_DIR}/gemm.cc
        ${TBE_TILING_DIR}/lock.cc
        ${TBE_TILING_DIR}/norm.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3_compile_info.cc
        ${TBE_TILING_DIR}/slice_dsl.cc
        ${TBE_TILING_DIR}/sort_dsl.cc
        ${TBE_TILING_DIR}/split_dsl.cc
        ${TBE_TILING_DIR}/trans_data_fz2fzg.cc
        ${TBE_TILING_DIR}/trans_data_fzg_to_fz.cc
        ${TBE_TILING_DIR}/transdata_dsl.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04_backward.cc
        ${TBE_TILING_DIR}/transdata_dsl_entrance.cc
        ${TBE_TILING_DIR}/transdata_dsl_borrow.cc
        ${TBE_TILING_DIR}/transdata_dsl_general.cc
        ${TBE_TILING_DIR}/transdata_dsl_util.cc
        ${TBE_TILING_DIR}/vector_op_info.cc
        ${TBE_TILING_DIR}/vector_tiling_key.cc
        ${TBE_TILING_DIR}/vector_tiling_rt2.cc
        ${TBE_TILING_DIR}/vector_tiling_util.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/runtime_bank_manager.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_hash.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_runtime_bank.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/configuration.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/file_utils.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/system_utils.cc
        ${TBE_TILING_DIR}/cube/algorithm/cache_tiling_impl.cc
        ${TBE_TILING_DIR}/cube/algorithm/calculator/calculator.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/shape.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/status.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/cache_runinfo.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/hash.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/tiling_cache.cc
        ${TBE_TILING_DIR}/cube/impl/cache_tiling.cc
        ${TBE_TILING_DIR}/cube/impl/cube_run_info.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling_param.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling.cc
        ${TBE_TILING_DIR}/cube/platform/instruction_param.cc
        ${TBE_TILING_DIR}/cube/platform/platform_info.cc
        ${TBE_TILING_DIR}/cube/util/cube_util.cc
        ${TBE_TILING_DIR}/cube/util/math_util.cc
        ${TBE_TILING_DIR}/cube/util/timer.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block_calc.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_align_count.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_request_bytes.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_cycle_model.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_basic_block_est.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_est.cc
        ${TBE_TILING_DIR}/runtime/as_strided.cc
        ${TBE_TILING_DIR}/runtime/as_strided_tiling.cc
        ${TBE_TILING_DIR}/runtime/cumsum.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling_ascendc.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling_ascendc_int.cc
        ${TBE_TILING_DIR}/runtime/concat_d.cc
        ${TBE_TILING_DIR}/runtime/concat_tiling.cc
        ${TBE_TILING_DIR}/runtime/expand.cc
        ${TBE_TILING_DIR}/runtime/elewise_tiling.cc
        ${TBE_TILING_DIR}/runtime/elementwise_template/fill_tiling.cc
        ${TBE_TILING_DIR}/runtime/gatherv2.cc
        ${TBE_TILING_DIR}/runtime/gather_v2_tiling.cpp
        ${TBE_TILING_DIR}/runtime/inplace_index_add.cc
        ${TBE_TILING_DIR}/runtime/inplace_index_add_tiling.cc
        ${TBE_TILING_DIR}/runtime/layer_norm_v3.cc
        ${TBE_TILING_DIR}/runtime/layer_norm_v3/layer_norm_v3_tiling.cc
        ${TBE_TILING_DIR}/runtime/layer_norm_v3/layer_norm_v3_tiling_base.cc
        ${OPS_THIRD_PARTY_DIR}/canndev/ops/norm/layer_norm_v4/op_host/layer_norm_v4_tiling.cpp
        ${OPS_THIRD_PARTY_DIR}/canndev/ops/norm/layer_norm_v4/op_host/layer_norm_v4_tiling_base.cpp
        ${TBE_TILING_DIR}/runtime/one_hot/one_hot.cc
        ${TBE_TILING_DIR}/runtime/one_hot/one_hot_tiling.cc
        ${TBE_TILING_DIR}/runtime/pack_tiling.cc
        ${TBE_TILING_DIR}/runtime/reverse.cc
        ${TBE_TILING_DIR}/runtime/reverse_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/tensor_move_tiling.cc
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/runtime/slice.cc
        ${TBE_TILING_DIR}/runtime/slice_tiling.cc
        ${TBE_TILING_DIR}/runtime/strided_slice_tiling.cc
        ${TBE_TILING_DIR}/runtime/top_k.cc
        ${TBE_TILING_DIR}/runtime/top_k_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/trans_data.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_ntc.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_tc_201.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_ntc_100.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1010.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1011.cc
        ${TBE_TILING_DIR}/runtime/transdata_tiling.cc
        ${TBE_TILING_DIR}/runtime/transpose.cc
        ${TBE_TILING_DIR}/runtime/transpose_tiling.cc
        ${TBE_TILING_DIR}/runtime/depth_to_space_tiling.cc
        ${TBE_TILING_DIR}/runtime/space_to_depth_tiling.cc
        ${TBE_TILING_DIR}/runtime/view_copy.cc
        ${TBE_TILING_DIR}/runtime/view_copy_tiling.cc
        ${OPS_THIRD_PARTY_DIR}/canndev/ops/index/scatter_elements_v2/op_host/scatter_elements_v2_tiling.cpp
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/../fusion_pass/common/fp16_t.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/op_util.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/getcdim.cc
    )
else()
    set(ASDOPS_METADEF_FILES
        ${METADEF_DIR}/exe_graph/lowering/getcdim.cc
        ${METADEF_DIR}/exe_graph/lowering/shape_utils.cc
        ${METADEF_DIR}/exe_graph/runtime/compute_node_info.cc
        ${METADEF_DIR}/exe_graph/runtime/runtime_attrs.cc
        ${METADEF_DIR}/exe_graph/runtime/tiling_data.cc
        ${METADEF_DIR}/graph/attr/ge_attr_define.cc
        ${METADEF_DIR}/graph/normal_graph/any_value.cc
        ${METADEF_DIR}/graph/normal_graph/tensor.cc
        ${METADEF_DIR}/graph/type/ascend_string.cc
        ${METADEF_DIR}/graph/type/types.cc
        ${METADEF_DIR}/graph/utils/aligned_ptr.cc
        ${METADEF_DIR}/graph/utils/type_utils.cc
        ${METADEF_DIR}/graph/utils/type_utils_ex.cc
        ${METADEF_DIR}/register/op_impl_registry.cc
        ${METADEF_DIR}/register/op_tiling/op_tiling_attr_utils.cc
        ${METADEF_DIR}/register/op_tiling/op_tiling_info.cc
        ${METADEF_DIR}/register/op_binary_resource_manager.cc
        ${METADEF_DIR}/register/tuning_bank_key_registry.cc
        ${METADEF_DIR}/register/tuning_tiling_registry.cc
        ${METADEF_DIR}/register/ascendc/tilingdata_base.cc
        ${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc
        ${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc
        ${METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc
        ${METADEF_DIR}/third_party/transformer/src/axis_util.cc
    )

    # Add tbe_tiling.cc here
    set(ASDOPS_TILING_FILES
        ${TBE_TILING_DIR}/auto_tiling_context.cc
        ${TBE_TILING_DIR}/auto_tiling_rt2.cc
        ${TBE_TILING_DIR}/broadcast_v3.cc
        ${TBE_TILING_DIR}/cache_tiling.cc
        ${TBE_TILING_DIR}/compress_dequant_cache_tiling.cc
        ${TBE_TILING_DIR}/concat_dsl.cc
        ${TBE_TILING_DIR}/cube_tiling_runtime.cc
        ${TBE_TILING_DIR}/elewise_v3.cc
        ${TBE_TILING_DIR}/fusion.cc
        ${TBE_TILING_DIR}/gather_dsl.cc
        ${TBE_TILING_DIR}/gemm_ub_cache_tiling.cc
        ${TBE_TILING_DIR}/gemm.cc
        ${TBE_TILING_DIR}/lock.cc
        ${TBE_TILING_DIR}/norm.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3_compile_info.cc
        ${TBE_TILING_DIR}/slice_dsl.cc
        ${TBE_TILING_DIR}/sort_dsl.cc
        ${TBE_TILING_DIR}/split_dsl.cc
        ${TBE_TILING_DIR}/trans_data_fz2fzg.cc
        ${TBE_TILING_DIR}/trans_data_fzg_to_fz.cc
        ${TBE_TILING_DIR}/transdata_dsl.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04_backward.cc
        ${TBE_TILING_DIR}/transdata_dsl_entrance.cc
        ${TBE_TILING_DIR}/transdata_dsl_borrow.cc
        ${TBE_TILING_DIR}/transdata_dsl_general.cc
        ${TBE_TILING_DIR}/transdata_dsl_util.cc
        ${TBE_TILING_DIR}/vector_op_info.cc
        ${TBE_TILING_DIR}/vector_tiling_key.cc
        ${TBE_TILING_DIR}/vector_tiling_rt2.cc
        ${TBE_TILING_DIR}/vector_tiling_util.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/runtime_bank_manager.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_hash.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_runtime_bank.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/configuration.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/file_utils.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/system_utils.cc
        ${TBE_TILING_DIR}/cube/algorithm/cache_tiling_impl.cc
        ${TBE_TILING_DIR}/cube/algorithm/calculator/calculator.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/shape.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/status.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/cache_runinfo.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/hash.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/tiling_cache.cc
        ${TBE_TILING_DIR}/cube/impl/cache_tiling.cc
        ${TBE_TILING_DIR}/cube/impl/cube_run_info.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling_param.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling.cc
        ${TBE_TILING_DIR}/cube/platform/instruction_param.cc
        ${TBE_TILING_DIR}/cube/platform/platform_info.cc
        ${TBE_TILING_DIR}/cube/util/cube_util.cc
        ${TBE_TILING_DIR}/cube/util/math_util.cc
        ${TBE_TILING_DIR}/cube/util/timer.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block_calc.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_align_count.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_request_bytes.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_cycle_model.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_basic_block_est.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_est.cc
        ${TBE_TILING_DIR}/runtime/as_strided.cc
        ${TBE_TILING_DIR}/runtime/as_strided_tiling.cc
        ${TBE_TILING_DIR}/runtime/cumsum.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling_ascendc.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling_ascendc_int.cc
        ${TBE_TILING_DIR}/runtime/concat_d.cc
        ${TBE_TILING_DIR}/runtime/concat_tiling.cc
        ${TBE_TILING_DIR}/runtime/expand.cc
        ${TBE_TILING_DIR}/runtime/elewise_tiling.cc
        ${TBE_TILING_DIR}/runtime/elementwise_template/fill_tiling.cc
        ${TBE_TILING_DIR}/runtime/gatherv2.cc
        ${TBE_TILING_DIR}/runtime/gather_v2_tiling.cpp
        ${TBE_TILING_DIR}/runtime/inplace_index_add.cc
        ${TBE_TILING_DIR}/runtime/inplace_index_add_tiling.cc
        ${TBE_TILING_DIR}/runtime/layer_norm_v3.cc
        ${TBE_TILING_DIR}/runtime/one_hot/one_hot.cc
        ${TBE_TILING_DIR}/runtime/one_hot/one_hot_tiling.cc
        ${TBE_TILING_DIR}/runtime/pack_tiling.cc
        ${TBE_TILING_DIR}/runtime/reverse.cc
        ${TBE_TILING_DIR}/runtime/reverse_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/tensor_move_tiling.cc
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/runtime/slice.cc
        ${TBE_TILING_DIR}/runtime/slice_tiling.cc
        ${TBE_TILING_DIR}/runtime/strided_slice_tiling.cc
        ${TBE_TILING_DIR}/runtime/top_k.cc
        ${TBE_TILING_DIR}/runtime/top_k_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/trans_data.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_ntc.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_tc_201.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_ntc_100.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1010.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1011.cc
        ${TBE_TILING_DIR}/runtime/transdata_tiling.cc
        ${TBE_TILING_DIR}/runtime/transpose.cc
        ${TBE_TILING_DIR}/runtime/transpose_tiling.cc
        ${TBE_TILING_DIR}/runtime/depth_to_space_tiling.cc
        ${TBE_TILING_DIR}/runtime/space_to_depth_tiling.cc
        ${TBE_TILING_DIR}/runtime/view_copy.cc
        ${TBE_TILING_DIR}/runtime/view_copy_tiling.cc
        ${TBE_TILING_DIR}/runtime/scatter_elements_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/../fusion_pass/common/fp16_t.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/op_util.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/getcdim.cc
    )
endif()

# Sink kernels
file(GLOB_RECURSE SINK_TILING_SRCS
    ${TBE_TILING_DIR}/runtime/rotary_pos_emb_infer/*
)
list(APPEND ASDOPS_TILING_FILES ${SINK_TILING_SRCS})

# Tiling Api
set(TILING_API_BASE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/api)
set(TILING_API_SRC_DIR ${PROJECT_SOURCE_DIR}/3rdparty/api/impl)

# Generate kernel_tiling.h
set(GEN_KERENL_TILING_DATA_SCRIPT ${TILING_API_BASE_DIR}/cmake/scripts/gen_kernel_tiling_data_def.py)
set(TILING_DATA_DEF_DIR ${TILING_API_BASE_DIR}/lib)
set(KERNEL_TILING_HEAD  ${CMAKE_BINARY_DIR}/generated_include/kernel_tiling/kernel_tiling.h)

add_custom_command(OUTPUT ${KERNEL_TILING_HEAD}
    COMMAND python3 ${GEN_KERENL_TILING_DATA_SCRIPT} ${TILING_DATA_DEF_DIR} ${KERNEL_TILING_HEAD}
    DEPENDS ${GEN_KERENL_TILING_DATA_SCRIPT})

add_custom_target(GEN_KERNEL_TILING ALL
    DEPENDS ${KERNEL_TILING_HEAD})

if("${PACKAGE_COMPILE}" STREQUAL "ON" AND NOT "${LOCAL_RELEASE_COMPILE}" STREQUAL "ON")
    set(TILING_API_SRCS
        ${TILING_API_SRC_DIR}/quantization/dequant/ascend_dequant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/quantization/quant/ascend_quant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/quantization/antiquant/ascend_antiquant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/filter/dropout/dropout_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/gelu/gelu_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/bmm_tiling.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling_base.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling_algorithm.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/math_util.cpp
        ${TILING_API_SRC_DIR}/hccl/tiling/hccl_tiling.cpp
        ${TILING_API_SRC_DIR}/math/clamp/clamp_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/acos/acos_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/acosh/acosh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/asin/asin_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/asinh/asinh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/atan/atan_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/atanh/atanh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/cos/cos_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/cosh/cosh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/erf/erf_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/erfc/erfc_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/exp/exp_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/frac/frac_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/geglu/geglu_tiling.cpp
        ${TILING_API_SRC_DIR}/math/lgamma/lgamma_tiling.cpp
        ${TILING_API_SRC_DIR}/math/digamma/digamma_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/log/log_tiling.cpp
        ${TILING_API_SRC_DIR}/math/sin/sin_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/sinh/sinh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/power/power_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/sigmoid/sigmoid_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/round/round_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/tan/tan_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/tanh/tanh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/trunc/trunc_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/axpy/axpy_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/hypot/hypot_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/swiglu/swiglu_tiling.cpp
        ${TILING_API_SRC_DIR}/math/ceil/ceil_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/floor/floor_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/softmax/softmax_tiling.cpp
        ${TILING_API_SRC_DIR}/activation/softmax/logsoftmax_tiling.cpp
        ${TILING_API_SRC_DIR}/normalization/rmsnorm/rmsnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/batchnorm/batchnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/sort/sort/sort_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/sort/topk/topk_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/deepnorm/deepnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/select/selectwithbytesmask/selectwithbytesmask_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernorm/layernorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/normalize/normalize_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernormgrad/layernorm_grad_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernormgrad/layernorm_grad_beta_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/groupnorm/groupnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/welfordfinalize/welfordfinalize_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/pad/pad/pad_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/transpose/confusion_transpose/confusion_transpose_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/pad/broadcast/broadcast_tiling.cpp
        ${TILING_API_SRC_DIR}/pad/broadcast/broadcast_tiling.cpp
        ${TILING_API_SRC_DIR}/math/xor/xor_tiling.cpp
        ${TILING_API_SRC_DIR}/math/cumsum/cumsum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/mean/mean_tiling.cpp
        ${TILING_API_SRC_DIR}/math/sign/sign_tiling.cpp
        ${TILING_API_SRC_DIR}/activation/reglu/reglu_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/reduce/reduce_xor_sum/reduce_xor_sum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/sum/sum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/reduce_tiling.cpp
        ${TILING_API_SRC_DIR}/index/arithprogression/arithprogression_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/fmod/fmod_tiling_impl.cpp
    )
else()
    set(TILING_API_SRCS
        ${TILING_API_SRC_DIR}/quantization/dequant/ascend_dequant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/quantization/quant/ascend_quant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/quantization/antiquant/ascend_antiquant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/filter/dropout/dropout_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/gelu/gelu_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/bmm_tiling.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling_base.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling_algorithm.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/math_util.cpp
        ${TILING_API_SRC_DIR}/hccl/hccl_tiling.cpp
        ${TILING_API_SRC_DIR}/math/clamp/clamp_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/acos/acos_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/acosh/acosh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/asin/asin_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/asinh/asinh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/atan/atan_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/atanh/atanh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/cos/cos_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/cosh/cosh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/erf/erf_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/erfc/erfc_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/exp/exp_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/frac/frac_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/geglu/geglu_tiling.cpp
        ${TILING_API_SRC_DIR}/math/lgamma/lgamma_tiling.cpp
        ${TILING_API_SRC_DIR}/math/digamma/digamma_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/log/log_tiling.cpp
        ${TILING_API_SRC_DIR}/math/sin/sin_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/sinh/sinh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/power/power_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/sigmoid/sigmoid_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/round/round_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/tan/tan_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/tanh/tanh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/trunc/trunc_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/axpy/axpy_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/hypot/hypot_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/swiglu/swiglu_tiling.cpp
        ${TILING_API_SRC_DIR}/math/ceil/ceil_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/floor/floor_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/softmax/softmax_tiling.cpp
        ${TILING_API_SRC_DIR}/activation/softmax/logsoftmax_tiling.cpp
        ${TILING_API_SRC_DIR}/normalization/rmsnorm/rmsnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/batchnorm/batchnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/sort/sort/sort_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/sort/topk/topk_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/deepnorm/deepnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/select/selectwithbytesmask/selectwithbytesmask_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernorm/layernorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/normalize/normalize_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernormgrad/layernorm_grad_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernormgrad/layernorm_grad_beta_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/groupnorm/groupnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/welfordfinalize/welfordfinalize_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/pad/pad/pad_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/transpose/confusion_transpose/confusion_transpose_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/pad/broadcast/broadcast_tiling.cpp
        ${TILING_API_SRC_DIR}/pad/broadcast/broadcast_tiling.cpp
        ${TILING_API_SRC_DIR}/math/xor/xor_tiling.cpp
        ${TILING_API_SRC_DIR}/math/cumsum/cumsum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/mean/mean_tiling.cpp
        ${TILING_API_SRC_DIR}/math/sign/sign_tiling.cpp
        ${TILING_API_SRC_DIR}/activation/reglu/reglu_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/reduce/reduce_xor_sum/reduce_xor_sum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/sum/sum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/reduce_tiling.cpp
        ${TILING_API_SRC_DIR}/index/arithprogression/arithprogression_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/fmod/fmod_tiling_impl.cpp
    )
endif()

if("${PACKAGE_COMPILE}" STREQUAL "ON" AND NOT "${LOCAL_RELEASE_COMPILE}" STREQUAL "ON")
    set(ASDOPS_INC_DIRS
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc/graph
        ${CMAKE_CURRENT_LIST_DIR}/platform
        ${CMAKE_CURRENT_LIST_DIR}/platform/tiling
        ${KERNEL_TILING_DIR}
        ${TILING_API_BASE_DIR}
        ${TILING_API_BASE_DIR}/tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/conversion/fill/op_kernel
        ${METADEF_DIR}
        ${METADEF_DIR}/base/runtime
        ${METADEF_DIR}/exe_graph/lowering
        ${METADEF_DIR}/exe_graph/runtime
        ${METADEF_DIR}/graph
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/common
        ${METADEF_DIR}/inc/exe_graph
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/external/exe_graph
        ${METADEF_DIR}/inc/external/exe_graph/lowering
        ${METADEF_DIR}/inc/external/exe_graph/runtime
        ${METADEF_DIR}/inc/external/graph
        ${METADEF_DIR}/register
        ${METADEF_DIR}/third_party/transformer/inc
        ${METADEF_DIR}/third_party/transformer/src
        ${METADEF_DIR}/inc/common/ge_common/debug
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/utils/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling/runtime
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/third_party/fwkacllib/inc
        ${METADEF_DIR}/inc/graph
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/tiling
    )
else()
    set(ASDOPS_INC_DIRS
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc/graph
        ${CMAKE_CURRENT_LIST_DIR}/platform
        ${CMAKE_CURRENT_LIST_DIR}/platform/tiling
        ${KERNEL_TILING_DIR}
        ${TILING_API_BASE_DIR}
        ${TILING_API_BASE_DIR}/tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/conversion/fill/op_kernel
        ${METADEF_DIR}
        ${METADEF_DIR}/exe_graph/lowering
        ${METADEF_DIR}/exe_graph/runtime
        ${METADEF_DIR}/graph
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/common
        ${METADEF_DIR}/inc/exe_graph
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/external/exe_graph
        ${METADEF_DIR}/inc/external/exe_graph/lowering
        ${METADEF_DIR}/inc/external/exe_graph/runtime
        ${METADEF_DIR}/inc/external/graph
        ${METADEF_DIR}/register
        ${METADEF_DIR}/third_party/transformer/inc
        ${METADEF_DIR}/inc/common/ge_common/debug
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/utils/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling/runtime
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/third_party/fwkacllib/inc
        ${METADEF_DIR}/inc/graph
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/tiling
    )
endif()

# Ignore warnings from 3rdparty srcs
# DO NOT USE THIS FOR OTHER SRCS
set_source_files_properties(
    ${TILING_API_SRCS}
    ${ASDOPS_TILING_FILES}
    ${ASDOPS_METADEF_FILES}
    PROPERTIES
    COMPILE_FLAGS "-w"
)

# MIX OPS
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/mixops/)
execute_process(COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/update_tbe_tactic_json.py
                                --src_ini_path ${PROJECT_SOURCE_DIR}/src/kernels/configs/mixops/tbe_tactic_info.ini
                                --dst_ini_path ${CMAKE_BINARY_DIR}/mixops/tbe_tactic_json.ini
                OUTPUT_VARIABLE MIX_PYTHON_OUTPUT
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "tbe info update failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()
 
string(REPLACE "\n" ";" MIX_REUSE_BINARY_LIST "${MIX_PYTHON_OUTPUT}")
list(POP_BACK MIX_REUSE_BINARY_LIST)
set_source_files_properties(${MIX_REUSE_BINARY_LIST} PROPERTIES GENERATED TRUE)
 
add_custom_command(
    OUTPUT ${MIX_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp
    DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/configs/mixops/tbe_tactic_info.ini
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type tbe
                    --tbe_ini_path ${CMAKE_BINARY_DIR}/mixops/tbe_tactic_json.ini
    COMMAND cmake -E sleep 10
    COMMAND cmake -E echo "wait 10 sec done"
    COMMAND cmake -E touch ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp
)
 
add_custom_target(MIX_REUSE_SRC_TARGET ALL
    DEPENDS ${MIX_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp
)
add_library(mix_reuse_kernels OBJECT ${MIX_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp)
add_dependencies(mix_reuse_kernels MIX_REUSE_SRC_TARGET)
target_compile_definitions(mix_reuse_kernels PRIVATE OpSpace=AtbOps)
 
# Ops
execute_process(COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/update_tbe_tactic_json.py
                OUTPUT_VARIABLE OPS_PYTHON_OUTPUT
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "tbe info update failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()
 
string(REPLACE "\n" ";" OPS_REUSE_BINARY_LIST "${OPS_PYTHON_OUTPUT}")
list(POP_BACK OPS_REUSE_BINARY_LIST)
set_source_files_properties(${OPS_REUSE_BINARY_LIST} PROPERTIES GENERATED TRUE)
 
add_custom_command(
    OUTPUT ${OPS_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp
    DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/configs/ops/tbe_tactic_info.ini
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type tbe
                    --tbe_ini_path ${CMAKE_BINARY_DIR}/tbe_tactic_json.ini
    COMMAND cmake -E sleep 10
    COMMAND cmake -E echo "wait 10 sec done"
    COMMAND cmake -E touch ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp
)
 
add_custom_target(OPS_REUSE_SRC_TARGET ALL
    DEPENDS ${OPS_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp
)
add_library(ops_reuse_kernels OBJECT ${OPS_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp)
add_dependencies(ops_reuse_kernels OPS_REUSE_SRC_TARGET)
target_compile_definitions(ops_reuse_kernels PRIVATE OpSpace=AsdOps)
 
# Target
add_library(tbe_adapter SHARED ${SOURCE_FILES} ${ASDOPS_METADEF_FILES} ${ASDOPS_TILING_FILES} ${TILING_API_SRCS})
add_dependencies(tbe_adapter GEN_KERNEL_TILING)
target_include_directories(tbe_adapter PRIVATE ${ASDOPS_INC_DIRS})
target_link_libraries(tbe_adapter PRIVATE mmpa c_sec ascendalog mix_reuse_kernels ops_reuse_kernels)

add_definitions(-DLOG_CPP)

target_compile_definitions(tbe_adapter PUBLIC
    OPS_UTILS_LOG_SUB_MOD_NAME="ASDOPS_TILING"
)

target_compile_definitions(tbe_adapter PUBLIC
    fe=AsdOpsFe
    ge=AsdOpsGe
    gert=AsdOpsGeRt
    optiling=AsdOpsTiling
    error_message=AsdOpsErrorMessage
    ErrorManager=AsdOpsErrorManager
    td_dsl=AsdOpsTdDsl
    tuningtiling=AsdOpsTuningTiling
    RuntimeKb=AsdOpsRuntimeKb
    gemm_cache_tiling=AsdOpsGemmCacheTiling
    transformer=AsdOpsTransformer
    ops=AsdOpsOps
    AscendC=AsdOpsAscendC
    matmul_tiling=AsdOpsMatmulTiling
    platform_ascendc=AsdOpsPlatformAscendC
)

install(TARGETS tbe_adapter DESTINATION lib)
