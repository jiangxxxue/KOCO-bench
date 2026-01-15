#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch_atb
import unittest
import logging

# 模糊校验小数点后6位相同，即c++ float默认单精度
decimal_palace = 6
message = "almost equal assertion failed"

class TestOpParam(unittest.TestCase):
    def test_layer_norm(self):
        layer_norm_param = torch_atb.LayerNormParam()

        expected_layer_type = 0

        expected_norm_param_begin_norm_axis = 0
        expected_norm_param_begin_params_axis = 0
        expected_norm_param_epsilon = 1e-5
        expected_norm_param_quant_type = 0
        expected_post_norm_param_epsilon = 1e-5
        expected_post_norm_param_op_mode = 0
        expected_post_norm_param_quant_type = 0
        expected_post_norm_param_zoom_scale_value = 1.0
        expected_pre_norm_param_epsilon = 1e-5
        expected_pre_norm_param_op_mode = 0
        expected_pre_norm_param_quant_type = 0
        expected_pre_norm_param_zoom_scale_value = 1.0

        self.assertEqual(layer_norm_param.layer_type, expected_layer_type)
        self.assertEqual(layer_norm_param.norm_param.begin_norm_axis, expected_norm_param_begin_norm_axis)
        self.assertEqual(layer_norm_param.norm_param.begin_params_axis, expected_norm_param_begin_params_axis)
        self.assertAlmostEqual(layer_norm_param.norm_param.epsilon, expected_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(layer_norm_param.norm_param.quant_type, expected_norm_param_quant_type)
        self.assertAlmostEqual(layer_norm_param.post_norm_param.epsilon, expected_post_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(layer_norm_param.post_norm_param.op_mode, expected_post_norm_param_op_mode)
        self.assertEqual(layer_norm_param.post_norm_param.quant_type, expected_post_norm_param_quant_type)
        self.assertEqual(layer_norm_param.post_norm_param.zoom_scale_value, expected_post_norm_param_zoom_scale_value)
        self.assertAlmostEqual(layer_norm_param.pre_norm_param.epsilon, expected_pre_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(layer_norm_param.pre_norm_param.op_mode, expected_pre_norm_param_op_mode)
        self.assertEqual(layer_norm_param.pre_norm_param.quant_type, expected_pre_norm_param_quant_type)
        self.assertEqual(layer_norm_param.pre_norm_param.zoom_scale_value, expected_pre_norm_param_zoom_scale_value)

    def test_elewise(self):
        elewise_param = torch_atb.ElewiseParam()

        expected_elewise_type = 0
        expected_var_attr = 0.0
        expected_tensor_type = -1
        expected_asymmetric = False
        expected_input_offset = 0
        expected_input_scale = 1.0

        self.assertEqual(elewise_param.elewise_type, expected_elewise_type)
        self.assertEqual(elewise_param.muls_param.var_attr, expected_var_attr)
        self.assertEqual(elewise_param.out_tensor_type, expected_tensor_type)
        self.assertEqual(elewise_param.quant_param.asymmetric, expected_asymmetric)
        self.assertEqual(elewise_param.quant_param.input_offset, expected_input_offset)
        self.assertEqual(elewise_param.quant_param.input_scale, expected_input_scale)

    def test_linear(self):
        linear_param = torch_atb.LinearParam()

        expected_en_accum = False
        expected_has_bias = True
        expected_out_data_type = -1
        expected_transpose_a = False
        expected_transpose_b = True
        expected_matmul_type = 0
        expected_quant_mode = 0
 
        self.assertEqual(linear_param.en_accum, expected_en_accum)
        self.assertEqual(linear_param.has_bias, expected_has_bias)
        self.assertEqual(linear_param.out_data_type, expected_out_data_type)
        self.assertEqual(linear_param.transpose_a, expected_transpose_a)
        self.assertEqual(linear_param.transpose_b, expected_transpose_b)
        self.assertEqual(linear_param.matmul_type, expected_matmul_type)
        self.assertEqual(linear_param.quant_mode, expected_quant_mode)
 
    def test_softmax(self):
        softmax_param = torch_atb.SoftmaxParam()
        expected_axes = []

        self.assertEqual(softmax_param.axes, expected_axes)

    def test_self_attention(self):
        self_attention_param = torch_atb.SelfAttentionParam()

        expected_batch_run_status_enable = False
        expected_cache_type = 0
        expected_calc_type = 0
        expected_clamp_max = 0.0
        expected_clamp_min = 0.0
        expected_clamp_type = 0
        expected_head_num = 0
        expected_input_layout = 0
        expected_is_triu_mask = 0
        expected_kernel_type = 0
        expected_kv_head_num = 0
        expected_kvcache_cfg = 0
        expected_mask_type = 0
        expected_mla_v_head_size = 0
        expected_out_data_type = -1
        expected_q_scale = 1.0
        expected_qk_scale = 1.0
        expected_quant_type = 0
        expected_scale_type = 0
        expected_window_size = 0

        self.assertEqual(self_attention_param.batch_run_status_enable, expected_batch_run_status_enable)
        self.assertEqual(self_attention_param.cache_type, expected_cache_type)
        self.assertEqual(self_attention_param.calc_type, expected_calc_type)
        self.assertEqual(self_attention_param.clamp_max, expected_clamp_max)
        self.assertEqual(self_attention_param.clamp_min, expected_clamp_min)
        self.assertEqual(self_attention_param.clamp_type, expected_clamp_type)
        self.assertEqual(self_attention_param.head_num, expected_head_num)
        self.assertEqual(self_attention_param.input_layout, expected_input_layout)
        self.assertEqual(self_attention_param.is_triu_mask, expected_is_triu_mask)
        self.assertEqual(self_attention_param.kernel_type, expected_kernel_type)
        self.assertEqual(self_attention_param.kv_head_num, expected_kv_head_num)
        self.assertEqual(self_attention_param.kvcache_cfg, expected_kvcache_cfg)
        self.assertEqual(self_attention_param.mask_type, expected_mask_type)
        self.assertEqual(self_attention_param.mla_v_head_size, expected_mla_v_head_size)
        self.assertEqual(self_attention_param.out_data_type, expected_out_data_type)
        self.assertEqual(self_attention_param.q_scale, expected_q_scale)
        self.assertEqual(self_attention_param.qk_scale, expected_qk_scale)
        self.assertEqual(self_attention_param.quant_type, expected_quant_type)
        self.assertEqual(self_attention_param.scale_type, expected_scale_type)
        self.assertEqual(self_attention_param.window_size, expected_window_size)

    def test_rope(self):
        rope_param = torch_atb.RopeParam()

        expected_cos_format = 0
        expected_rotary_coeff = 4

        self.assertEqual(rope_param.cos_format, expected_cos_format)
        self.assertEqual(rope_param.rotary_coeff, expected_rotary_coeff)

    def test_split(self):
        split_param = torch_atb.SplitParam()

        expected_split_dim = 0
        expected_split_num = 2
        expected_split_sizes = []

        self.assertEqual(split_param.split_dim, expected_split_dim)
        self.assertEqual(split_param.split_num, expected_split_num)
        self.assertEqual(split_param.split_sizes, expected_split_sizes)

    def test_gather(self):
        gather_param = torch_atb.GatherParam()

        expected_axis = 0
        expected_batch_dims = 0

        self.assertEqual(gather_param.axis, expected_axis)
        self.assertEqual(gather_param.batch_dims, expected_batch_dims)

    def test_activation(self):
        activation_param = torch_atb.ActivationParam()

        expected_activation_type = 0
        expected_dim = -1
        expected_scale = 1.0

        self.assertEqual(activation_param.activation_type, expected_activation_type)
        self.assertEqual(activation_param.dim, expected_dim)
        self.assertEqual(activation_param.scale, expected_scale)

    def test_rms_norm(self):
        rms_norm_param = torch_atb.RmsNormParam()

        expected_layer_type = 0
        expected_norm_param_epsilon = 1e-5
        expected_norm_param_layer_norm_eps = 1e-5
        expected_norm_param_model_type = 0
        expected_norm_param_precision_mode = 0
        expected_norm_param_quant_type = 0
        expected_norm_param_rstd = False
        expected_norm_param_dynamic_quant_type = 0
        expected_post_norm_param_epsilon = 1e-5
        expected_post_norm_param_has_bias = False
        expected_post_norm_param_quant_type = 0
        expected_pre_norm_param_epsilon = 1e-5
        expected_pre_norm_param_has_bias = False
        expected_pre_norm_param_quant_type = 0

        self.assertEqual(rms_norm_param.layer_type, expected_layer_type)
        self.assertAlmostEqual(rms_norm_param.norm_param.epsilon, expected_norm_param_epsilon, decimal_palace, message)
        self.assertAlmostEqual(rms_norm_param.norm_param.layer_norm_eps, expected_norm_param_layer_norm_eps, decimal_palace, message)
        self.assertEqual(rms_norm_param.norm_param.model_type, expected_norm_param_model_type)
        self.assertEqual(rms_norm_param.norm_param.precision_mode, expected_norm_param_precision_mode)
        self.assertEqual(rms_norm_param.norm_param.quant_type, expected_norm_param_quant_type)
        self.assertEqual(rms_norm_param.norm_param.rstd, expected_norm_param_rstd)
        self.assertEqual(rms_norm_param.norm_param.dynamic_quant_type, expected_norm_param_dynamic_quant_type)
        self.assertAlmostEqual(rms_norm_param.post_norm_param.epsilon, expected_post_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(rms_norm_param.post_norm_param.has_bias, expected_post_norm_param_has_bias)
        self.assertEqual(rms_norm_param.post_norm_param.quant_type, expected_post_norm_param_quant_type)
        self.assertAlmostEqual(rms_norm_param.pre_norm_param.epsilon, expected_pre_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(rms_norm_param.pre_norm_param.has_bias, expected_pre_norm_param_has_bias)
        self.assertEqual(rms_norm_param.pre_norm_param.quant_type, expected_pre_norm_param_quant_type)

    def test_all_gather(self):
        all_gather_param = torch_atb.AllGatherParam()

        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_backend = "hccl"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_rank_table_file = ""
        expected_comm_domain = ""

        self.assertEqual(all_gather_param.rank, expected_rank)
        self.assertEqual(all_gather_param.rank_size, expected_rank_size)
        self.assertEqual(all_gather_param.rank_root, expected_rank_root)
        self.assertEqual(all_gather_param.backend, expected_backend)
        self.assertEqual(all_gather_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(all_gather_param.comm_mode, expected_comm_mode)
        self.assertEqual(all_gather_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(all_gather_param.comm_domain, expected_comm_domain)
    
    def test_as_strided(self):
        as_strided_param = torch_atb.AsStridedParam()

        expected_size = []
        expected_stride = []
        expected_offset = []

        self.assertEqual(as_strided_param.size, expected_size)
        self.assertEqual(as_strided_param.stride, expected_stride)
        self.assertEqual(as_strided_param.offset, expected_offset)

    def test_cumsum(self):
        cumsum_param = torch_atb.CumsumParam()

        expected_axes = []
        expected_exclusive = False
        expected_reverse = False

        self.assertEqual(cumsum_param.axes, expected_axes)
        self.assertEqual(cumsum_param.exclusive, expected_exclusive)
        self.assertEqual(cumsum_param.reverse, expected_reverse)

    def test_dynamic_NTK(self):
        dynamic_NTK_param = torch_atb.DynamicNTKParam()

        expected_out_data_type = -1

        self.assertEqual(dynamic_NTK_param.out_data_type, expected_out_data_type)

    def test_multinomial(self):
        multinomial_param = torch_atb.MultinomialParam()

        expected_num_samples = 1
        expected_rand_seed = 0

        self.assertEqual(multinomial_param.num_samples, expected_num_samples)
        self.assertEqual(multinomial_param.rand_seed, expected_rand_seed)
    
    def test_concat(self):
        concat_param = torch_atb.ConcatParam()

        expected_concat_dim = 0

        self.assertEqual(concat_param.concat_dim, expected_concat_dim)

    def test_slice(self):
        slice_param = torch_atb.SliceParam()

        expected_offsets = []
        expected_size = []

        self.assertEqual(slice_param.offsets, expected_offsets)
        self.assertEqual(slice_param.size, expected_size)

    def test_transpose(self):
        transpose_param = torch_atb.TransposeParam()

        expected_perm = []

        self.assertEqual(transpose_param.perm, expected_perm)
    
    def test_gating(self):
        gating_param = torch_atb.GatingParam()

        expected_topk_expert_num = 1
        expected_cum_sum_num = 0
        expected_cum_sum_int64 = False
        expected_device_expert = []

        self.assertEqual(gating_param.topk_expert_num, expected_topk_expert_num)
        self.assertEqual(gating_param.cum_sum_num, expected_cum_sum_num)
        self.assertEqual(gating_param.cum_sum_int64, expected_cum_sum_int64)
        self.assertEqual(gating_param.device_expert, expected_device_expert)

    def test_reshape_and_cache(self):
        reshape_and_cache_param = torch_atb.ReshapeAndCacheParam()

        expected_compress_type = 0
        expected_kv_cache_cfg = 0

        self.assertEqual(reshape_and_cache_param.compress_type, expected_compress_type)
        self.assertEqual(reshape_and_cache_param.kv_cache_cfg, expected_kv_cache_cfg)

    def test_fill(self):
        fill_param = torch_atb.FillParam()

        expected_with_mask = True
        expected_value = []
        expected_out_dim = []

        self.assertEqual(fill_param.with_mask, expected_with_mask)
        self.assertEqual(fill_param.value, expected_value)
        self.assertEqual(fill_param.out_dim, expected_out_dim)

    def test_razor_fusion_attention(self):
        razor_fusion_attention_param = torch_atb.RazorFusionAttentionParam()

        expected_head_num = 1
        expected_kv_head_num = 1
        expected_qk_scale = 1
        expected_razor_len = 0
        expected_pre_tokens = 0
        expected_next_tokens = 0
        expected_tile_q = 0
        expected_tile_kv = 0
        expected_text_q_len = 0
        expected_text_kv_len = 0

        self.assertEqual(razor_fusion_attention_param.head_num, expected_head_num)
        self.assertEqual(razor_fusion_attention_param.kv_head_num, expected_kv_head_num)
        self.assertEqual(razor_fusion_attention_param.qk_scale, expected_qk_scale)
        self.assertEqual(razor_fusion_attention_param.razor_len, expected_razor_len)
        self.assertEqual(razor_fusion_attention_param.pre_tokens, expected_pre_tokens)
        self.assertEqual(razor_fusion_attention_param.next_tokens, expected_next_tokens)
        self.assertEqual(razor_fusion_attention_param.tile_q, expected_tile_q)
        self.assertEqual(razor_fusion_attention_param.tile_kv, expected_tile_kv)
        self.assertEqual(razor_fusion_attention_param.text_q_len, expected_text_q_len)
        self.assertEqual(razor_fusion_attention_param.text_kv_len, expected_text_kv_len)
    
    def test_all_reduce(self):
        all_reduce_param = torch_atb.AllReduceParam()

        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_all_reduce_type = "sum"
        expected_backend = "hccl"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_rank_table_file = ""
        expected_comm_domain = ""
        expected_quant_type = 0
        expected_out_data_type = -1

        self.assertEqual(all_reduce_param.rank, expected_rank)
        self.assertEqual(all_reduce_param.rank_size, expected_rank_size)
        self.assertEqual(all_reduce_param.rank_root, expected_rank_root)
        self.assertEqual(all_reduce_param.all_reduce_type, expected_all_reduce_type)
        self.assertEqual(all_reduce_param.backend, expected_backend)
        self.assertEqual(all_reduce_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(all_reduce_param.comm_mode, expected_comm_mode)
        self.assertEqual(all_reduce_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(all_reduce_param.comm_domain, expected_comm_domain)
        self.assertEqual(all_reduce_param.quant_type, expected_quant_type)
        self.assertEqual(all_reduce_param.out_data_type, expected_out_data_type)

    def test_broadcast(self):
        broadcast_param = torch_atb.BroadcastParam()

        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_backend = "hccl"
        expected_rank_table_file = ""
        expected_comm_domain = ""

        self.assertEqual(broadcast_param.rank, expected_rank)
        self.assertEqual(broadcast_param.rank_size, expected_rank_size)
        self.assertEqual(broadcast_param.rank_root, expected_rank_root)
        self.assertEqual(broadcast_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(broadcast_param.comm_mode, expected_comm_mode)
        self.assertEqual(broadcast_param.backend, expected_backend)
        self.assertEqual(broadcast_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(broadcast_param.comm_domain, expected_comm_domain)

    def test_reduce_scatter(self):
        reduce_scatter_param = torch_atb.ReduceScatterParam()

        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_reduce_type = "sum"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_backend = "lccl"
        expected_rank_table_file = ""
        expected_comm_domain = ""

        self.assertEqual(reduce_scatter_param.rank, expected_rank)
        self.assertEqual(reduce_scatter_param.rank_size, expected_rank_size)
        self.assertEqual(reduce_scatter_param.rank_root, expected_rank_root)
        self.assertEqual(reduce_scatter_param.reduce_type, expected_reduce_type)
        self.assertEqual(reduce_scatter_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(reduce_scatter_param.comm_mode, expected_comm_mode)
        self.assertEqual(reduce_scatter_param.backend, expected_backend)
        self.assertEqual(reduce_scatter_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(reduce_scatter_param.comm_domain, expected_comm_domain)

    def test_reduce_scatter_v(self):
        reduce_scatter_v_param = torch_atb.ReduceScatterVParam()

        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_send_counts = []
        expected_sdispls = []
        expected_recv_count = 0
        expected_reduce_type = "sum"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_backend = "hccl"
        expected_rank_table_file = ""
        expected_comm_domain = ""

        self.assertEqual(reduce_scatter_v_param.rank, expected_rank)
        self.assertEqual(reduce_scatter_v_param.rank_size, expected_rank_size)
        self.assertEqual(reduce_scatter_v_param.rank_root, expected_rank_root)
        self.assertEqual(reduce_scatter_v_param.send_counts, expected_send_counts)
        self.assertEqual(reduce_scatter_v_param.sdispls, expected_sdispls)
        self.assertEqual(reduce_scatter_v_param.recv_count, expected_recv_count)
        self.assertEqual(reduce_scatter_v_param.reduce_type, expected_reduce_type)
        self.assertEqual(reduce_scatter_v_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(reduce_scatter_v_param.comm_mode, expected_comm_mode)
        self.assertEqual(reduce_scatter_v_param.backend, expected_backend)
        self.assertEqual(reduce_scatter_v_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(reduce_scatter_v_param.comm_domain, expected_comm_domain)

    def test_linear_parallel(self):
        linear_parallel_param = torch_atb.LinearParallelParam()

        expected_trans_weight = True
        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_has_residual = False
        expected_backend = "hccl"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_rank_table_file = ""
        expected_type = 0
        expected_keep_intermediate = False
        expected_quant_type = -1
        expected_quant_group_size = 0
        expected_out_data_type = -1
        expected_comm_domain = ""
        expected_two_dim_TP_info_ag_dim = 0
        expected_two_dim_TP_info_rs_dim = 0
        expected_two_dim_TP_info_inner_dim_is_ag = 1

        self.assertEqual(linear_parallel_param.trans_weight, expected_trans_weight)
        self.assertEqual(linear_parallel_param.rank, expected_rank)
        self.assertEqual(linear_parallel_param.rank_size, expected_rank_size)
        self.assertEqual(linear_parallel_param.rank_root, expected_rank_root)
        self.assertEqual(linear_parallel_param.has_residual, expected_has_residual)
        self.assertEqual(linear_parallel_param.backend, expected_backend)
        self.assertEqual(linear_parallel_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(linear_parallel_param.comm_mode, expected_comm_mode)
        self.assertEqual(linear_parallel_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(linear_parallel_param.type, expected_type)
        self.assertEqual(linear_parallel_param.keep_intermediate, expected_keep_intermediate)
        self.assertEqual(linear_parallel_param.quant_type, expected_quant_type)
        self.assertEqual(linear_parallel_param.quant_group_size, expected_quant_group_size)
        self.assertEqual(linear_parallel_param.out_data_type, expected_out_data_type)
        self.assertEqual(linear_parallel_param.comm_domain, expected_comm_domain)
        self.assertEqual(linear_parallel_param.two_dim_TP_info.ag_dim, expected_two_dim_TP_info_ag_dim)
        self.assertEqual(linear_parallel_param.two_dim_TP_info.rs_dim, expected_two_dim_TP_info_rs_dim)
        self.assertEqual(linear_parallel_param.two_dim_TP_info.inner_dim_is_ag, expected_two_dim_TP_info_inner_dim_is_ag)

    def test_linear_sparse(self):
        linear_sparse_param = torch_atb.LinearSparseParam()

        expected_transpose_a = False
        expected_transpose_b = True
        expected_tiling_k = 8
        expected_tiling_n = 8

        self.assertEqual(linear_sparse_param.transpose_a, expected_transpose_a)
        self.assertEqual(linear_sparse_param.transpose_b, expected_transpose_b)
        self.assertEqual(linear_sparse_param.tiling_k, expected_tiling_k)
        self.assertEqual(linear_sparse_param.tiling_n, expected_tiling_n)

    def test_relay_attention(self):
        relay_attention_param = torch_atb.RelayAttentionParam()

        expected_head_num = 0
        expected_qk_scale = 1.0
        expected_kv_head_num = 0
        expected_mask_type = 0

        self.assertEqual(relay_attention_param.head_num, expected_head_num)
        self.assertEqual(relay_attention_param.qk_scale, expected_qk_scale)
        self.assertEqual(relay_attention_param.kv_head_num, expected_kv_head_num)
        self.assertEqual(relay_attention_param.mask_type, expected_mask_type)

    def test_topk_topp_sampling(self):
        test_topk_topp_sampling_param = torch_atb.TopkToppSamplingParam()
 
        expected_topk_topp_sampling_type = 0
        expected_rand_seeds = []
        expected_rand_seed = 0
        expected_topk = 100
        expected_log_probs_size = 0
 
        self.assertEqual(test_topk_topp_sampling_param.topk_topp_sampling_type, expected_topk_topp_sampling_type)
        self.assertEqual(test_topk_topp_sampling_param.rand_seeds, expected_rand_seeds)
        self.assertEqual(test_topk_topp_sampling_param.rand_seed, expected_rand_seed)
        self.assertEqual(test_topk_topp_sampling_param.topk, expected_topk)
        self.assertEqual(test_topk_topp_sampling_param.log_probs_size, expected_log_probs_size)
        
    def test_all_to_all(self):
        all_to_all_param = torch_atb.AllToAllParam()
        
        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_backend = "hccl"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_rank_table_file = ""
        expected_comm_domain = ""
        expected_transpose = False
        
        self.assertEqual(all_to_all_param.rank, expected_rank)
        self.assertEqual(all_to_all_param.rank_size, expected_rank_size)
        self.assertEqual(all_to_all_param.rank_root, expected_rank_root)
        self.assertEqual(all_to_all_param.backend, expected_backend)
        self.assertEqual(all_to_all_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(all_to_all_param.comm_mode, expected_comm_mode)
        self.assertEqual(all_to_all_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(all_to_all_param.comm_domain, expected_comm_domain)
        self.assertEqual(all_to_all_param.transpose, expected_transpose)

if __name__ == "__main__":
    print("----------- new_op_param test begin ------------")
    unittest.main()
    print("----------- new_op_param test begin ------------")
