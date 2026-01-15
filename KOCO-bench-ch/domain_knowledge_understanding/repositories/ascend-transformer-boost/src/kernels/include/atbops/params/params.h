/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_PARAMS_H
#define ATBOPS_PARAMS_PARAMS_H
#include "atbops/params/common.h"
#include "atbops/params/fastsoftmax.h"
#include "atbops/params/fastsoftmax_grad.h"
#include "atbops/params/ffn.h"
#include "atbops/params/fused_add_topk_div.h"
#include "atbops/params/gating.h"
#include "atbops/params/genattentionmask.h"
#include "atbops/params/kvcache.h"
#include "atbops/params/laser_attention.h"
#include "atbops/params/laser_attention_grad.h"
#include "atbops/params/pad.h"
#include "atbops/params/pad_with_hidden_state.h"
#include "atbops/params/pagedattention.h"
#include "atbops/params/reshape_and_cache.h"
#include "atbops/params/paged_cache_load.h"
#include "atbops/params/rope_grad.h"
#include "atbops/params/rope.h"
#include "atbops/params/stridedbatchmatmul.h"
#include "atbops/params/toppsample.h"
#include "atbops/params/toppsample_rand.h"
#include "atbops/params/unpad_flash_attention_nz.h"
#include "atbops/params/unpad_flash_attention.h"
#include "atbops/params/unpad_with_hidden_state.h"
#include "atbops/params/unpad.h"
#include "atbops/params/blockcopy.h"
#include "atbops/params/moe_gmm.h"
#include "atbops/params/gmm_add.h"
#include "atbops/params/rope_q_concat.h"
#include "atbops/params/swiglu_quant.h"
#include "atbops/params/rms_norm_and_rope_and_reshape_and_cache.h"
#include "atbops/params/mla_preprocess.h"
#include "atbops/params/mla.h"
#include "atbops/params/fusion.h"
#include "atbops/params/ring_mla.h"
#include "atbops/params/gmm_deq_swiglu_quant_gmm_deq.h"
#include "atbops/params/mm_deq_swiglu_quant_mm_deq.h"
#endif
