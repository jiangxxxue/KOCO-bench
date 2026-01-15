/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hosttensor_binder_creator.h"
#include <atb/utils/log.h>
#include "hosttensor_binders/selfattention_binder.h"
#include "hosttensor_binders/pagedattention_binder.h"
#include "hosttensor_binders/llama65blayer_binder.h"
#include "hosttensor_binders/reduce_scatterv_binder.h"
#include "hosttensor_binders/relayattention_binder.h"
#include "hosttensor_binders/all_gatherv_binder.h"
#include "hosttensor_binders/all_to_allvv2_binder.h"
#include "hosttensor_binders/multi_latent_attention_binder.h"
#include "hosttensor_binders/reshape_and_cache_with_stride_binder.h"
#include "hosttensor_binders/layernorm_with_stride_binder.h"
#include "hosttensor_binders/rmsnorm_with_stride_binder.h"
#include "hosttensor_binders/ring_mla_binder.h"


HostTensorBinder *CreateHostTensorBinder(const std::string &opName)
{
    if (opName == "SelfAttentionOperation") {
        return new SelfAttentionBinder();
    } else if (opName == "PagedAttentionOperation") {
        return new PagedAttentionBinder();
    } else if (opName == "Llama65BLayerOperation") {
        return new Llama65BLayerBinder();
    } else if (opName == "ReduceScatterVOperation") {
        return new ReduceScatterVBinder();
    } else if (opName == "RelayAttentionOperation") {
        return new RelayAttentionBinder();
    } else if (opName == "AllGatherVOperation") {
        return new AllGatherVBinder();
    } else if (opName == "AllToAllVV2Operation") {
        return new AllToAllVV2Binder();
    } else if (opName == "MultiLatentAttentionOperation") {
        return new MultiLatentAttentionBinder();
    } else if (opName == "ReshapeAndCacheWithStrideOperation") {
        return new ReshapeAndCacheWithStrideBinder();
    } else if (opName == "LayerNormWithStrideOperation") {
        return new LayerNormWithStrideBinder();
    } else if (opName == "RmsNormWithStrideOperation") {
        return new RmsNormWithStrideBinder();
    } else if (opName == "RingMLAOperation") {
        return new RingMLABinder();
    } else {
        ATB_LOG(DEBUG) << "opName:" << opName << " not host binder";
    }
    return nullptr;
}