/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"

using namespace atb;

TEST(TestOpParamSize, TestOpParamCompatibility)
{
    EXPECT_EQ(sizeof(infer::ActivationParam), 24);
    EXPECT_EQ(sizeof(infer::AsStridedParam), 1640);
    EXPECT_EQ(sizeof(infer::BlockCopyParam), 16);
    EXPECT_EQ(sizeof(infer::CohereLayerNormParam), 36);
    EXPECT_EQ(sizeof(infer::ConcatParam), 16);
    EXPECT_EQ(sizeof(infer::CumsumParam), 560);
    EXPECT_EQ(sizeof(infer::DynamicNTKParam), 16);
    EXPECT_EQ(sizeof(infer::ElewiseParam), 64);
    EXPECT_EQ(sizeof(infer::ElewiseParam::QuantParam), 32);
    EXPECT_EQ(sizeof(infer::ElewiseParam::MulsParam), 16);
    EXPECT_EQ(sizeof(infer::FillParam), 848);
    EXPECT_EQ(sizeof(infer::FusedAddTopkDivParam), 56);
    EXPECT_EQ(sizeof(infer::GatherParam), 32);
    EXPECT_EQ(sizeof(infer::GatingParam), 56);
    EXPECT_EQ(sizeof(infer::GroupTopkParam), 24);
    EXPECT_EQ(sizeof(infer::GroupedMatmulInplaceAddParam), 24);
    EXPECT_EQ(sizeof(infer::GroupedMatmulWithRoutingParam), 32);
    EXPECT_EQ(sizeof(infer::IndexAddParam), 32);
    EXPECT_EQ(sizeof(infer::KvCacheParam), 8);
    EXPECT_EQ(sizeof(infer::LayerNormParam), 136);
    EXPECT_EQ(sizeof(infer::LayerNormParam::NormParam), 40);
    EXPECT_EQ(sizeof(infer::LayerNormParam::PreNormParam), 40);
    EXPECT_EQ(sizeof(infer::LayerNormParam::PostNormParam), 40);
    EXPECT_EQ(sizeof(infer::LinearParam), 32);
    EXPECT_EQ(sizeof(infer::LinearSparseParam), 24);
    EXPECT_EQ(sizeof(infer::MultinomialParam), 16);
    EXPECT_EQ(sizeof(infer::NonzeroParam), 8);
    EXPECT_EQ(sizeof(infer::OnehotParam), 24);
    EXPECT_EQ(sizeof(infer::PadParam), 8);
    EXPECT_EQ(sizeof(infer::PagedAttentionParam), 120);
    EXPECT_EQ(sizeof(infer::ReduceParam), 560);
    EXPECT_EQ(sizeof(infer::RelayAttentionParam), 48);
    EXPECT_EQ(sizeof(infer::RepeatParam), 552);
    EXPECT_EQ(sizeof(infer::ReshapeAndCacheParam), 24);
    EXPECT_EQ(sizeof(infer::RmsNormParam), 144);
    EXPECT_EQ(sizeof(infer::RmsNormParam::NormParam), 64);
    EXPECT_EQ(sizeof(infer::RmsNormParam::PreNormParam), 32);
    EXPECT_EQ(sizeof(infer::RmsNormParam::PostNormParam), 32);
    EXPECT_EQ(sizeof(infer::RopeParam), 16);
    EXPECT_EQ(sizeof(infer::SelfAttentionParam), 144);
    EXPECT_EQ(sizeof(infer::SetValueParam), 1640);
    EXPECT_EQ(sizeof(infer::SliceParam), 1096);
    EXPECT_EQ(sizeof(infer::SoftmaxParam), 552);
    EXPECT_EQ(sizeof(infer::SortParam), 296);
    EXPECT_EQ(sizeof(infer::SplitParam), 304);
    EXPECT_EQ(sizeof(infer::TopkToppSamplingParam), 56);
    EXPECT_EQ(sizeof(infer::TransdataParam), 560);
    EXPECT_EQ(sizeof(infer::TransposeParam), 296);
    EXPECT_EQ(sizeof(infer::UnpadParam), 8);
    EXPECT_EQ(sizeof(infer::WhereParam), 8);

    EXPECT_EQ(sizeof(train::GenAttentionMaskParam), 304);
    EXPECT_EQ(sizeof(train::RopeGradParam), 32);
    EXPECT_EQ(sizeof(train::FastSoftMaxParam), 40);
    EXPECT_EQ(sizeof(train::FastSoftMaxGradParam), 40);
    EXPECT_EQ(sizeof(train::StridedBatchMatmulParam), 240);
    EXPECT_EQ(sizeof(train::UnpadWithHiddenStateParam), 40);
    EXPECT_EQ(sizeof(train::PadWithHiddenStateParam), 40);
    EXPECT_EQ(sizeof(train::RmsNormBackwardParam), 8);

#ifdef _GLIBCXX_USE_CXX11_ABI
#if _GLIBCXX_USE_CXX11_ABI == 0
    EXPECT_EQ(sizeof(infer::AllGatherParam), 120);
    EXPECT_EQ(sizeof(infer::AllReduceParam), 136);
    EXPECT_EQ(sizeof(infer::AllToAllParam), 120);
    EXPECT_EQ(sizeof(infer::AllToAllVParam), 216);
    EXPECT_EQ(sizeof(infer::BroadcastParam), 120);
    EXPECT_EQ(sizeof(infer::LinearParallelParam), 152);
    EXPECT_EQ(sizeof(infer::RecvParam), 120);
    EXPECT_EQ(sizeof(infer::ReduceScatterParam), 128);
    EXPECT_EQ(sizeof(infer::SendParam), 120);
    EXPECT_EQ(sizeof(train::LaserAttentionParam), 48);
    EXPECT_EQ(sizeof(train::LaserAttentionGradParam), 48);
#else
    EXPECT_EQ(sizeof(infer::AllGatherParam), 192);
    EXPECT_EQ(sizeof(infer::AllReduceParam), 232);
    EXPECT_EQ(sizeof(infer::AllToAllParam), 192);
    EXPECT_EQ(sizeof(infer::AllToAllVParam), 288);
    EXPECT_EQ(sizeof(infer::BroadcastParam), 192);
    EXPECT_EQ(sizeof(infer::LinearParallelParam), 224);
    EXPECT_EQ(sizeof(infer::RecvParam), 192);
    EXPECT_EQ(sizeof(infer::ReduceScatterParam), 224);
    EXPECT_EQ(sizeof(infer::SendParam), 192);
    EXPECT_EQ(sizeof(train::LaserAttentionParam), 72);
    EXPECT_EQ(sizeof(train::LaserAttentionGradParam), 72);
#endif
#endif
}