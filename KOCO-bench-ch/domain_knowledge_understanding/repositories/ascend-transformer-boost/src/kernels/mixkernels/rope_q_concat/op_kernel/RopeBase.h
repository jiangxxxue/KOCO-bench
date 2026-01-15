/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ROPR_Q_CONCAT_BASE
#define ROPR_Q_CONCAT_BASE

#include "mixkernels/rope_q_concat/tiling/tiling_data.h"
#include "kernels/utils/kernel/simd.h"
#include "mixkernels/rope/op_kernel/common_val.h"
#include "kernel_operator.h"
using AscendC::Duplicate;

template <typename QkDtype, typename CosDtype, bool IF_COS_BROADCAST>
class RopeBase {
public:
    // QkDtype ：输入qk和输出qk的数据类型
    // CosDtype ：输入cos/sin的数据类型
    // IF_COS_BROADCAST ：cos sin是否已扩展
    // 构造函数
    __aicore__ inline RopeBase(const AtbOps::RopeQConcatTilingData* tilingData, AscendC::TPipe* pipe)
        : pipe_(pipe), blockIdx_(AscendC::GetBlockIdx())
    {
        this->tilingData_ = tilingData;
        loopTime = (blockIdx_ == tilingData_->realCore - 1) ?
            tilingData_->lastCoreLoopTime : tilingData_->preCoreLoopTime;
        lastLoopN = (blockIdx_ == tilingData_->realCore - 1) ?
            tilingData_->lastCoreLoopNLast : tilingData_->preCoreLoopNLast;
    }

    // 初始化Gm
    __aicore__ inline void RopeInitGm(__gm__ uint8_t *q, __gm__ uint8_t *cos, __gm__ uint8_t *sin,
     __gm__ uint8_t *concatInput, __gm__ uint8_t *outRopeQConcat)
    {
        qGm_.SetGlobalBuffer((__gm__ QkDtype *)q);
        cosGm_.SetGlobalBuffer((__gm__ CosDtype *)cos);
        sinGm_.SetGlobalBuffer((__gm__ CosDtype *)sin);
        concatInputGm_.SetGlobalBuffer((__gm__ QkDtype *)concatInput);
        outRopeConcatGm_.SetGlobalBuffer((__gm__ QkDtype *)outRopeQConcat);
    }

public:
    const AtbOps::RopeQConcatTilingData *tilingData_ = nullptr;
    AscendC::GlobalTensor<QkDtype> qGm_;
    AscendC::GlobalTensor<CosDtype> cosGm_;
    AscendC::GlobalTensor<CosDtype> sinGm_;
    AscendC::GlobalTensor<QkDtype> concatInputGm_;
    AscendC::GlobalTensor<QkDtype> outRopeConcatGm_;
    AscendC::TPipe *pipe_;

    uint32_t repeatSize_{0};         // 一拍做几个元素
    uint32_t rotateStride_{0};       // headdim / 旋转系数

    uint32_t blockIdx_;
    uint64_t loopTime{0};   // 当前核批处理数据轮数
    uint32_t lastLoopN{0};  // 当前核尾处理行数
};

#endif
