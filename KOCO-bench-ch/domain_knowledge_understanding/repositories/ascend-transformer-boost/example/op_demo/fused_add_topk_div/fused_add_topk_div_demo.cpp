/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../demo_util.h"

using namespace atb;
using namespace std;
const uint32_t K = 8;
const uint32_t BATCH_SIZE = 512;
const uint32_t EXPERT_NUM = 256;
const float INIT_VALUE = 2.0f; // 向量的初始值

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @note 需要传入所有host侧tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建tensor0
    std::vector<float> tensormul0(BATCH_SIZE * EXPERT_NUM, INIT_VALUE);
    atb::Tensor tensorMul0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, tensormul0, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {BATCH_SIZE, EXPERT_NUM}, tensorMul0));
    // 创建tensor1
    std::vector<float> tensormul1(EXPERT_NUM, INIT_VALUE);
    atb::Tensor tensorMul1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, tensormul1, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {EXPERT_NUM}, tensorMul1));
    // 根据顺序将所有输入tensor放入SVector
    inTensors = {tensorMul0, tensorMul1};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个FusedAddTopkDivOperation，并设置参数
 * @param atb::Operation * 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status PrepareOperation(atb::Operation **op)
{
    atb::infer::FusedAddTopkDivParam fusedAddTopkDivParam;                                // FusedAddTopkDivParam
    fusedAddTopkDivParam.groupNum = 8;                                                    // 分组数量。
    fusedAddTopkDivParam.groupTopk = 4;                                                   // 选择k个组。
    fusedAddTopkDivParam.n = 2;                                                           // 组内选取n个最大值求和。
    fusedAddTopkDivParam.k = K;                                                           // topk选择前k个值。
    fusedAddTopkDivParam.activationType = atb::infer::ActivationType::ACTIVATION_SIGMOID; // 激活类型。
    fusedAddTopkDivParam.isNorm = true;                                                   // 是否归一化。
    fusedAddTopkDivParam.scale = 2.5;                                                     // 归一化后的乘系数。
    fusedAddTopkDivParam.enableExpertMapping = false;                                     // 是否使能物理专家向逻辑专家的映射。
    return atb::CreateOperation(fusedAddTopkDivParam, op);
}

int main(int argc, char **argv)
{
    // 1.设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // FusedAddTopkDiv示例
    atb::Operation *op = nullptr;
    CHECK_STATUS(PrepareOperation(&op));

    // 准备输入张量
    atb::VariantPack variantPack;
    PrepareInTensor(context, stream, variantPack.inTensors); // 放入输入tensor

    // 准备输出张量
    atb::Tensor tensorOut0;
    CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, K}, tensorOut0); // 创建输出tensor
    variantPack.outTensors.push_back(tensorOut0);                                   // 放入输出tensor
    atb::Tensor tensorOut1;
    CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, K}, tensorOut1); // 创建输出tensor
    variantPack.outTensors.push_back(tensorOut1);                                   // 放入输出tensor

    // setup阶段，计算workspace大小
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // execute阶段
    op->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 释放内存
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "fused_add_topk_div demo success!" << std::endl;
    return 0;
}
