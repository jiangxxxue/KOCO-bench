/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <sys/wait.h>

#include "../demo_util.h"

atb::Status ExcuteImpl(atb::Operation *op, atb::VariantPack variantPack, atb::Context *context, aclrtStream &stream)
{
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_STATUS(op->Execute(variantPack, (uint8_t *)workspace, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    if (workspace) {
        CHECK_STATUS(aclrtFree(workspace)); // 销毁workspace
    }
    return atb::ErrorType::NO_ERROR;
}

atb::Status LinearParallelSample(int rank, int rankSize)
{
    int ret = aclInit(nullptr);
    // 设置每个进程对应的deviceId
    int deviceId = rank;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    aclrtStream stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    atb::Tensor input;
    CreateTensorFromVector(context, stream, std::vector<float>(64, 2.0), aclDataType::ACL_FLOAT16,
                           aclFormat::ACL_FORMAT_ND, {2, 32}, input);

    atb::Tensor weight;
    CreateTensorFromVector(context, stream, std::vector<int8_t>(64, 2), aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND,
                           {32, 2}, weight);

    atb::Tensor bias;
    CreateTensorFromVector(context, stream, std::vector<float>(1, 1.0), aclDataType::ACL_FLOAT16,
                           aclFormat::ACL_FORMAT_ND, {1}, bias);

    atb::Tensor deqScale;
    CreateTensorFromVector(context, stream, std::vector<float>(1, 1.0), aclDataType::ACL_FLOAT16,
                           aclFormat::ACL_FORMAT_ND, {1}, deqScale);

    atb::Tensor output;
    CreateTensor(aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {2, 2}, output);

    atb::infer::LinearParallelParam param;
    param.transWeight = false;
    param.rank = rank;
    param.rankRoot = 0;
    param.rankSize = rankSize;
    param.backend = "lcoc";
    param.keepIntermediate = false;
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;
    param.quantType = atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TENSOR;
    param.outDataType = ACL_FLOAT16;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &op));

    atb::VariantPack variantPack;
    variantPack.inTensors = {input, weight, bias, deqScale};
    variantPack.outTensors = {output};
    CHECK_STATUS(ExcuteImpl(op, variantPack, context, stream));
    std::cout << "rank: " << rank << " executed END." << std::endl;

    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op));    // 销毁op对象
    CHECK_STATUS(aclrtDestroyStream(stream));   // 销毁stream
    CHECK_STATUS(atb::DestroyContext(context)); // 销毁context
    CHECK_STATUS(aclFinalize());
    std::cout << "demo excute success" << std::endl;
    return atb::ErrorType::NO_ERROR;
}

int main(int argc, const char *argv[])
{
    const int processCount = 2;
    for (int i = 0; i < processCount; i++) {
        pid_t pid = fork();
        // 子进程
        if (pid == 0) {
            CHECK_STATUS(LinearParallelSample(i, processCount));
            return 0;
        } else if (pid < 0) {
            std::cerr << "Failed to create process." << std::endl;
            return 1;
        }
    }
    // 父进程等待子进程执行完成
    for (int i = 0; i < processCount; ++i) {
        wait(nullptr);
    }
    std::cout << "The communication operator is successfully executed. Parent process exit" << std::endl;
    return 0;
}
