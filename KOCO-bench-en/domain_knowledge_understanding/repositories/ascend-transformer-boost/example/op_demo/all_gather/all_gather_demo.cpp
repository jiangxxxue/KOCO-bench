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

atb::Status ExcuteImpl(atb::Operation *op, atb::VariantPack variantPack, atb::Context *context)
{
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_STATUS(op->Execute(variantPack, (uint8_t *)workspace, workspaceSize, context));
    if (workspace) {
        CHECK_STATUS(aclrtFree(workspace));  // 销毁workspace
    }
    return atb::ErrorType::NO_ERROR;
}

atb::Status AllGatherSample(int rank, int rankSize)
{
    int ret = aclInit(nullptr);
    // 设置每个进程对应的deviceId
    int deviceId = rank;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    aclrtStream stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    CHECK_STATUS(context->SetExecuteStream(stream));
    atb::Tensor input;
    input.desc.dtype = ACL_FLOAT16;
    input.desc.format = ACL_FORMAT_ND;
    input.desc.shape.dimNum = 2;
    input.desc.shape.dims[0] = 3;
    input.desc.shape.dims[1] = 5;
    input.dataSize = atb::Utils::GetTensorSize(input);
    CHECK_STATUS(aclrtMalloc(&input.deviceData, input.dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
    atb::Tensor output;
    output.desc.dtype = ACL_FLOAT16;
    output.desc.format = ACL_FORMAT_ND;
    output.desc.shape.dimNum = 3;
    output.desc.shape.dims[0] = 2;
    output.desc.shape.dims[1] = 3;
    output.desc.shape.dims[2] = 5;
    output.dataSize = atb::Utils::GetTensorSize(output);
    CHECK_STATUS(aclrtMalloc(&output.deviceData, output.dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
    atb::infer::AllGatherParam param;
    param.rank = rank;
    param.rankRoot = 0;
    param.rankSize = rankSize;
    param.backend = "hccl";
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &op));
    atb::VariantPack variantPack;
    variantPack.inTensors = {input};
    variantPack.outTensors = {output};
    CHECK_STATUS(ExcuteImpl(op, variantPack, context));
    std::cout << "rank: " << rank << " executed END." << std::endl;
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op));     // 销毁op对象
    CHECK_STATUS(aclrtDestroyStream(stream));    // 销毁stream
    CHECK_STATUS(atb::DestroyContext(context));  // 销毁context
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
            CHECK_STATUS(AllGatherSample(i, processCount));
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
