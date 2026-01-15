/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <future>
#include "../demo_util.h"

struct Args {
    int rankId;
    aclrtStream stream;
    atb::Context *context;
    atb::infer::AllReduceParam param;
};

/**
 * @brief 准备atb::VariantPack中的所有tensor
 * @param args 线程参数结构体
 * @return atb::VariantPack
 */
atb::VariantPack PrepareVariantPack(Args &args)
{
    atb::VariantPack variantPack;
    std::vector<int64_t> shape = {2, 1024};
    // 创建Host侧数据
    std::vector<float> xHostData(shape[0] * shape[1], 2.0);
    std::vector<float> outputHostData(shape[0] * shape[1], 0);
    // 生成ATB tensor
    atb::Tensor tensorX;
    CreateTensorFromVector(args.context, args.stream, xHostData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, shape, tensorX);
    atb::Tensor tensorOutput;
    CreateTensorFromVector(args.context, args.stream, outputHostData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, shape,
                           tensorOutput);
    // 根据顺序将所有输入tensor放入SVector
    variantPack.inTensors = {tensorX};
    variantPack.outTensors = {tensorOutput};
    return variantPack;
}

/**
 * @brief 创建一个AllReduce的Operation，并设置参数
 * @return 错误码
 */
int RunAllReduceOp(Args &args)
{
    int rankId = args.rankId;
    auto aclRet = aclrtSetDevice(rankId);
    CHECK_STATUS_EXPR(aclRet, std::cout << "aclrtSetDevice failed" << std::endl; return aclRet);
    atb::VariantPack variantPack = PrepareVariantPack(args);
    atb::Operation *allReduceOp = nullptr;
    atb::Status st = atb::CreateOperation(args.param, &allReduceOp);
    CHECK_STATUS_EXPR((st != atb::ErrorType::NO_ERROR || !allReduceOp),
                      std::cout << "atb create operation failed" << std::endl;
                      return st);
    uint64_t workspaceSize = 0;
    // ATB Operation 第一阶段接口调用：对输入输出进行检查，并根据需要计算workspace大小
    st = allReduceOp->Setup(variantPack, workspaceSize, args.context);
    CHECK_STATUS_EXPR(st, std::cout << "operation setup failed" << std::endl; return st);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        aclRet = aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_STATUS_EXPR(aclRet, std::cout << "aclrtMalloc failed" << std::endl; return aclRet);
    }
    // ATB Operation 第二阶段接口调用：执行算子
    st = allReduceOp->Execute(variantPack, workspacePtr, workspaceSize, args.context);
    CHECK_STATUS_EXPR(st, std::cout << "operation execute failed" << std::endl; return aclRet);
    // 同步等待
    aclRet = aclrtSynchronizeStream(args.stream);
    CHECK_STATUS_EXPR(aclRet, std::cout << "aclrtSynchronizeStreame failed" << std::endl; return aclRet);
    // 资源释放
    for (atb::Tensor &tensor : variantPack.inTensors) {
        if (tensor.deviceData) {
            aclrtFree(tensor.deviceData);
        }
    }
    for (atb::Tensor &tensor : variantPack.outTensors) {
        if (tensor.deviceData) {
            aclrtFree(tensor.deviceData);
        }
    }
    if (workspaceSize > 0 && workspacePtr != nullptr) {
        aclrtFree(workspacePtr);
    }
    st = atb::DestroyOperation(allReduceOp);
    CHECK_STATUS_EXPR(st, std::cout << "destroy operation failed" << std::endl; return st);
    aclRet = aclrtDestroyStream(args.stream);
    CHECK_STATUS_EXPR(aclRet, std::cout << "aclrtDestroyStream failed" << std::endl; return aclRet);
    st = atb::DestroyContext(args.context);
    CHECK_STATUS_EXPR(st, std::cout << "destroy context failed" << std::endl; return st);
    aclRet = aclrtResetDevice(args.rankId);
    CHECK_STATUS_EXPR(aclRet, std::cout << "aclrtResetDevice failed" << std::endl; return aclRet);
    return 0;
}

int main(int argc, char **argv)
{
    // 设置通信设备数量
    static const int DEV_NUM = 8;
    // 设置通信计算类型
    static const std::string ALL_REDUCE_TYPE = "sum";
    // 设置通信计算后端
    static const std::string BACKEND = "lccl";
    CHECK_STATUS(aclInit(nullptr));
    Args args[DEV_NUM];
    for (int rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        // 设置当前设备
        CHECK_STATUS(aclrtSetDevice(rankId));
        // 创建ATB Context
        CHECK_STATUS(atb::CreateContext(&(args[rankId].context)));
        // 创建流
        CHECK_STATUS(aclrtCreateStream(&(args[rankId].stream)));
        args[rankId].context->SetExecuteStream(args[rankId].stream);
        // 配置AllReduce Param
        atb::infer::AllReduceParam &param = args[rankId].param;
        param.rank = rankId;
        param.rankSize = DEV_NUM;
        param.allReduceType = ALL_REDUCE_TYPE;
        param.backend = BACKEND;
        param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
        param.commDomain = "domain0"; // 单通信域demo
    }
    // 启动多线程任务
    std::vector<std::future<int>> threads(DEV_NUM);
    for (int rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId] = std::async(&RunAllReduceOp, std::ref(args[rankId]));
    }
    for (int rankId = 0; rankId < DEV_NUM; rankId++) {
        int aclRet = threads[rankId].get();
        CHECK_STATUS_EXPR(aclRet, std::cout << "multithread task " << rankId << " failed" << std::endl);
    }
    CHECK_STATUS(aclFinalize());
    std::cout << "AllReduce demo finished!" << std::endl;
    return 0;
}
