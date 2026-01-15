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
#include <cmath>

const uint32_t BATCH_SIZE = 10;                  // 批处理大小
std::vector<int32_t> seqLenHost(BATCH_SIZE, 16); // host侧tensor值，用于存储每个批处理中的序列长度
std::vector<int32_t> tokenOffsetHost(BATCH_SIZE, 16);                         // host侧tensor值，token偏移
std::vector<int32_t> layerId(1, 0);                                           // device侧，kvCache中取哪个计算
const uint32_t NTOKENS = accumulate(seqLenHost.begin(), seqLenHost.end(), 0); // sum(seqLenHost)
const uint32_t MAX_SEQ_LEN = 1024;                                            // 最大序列长度
const uint32_t HEAD_NUM = 32;                                                 // 头数
const uint32_t KV_HEAD_NUM = 32;                                              // kv头数
const uint32_t HEAD_SIZE = 64;                                                // 头大小
const int32_t IN_Q_INDEX = 0;
const int32_t IN_K_INDEX = 1;
const int32_t IN_V_INDEX = 2;
const int32_t IN_CACHE_K_INDEX = 3;
const int32_t IN_CACHE_V_INDEX = 4;
const int32_t IN_CACHE_MASK_INDEX = 5;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param seqLenHost host侧tensor。序列长度向量，等于1时，为增量或全量；大于1时，为全量
 * @param tokenOffsetHost host侧tensor。计算完成后的token偏移
 * @param layerId layerId，取cache的kv中哪一个kv进行计算
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status atb错误码
 * @note 需要传入所有host侧tensor
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, std::vector<int32_t> &seqLenHost,
                            std::vector<int32_t> &tokenOffsetHost, std::vector<int32_t> &layerId,
                            atb::SVector<atb::Tensor> &inTensors)
{
    uint32_t qHiddenSize = HEAD_NUM * HEAD_SIZE;
    uint32_t kvHiddenSize = KV_HEAD_NUM * HEAD_SIZE;
    atb::Tensor tensorQ;
    atb::Tensor tensorK;
    atb::Tensor tensorV;
    atb::Tensor tensorCacheK;
    atb::Tensor tensorCacheV;
    atb::Tensor tensorMask;
    // 创建query tensor
    std::vector<float> qData(NTOKENS * qHiddenSize, 1.0);
    std::vector<float> kvData(NTOKENS * kvHiddenSize, 1.0);
    std::vector<float> kvCacheData(BATCH_SIZE * MAX_SEQ_LEN * kvHiddenSize, 1.0);
    // 创建norm mask，值为-inf的上三角mask
    std::vector<float> maskData(BATCH_SIZE * MAX_SEQ_LEN * MAX_SEQ_LEN, 0);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < MAX_SEQ_LEN; ++j) {
            for (int k = j + 1; k < MAX_SEQ_LEN; ++k) {
                maskData[i * MAX_SEQ_LEN * MAX_SEQ_LEN + j * MAX_SEQ_LEN + k] = -32768;
            }
        }
    }
    // 创建tokenOffset，host侧tensor
    atb::Tensor tensorTokenOffset;
    atb::Tensor tensorSeqLen;
    atb::Tensor tensorLayerId;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, tensorTokenOffset));
    tensorTokenOffset.hostData = tokenOffsetHost.data(); // host侧tensor，拷贝值
    // 创建seqLen，host侧tensor
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, tensorSeqLen));
    tensorSeqLen.hostData = seqLenHost.data(); // host侧tensor，拷贝值
    // 创建layerId
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {1}, tensorLayerId));
    CHECK_STATUS(aclrtMemcpy(tensorLayerId.deviceData, tensorLayerId.dataSize, layerId.data(),
                             sizeof(short) * layerId.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    // 根据顺序将所有输入tensor放入SVector
    inTensors = {tensorQ,    tensorK,           tensorV,      tensorCacheK, tensorCacheV,
                 tensorMask, tensorTokenOffset, tensorSeqLen, tensorLayerId};
    for (int32_t i = 0; i <= IN_CACHE_MASK_INDEX; ++i) {
        if (i == IN_Q_INDEX) {
            CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, qData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                {NTOKENS, qHiddenSize}, inTensors[i]));
        } else if (i == IN_K_INDEX || i == IN_V_INDEX) {
            CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kvData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                {NTOKENS, kvHiddenSize}, inTensors[i]));
        } else if (i == IN_CACHE_K_INDEX || i == IN_CACHE_V_INDEX) {
            CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kvCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                {1, BATCH_SIZE, MAX_SEQ_LEN, kvHiddenSize}, inTensors[i]));
        } else if (i == IN_CACHE_MASK_INDEX) {
            CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, maskData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                {BATCH_SIZE, MAX_SEQ_LEN, MAX_SEQ_LEN}, inTensors[i]));
        }
    }
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个FA encoder的Operation，并设置参数
 * @param 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status PrepareOperation(atb::Operation **encoderOp)
{
    atb::infer::SelfAttentionParam opParam;
    opParam.headNum = HEAD_NUM;            // query 头数
    opParam.kvHeadNum = KV_HEAD_NUM;       // key, value 头数
    opParam.qkScale = 1 / sqrt(HEAD_SIZE); // tor值，Q*K^T后的缩放系数，根据HEAD_SIZE做归一化
    opParam.calcType = atb::infer::SelfAttentionParam::CalcType::ENCODER; // 计算类型/场景分类，使用FA Encoder
    opParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM; // 普通全量mask
    return atb::CreateOperation(opParam, encoderOp);
}

int main(int argc, char **argv)
{
    // kv隐藏层大小，用于输出tensor shape
    uint32_t kvHiddenSize = KV_HEAD_NUM * HEAD_SIZE;
    // 设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);
    // FA Encoder示例
    atb::Operation *encoderOp = nullptr;
    CHECK_STATUS(PrepareOperation(&encoderOp));
    // 准备输入张量
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, seqLenHost, tokenOffsetHost, layerId,
                                 variantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, kvHiddenSize}, tensorOut));
    variantPack.outTensors.push_back(tensorOut); // 放入输出tensor
    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(encoderOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // FA Encoder执行
    encoderOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(encoderOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "FA Encoder demo for Atlas A2/A3 success!" << std::endl;
    return 0;
}
