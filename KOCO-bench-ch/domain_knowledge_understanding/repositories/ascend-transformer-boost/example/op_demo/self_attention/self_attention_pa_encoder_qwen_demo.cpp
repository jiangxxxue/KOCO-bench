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
#include <random>
#include <limits>

const uint32_t BATCH_SIZE = 1; // 批处理大小
std::vector<int32_t> seqLenHost = {1024}; // host侧tensor值，用于存储每个批处理中的序列长度，长度等于BATCH_SIZE
const uint32_t NTOKENS = accumulate(seqLenHost.begin(), seqLenHost.end(), 0); // sum(seqLenHost)
const uint32_t MAX_SEQ_LEN = 1024;                                            // 最大序列长度
const uint32_t HEAD_NUM = 5;                                                  // 头数
const uint32_t KV_HEAD_NUM = 1;                                               // kv头数
const uint32_t HEAD_SIZE = 128;                                               // 头大小
const uint32_t MASK_SEQ_LEN = 128;                                            // mask的block大小

/**
 * @brief 随机填充inData数值
 * @param inData 需要随机填充的vector
 * @param low 最小值
 * @param high 最大值
 */
void AssignRandomValue(std::vector<float> &inData, int low = -5, int high = 5)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(low, high); // 生成[-5,5)正态分布数据
    for (size_t i = 0; i < inData.size(); ++i) {
        inData[i] = dis(gen);
    }
}

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param seqLenHost host侧tensor。序列长度向量，等于1时，为增量或全量；大于1时，为全量
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status atb错误码
 * @note 需要传入所有host侧tensor
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, std::vector<int32_t> &seqLenHost,
                            atb::SVector<atb::Tensor> &inTensors)
{
    std::vector<float> qData(NTOKENS * HEAD_NUM * HEAD_SIZE);
    AssignRandomValue(qData);
    // 创建query tensor
    // shape: [nTokens, headNum, headSize], DeepSeek场景headSize多为128
    atb::Tensor tensorQ;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, qData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, HEAD_NUM, HEAD_SIZE}, tensorQ));
    // 创建key，value tensor
    // shape: [nTokens, kvHeadNum, headSize], DeepSeek场景kvHeadNum与q不同，但headSize和q相同
    std::vector<float> kData(NTOKENS * KV_HEAD_NUM * HEAD_SIZE);
    AssignRandomValue(kData);
    atb::Tensor tensorK;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, KV_HEAD_NUM, HEAD_SIZE}, tensorK));
    std::vector<float> vData(NTOKENS * KV_HEAD_NUM * HEAD_SIZE);
    AssignRandomValue(vData);
    atb::Tensor tensorV;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, vData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, KV_HEAD_NUM, HEAD_SIZE}, tensorV));
    std::vector<float> maskData(MASK_SEQ_LEN * MASK_SEQ_LEN, 0);
    // 创建倒三角压缩mask，值为-inf，shape为[128,128]的上三角mask
    float negInf = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < MASK_SEQ_LEN; ++j) {
        // 对角线置0
        for (int k = j + 1; k < MASK_SEQ_LEN; ++k) {
            maskData[j * MASK_SEQ_LEN + k] = negInf;
        }
    }
    atb::Tensor tensorMask;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, maskData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {MASK_SEQ_LEN, MASK_SEQ_LEN}, tensorMask));
    // 创建seqLen，host侧tensor，表示每个batch对应的序列长度，总和nTokens与q的第一维度对应
    atb::Tensor tensorSeqLen;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, tensorSeqLen));
    // 不需要拷贝值到device测，直接传入cpu的vector的值即可
    tensorSeqLen.hostData = seqLenHost.data(); // seqLenHost中的值为seqLen
    // 根据顺序将所有输入tensor放入SVector
    inTensors = {tensorQ, tensorK, tensorV, tensorMask, tensorSeqLen};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个FA encoder的Operation，并设置参数
 * @param 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status PrepareOperation(atb::Operation **paEncoderOp)
{
    atb::infer::SelfAttentionParam faPaOpParam;
    faPaOpParam.headNum = HEAD_NUM;             // query 头数
    faPaOpParam.kvHeadNum = KV_HEAD_NUM;        // key, value 头数
    faPaOpParam.qkScale = 1 / sqrt(HEAD_SIZE);  // tor值，Q*K^T后的缩放系数，根据HEAD_SIZE做归一化
    faPaOpParam.isTriuMask = 1; // 是否开启mask倒三角优化，这里开启，和压缩mask一起使用
    faPaOpParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER; // 计算类型/场景分类，使用PA Encoder
    // 高性能，softmax使用float16
    faPaOpParam.kernelType = atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_DEFAULT;
    faPaOpParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM_COMPRESS; // 使用128x128的上三角
    return atb::CreateOperation(faPaOpParam, paEncoderOp);
}

int main(int argc, char **argv)
{
    CHECK_STATUS(aclInit(nullptr));
    // 设置卡号、创建context、设置stream
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // FA PAEncoder qwen示例
    atb::Operation *paEncoderOp = nullptr;
    CHECK_STATUS(PrepareOperation(&paEncoderOp));
    // 准备输入张量
    atb::VariantPack paVariantPack;
    paVariantPack.inTensors;
    CHECK_STATUS(PrepareInTensor(context, stream, seqLenHost, paVariantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_BF16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE}, tensorOut);
    paVariantPack.outTensors.push_back(tensorOut); // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(paEncoderOp->Setup(paVariantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // FA Encoder执行
    CHECK_STATUS(paEncoderOp->Execute(paVariantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    for (atb::Tensor &inTensor : paVariantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(paEncoderOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "FA PA Encoder Qwen demo success!" << std::endl;
    return 0;
}
