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

const uint32_t BATCH_SIZE = 3; // 批处理大小

// host侧tensor值，shape可以为[batch]/[2, batch]
//      当shape为[batch]时：query和key, value的词元长度一样
//      当shape为[2, batch]时：query词元长度为前batch个，key，value词元长度为后batch个
//      这里演示query和key, value的词元长度不同时场景
//      因为作为host侧tensor，需要填入1维tensor，并设置shape为[2,batch]
//      对应：qSeqLen = {100, 1000, 128}, kvSeqLen = {700, 0, 128}
std::vector<int32_t> seqLenHost = {100, 1000, 128, 700, 0, 128};
const uint32_t Q_NTOKENS = accumulate(seqLenHost.begin(), seqLenHost.begin() + BATCH_SIZE, 0); // sum(seqLenHost)
const uint32_t kV_NTOKENS = accumulate(seqLenHost.begin() + BATCH_SIZE, seqLenHost.end(), 0);  // sum(seqLenHost)
const uint32_t HEAD_NUM = 16;                                                                  // 头数
const uint32_t KV_HEAD_NUM = 8;                                                                // kv头数
const uint32_t NOPE_HEAD_SIZE = 128;                                                           // 头大小
const uint32_t ROPE_HEAD_SIZE = 64;                                                            // 头大小
const uint32_t MASK_SEQ_LEN = 512;                                                             // mask的block大小
const int32_t MASK_INDEX = 5;                                                                  // mask对应索引
const int32_t SEQLEN_INDEX = 6;                                                                // seqlen对应索引

/**
 * @brief 随机填充inData数值，值域为[low, high)
 * @param inData 需要随机填充的vector
 * @param low 最小值
 * @param high 最大值
 */
void AssignRandomValue(std::vector<float> &inData, int low = -5, int high = -5)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(low, high); // 生成[low,high)正态分布数据
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
atb::Status PrepareFirstRingInTensor(atb::Context *contextPtr, aclrtStream stream, std::vector<int32_t> &seqLenHost,
                                     atb::SVector<atb::Tensor> &inTensors)
{
    atb::Tensor tensorQNope; // q不带rope转置分量
    atb::Tensor tensorQRope; // q带rope转置分量
    atb::Tensor tensorKNope;
    atb::Tensor tensorKRope;
    atb::Tensor tensorV;
    atb::Tensor tensorMask;
    std::vector<float> maskData(MASK_SEQ_LEN * MASK_SEQ_LEN, 0);
    // 创建512x512，值为1的上三角mask，高精度时使用1为填充值
    for (int i = 0; i < MASK_SEQ_LEN; ++i) {
        for (int j = i + 1; j < MASK_SEQ_LEN; ++j) {
            maskData[i * MASK_SEQ_LEN + j] = 1;
        }
    }

    atb::Tensor tensorSeqLen;
    std::vector<std::vector<int64_t>> tensorDim = {
        {Q_NTOKENS, HEAD_NUM, NOPE_HEAD_SIZE},     // qNope
        {Q_NTOKENS, HEAD_NUM, ROPE_HEAD_SIZE},     // qRope
        {kV_NTOKENS, KV_HEAD_NUM, NOPE_HEAD_SIZE}, // kNope
        {kV_NTOKENS, KV_HEAD_NUM, ROPE_HEAD_SIZE}, // kRope
        {kV_NTOKENS, KV_HEAD_NUM, NOPE_HEAD_SIZE}, // v
        {MASK_SEQ_LEN, MASK_SEQ_LEN},              // mask
        {2, BATCH_SIZE},                           // seqLen
    };
    std::vector<float> randomData = {};
    int64_t totalSize = 0;
    // 根据顺序将所有输入tensor放入SVector
    inTensors = {tensorQNope, tensorQRope, tensorKNope, tensorKRope, tensorV, tensorMask, tensorSeqLen};
    for (int32_t i = 0; i <= SEQLEN_INDEX; ++i) {
        if (i == MASK_INDEX) {
            CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, maskData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                                {MASK_SEQ_LEN, MASK_SEQ_LEN}, inTensors[i]));
        } else if (i == SEQLEN_INDEX) {
            CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {2, BATCH_SIZE}, inTensors[i]));
            inTensors[i].hostData = seqLenHost.data(); // host侧tensor，拷贝值
        } else {
            totalSize = 1;
            for (int j = 0; j < tensorDim[i].size(); ++j) {
                totalSize *= tensorDim[i][j];
            }
            randomData.reserve(totalSize);
            randomData.resize(totalSize);
            AssignRandomValue(randomData);
            CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, randomData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                                tensorDim[i], inTensors[i]));
        }
    }
    return atb::ErrorType::NO_ERROR;
}

atb::Status RunRingMLADemo(atb::Context *contextPtr, aclrtStream stream, atb::Operation *ringMLAOp)
{
    atb::infer::RingMLAParam ringMLAParam;
    // 场景：第一轮选择first ring，不传入prevOut和prevLse
    ringMLAParam.calcType = atb::infer::RingMLAParam::CalcType::CALC_TYPE_FISRT_RING;
    ringMLAParam.headNum = HEAD_NUM;      // query头数
    ringMLAParam.kvHeadNum = KV_HEAD_NUM; // key，value 头数
    // query， key分带、不带rope切分传入，实际headSize为俩个headSize之和；Q*K^T后的缩放系数，根据实际headSize做归一化
    ringMLAParam.qkScale = 1 / sqrt(NOPE_HEAD_SIZE + ROPE_HEAD_SIZE);
    // 计算类型，高精度，softmax使用float32
    ringMLAParam.kernelType = atb::infer::RingMLAParam::KernelType::KERNELTYPE_HIGH_PRECISION;
    ringMLAParam.maskType = atb::infer::RingMLAParam::MaskType::MASK_TYPE_TRIU; // 上三角mask
    CHECK_STATUS(atb::CreateOperation(ringMLAParam, &ringMLAOp));
    // 准备输入张量
    atb::VariantPack ringMLAVariantPack;
    ringMLAVariantPack.inTensors;
    CHECK_STATUS(
        PrepareFirstRingInTensor(contextPtr, stream, seqLenHost, ringMLAVariantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOutput;
    atb::Tensor tensorSoftmaxLse;
    CreateTensor(ACL_BF16, aclFormat::ACL_FORMAT_ND, {Q_NTOKENS, HEAD_NUM, NOPE_HEAD_SIZE}, tensorOutput);
    CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {HEAD_NUM, Q_NTOKENS}, tensorSoftmaxLse);
    ringMLAVariantPack.outTensors = {tensorOutput, tensorSoftmaxLse}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(ringMLAOp->Setup(ringMLAVariantPack, workspaceSize, contextPtr));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // 计算第一轮输出
    CHECK_STATUS(ringMLAOp->Execute(ringMLAVariantPack, workspacePtr, workspaceSize, contextPtr));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    // 第一轮计算后产生output和softmaxLsekey放入后一轮计算
    // 场景：第二轮，此时修改场景为default，即intensor输入prevOut和prevLse计算
    ringMLAParam.calcType = atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT;
    CHECK_STATUS(atb::CreateOperation(ringMLAParam, &ringMLAOp));
    // 放入上一轮的output和softmaxLse
    ringMLAVariantPack.inTensors.push_back(tensorOutput);
    ringMLAVariantPack.inTensors.push_back(tensorSoftmaxLse);
    CHECK_STATUS(ringMLAOp->Setup(ringMLAVariantPack, workspaceSize, contextPtr));
    CHECK_STATUS(ringMLAOp->Execute(ringMLAVariantPack, workspacePtr, workspaceSize, contextPtr));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    for (atb::Tensor &inTensor : ringMLAVariantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    return atb::ErrorType::NO_ERROR;
}

int main(int argc, char **argv)
{
    CHECK_STATUS(aclInit(nullptr));
    // 设置卡号、创建context、设置stream
    int32_t deviceId = 1;
    CHECK_STATUS(aclrtSetDevice(deviceId));

    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // RingMLA operation实例
    atb::Operation *ringMLAOp = nullptr;
    RunRingMLADemo(context, stream, ringMLAOp);
    CHECK_STATUS(atb::DestroyOperation(ringMLAOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "RingMLA demo success!" << std::endl;
    return 0;
}
