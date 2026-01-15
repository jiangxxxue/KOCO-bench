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

const int32_t DEVICE_ID = 2;
const uint32_t blockSize = 128;
const uint32_t blockNum = 161;

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::SVector<atb::Tensor> *atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor1(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum,
                             atb::SVector<atb::Tensor> *inTensors)
{
    // 创建shape为[32, 7168]的输入input tensor
    atb::Tensor input;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * 7168, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, 7168}, input, dtype));
    // 创建shape为[7168]的输入gamma0 tensor
    atb::Tensor gamma0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(7168, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {7168}, gamma0, dtype));
    // 创建shape为[7168]的输入beta0 tensor
    atb::Tensor beta0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(7168, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {7168}, beta0, dtype));
    // 创建shape为[1]的输入quantScale0 tensor
    atb::Tensor quantScale0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, quantScale0, dtype));
    // 创建shape为[1]的输入quantOffset0 tensor
    atb::Tensor quantOffset0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(1, 1), ACL_INT8,
                                        aclFormat::ACL_FORMAT_ND, {1}, quantOffset0));
    // 创建shape为[2112, 7168]的输入wdqkv tensor
    atb::Tensor wdqkv;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(2112 * 7168), ACL_INT8,
                                        aclFormat::ACL_FORMAT_FRACTAL_NZ, {2112, 7168}, wdqkv, ACL_INT8));
    // 创建shape为[2112]的输入deScale0 tensor
    atb::Tensor deScale0;
    if (dtype == ACL_BF16) {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(2112, 1), ACL_FLOAT,
                                            aclFormat::ACL_FORMAT_ND, {2112}, deScale0));
    } else {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int64_t>(2112, 10), ACL_INT64,
                                            aclFormat::ACL_FORMAT_ND, {2112}, deScale0));
    }
    // 创建shape为[2112]的输入bias0 tensor
    atb::Tensor bias0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(2112, 1), ACL_INT32,
                                        aclFormat::ACL_FORMAT_ND, {2112}, bias0));
    // 创建shape为[1536]的输入gamma1 tensor
    atb::Tensor gamma1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1536, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {1536}, gamma1, dtype));
    // 创建shape为[1536]的输入beta1 tensor
    atb::Tensor beta1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1536, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {1536}, beta1, dtype));
    // 创建shape为[1]的输入quantScale1 tensor
    atb::Tensor quantScale1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, quantScale1, dtype));
    // 创建shape为[1]的输入quantOffset1 tensor
    atb::Tensor quantOffset1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(1, 1), ACL_INT8,
                                        aclFormat::ACL_FORMAT_ND, {1}, quantOffset1));
    *inTensors = {input,    gamma0, beta0,  quantScale0, quantOffset0, wdqkv,
                  deScale0, bias0,  gamma1, beta1,       quantScale1,  quantOffset1};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::SVector<atb::Tensor> *atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor2(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum, int headNum,
                             atb::SVector<atb::Tensor> *inTensors)
{
    // 创建shape为[headNum * 192, 48 * 32]的输入wuq tensor
    atb::Tensor wuq;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(headNum * 192 * 48 * 32), ACL_INT8,
                                        aclFormat::ACL_FORMAT_FRACTAL_NZ, {headNum * 192, 48 * 32}, wuq));
    // 创建shape为[headNum * 192]的输入deScale1 tensor
    atb::Tensor deScale1;
    if (dtype == ACL_BF16) {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(headNum * 192, 1), ACL_FLOAT,
                                            aclFormat::ACL_FORMAT_ND, {headNum * 192}, deScale1));
    } else {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int64_t>(headNum * 192, 10), ACL_INT64,
                                            aclFormat::ACL_FORMAT_ND, {headNum * 192}, deScale1));
    }
    // 创建shape为[headNum*192]的输入bias1 tensor
    atb::Tensor bias1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(headNum * 192, 1), ACL_INT32,
                                        aclFormat::ACL_FORMAT_ND, {headNum * 192}, bias1));
    // 创建shape为[512]的输入gamma2 tensor
    atb::Tensor gamma2;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(512, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {512}, gamma2, dtype));
    // 创建shape为[tokenNum,64]的输入cos tensor
    atb::Tensor cos;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * 64, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, 64}, cos, dtype));
    // 创建shape为[tokenNum,64]的输入sin tensor
    atb::Tensor sin;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * 64, 0.5), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, 64}, sin, dtype));
    // 创建shape为[headNum,128,512]的输入wuk tensor
    atb::Tensor wuk;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(headNum * 128 * 512), dtype,
                                        aclFormat::ACL_FORMAT_ND, {headNum, 128, 512}, wuk, dtype));
    // 创建shape为[blockNum, block_size, 1, 512]的输入kvCache tensor
    atb::Tensor kvCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(blockNum * 512 * blockSize * 1), dtype,
                                        aclFormat::ACL_FORMAT_ND, {blockNum, blockSize, 1, 512}, kvCache, dtype));
    // 创建shape为[blockNum, blockSize, 1, 64]的输入kvCacheRope tensor
    atb::Tensor kvCacheRope;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(blockNum * blockSize * 64, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {blockNum, blockSize, 1, 64}, kvCacheRope, dtype));
    auto slotmappingHost = std::vector<int32_t>(1, tokenNum);
    for (size_t i = 0; i < slotmappingHost.size(); i++)
        slotmappingHost[i] = static_cast<int32_t>(i);
    // 创建shape为[headNum]的输入slotmapping tensor
    atb::Tensor slotmapping;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, slotmappingHost, ACL_INT32, aclFormat::ACL_FORMAT_ND,
                                        {tokenNum}, slotmapping));
    // 创建shape为[1]的输入ctkvScale tensor
    atb::Tensor ctkvScale;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, ctkvScale, dtype));
    // 创建shape为[headNum]的输入qNopeScale tensor
    atb::Tensor qNopeScale;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, qNopeScale, dtype));
    atb::SVector<atb::Tensor> tempTensors = {wuq, deScale1, bias1,       gamma2,      cos,       sin,
                                             wuk, kvCache,  kvCacheRope, slotmapping, ctkvScale, qNopeScale};
    for (auto &tensor : tempTensors) {
        inTensors->push_back(tensor);
    }
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个MlaPreprocessOperation operation
 * @param atb::Operation * 返回一个Operation指针
 */
atb::Status CreateMlaPreprocessOperation(atb::Operation **mlaPreprocessOp)
{
    atb::infer::MlaPreprocessParam param;
    param.wdqDim = 1536;  // 经过matMul后拆分的dim大小
    param.qRopeDim = 64;  // q传入rope的dim大小
    param.kRopeDim = 64;  // k传入rope的dim大小
    param.epsilon = 1e-5; // rmsNorm公式中加在分母上防止除零的值
    param.qRotaryCoeff = 2; // query和key分别取一半，对半旋转，对应rotaryCoeff=2
    param.kRotaryCoeff = 2; // 对半旋转，对应rotaryCoeff=2
    param.transposeWdq = true; // 转置wdq，用于matMul加速
    param.transposeWuq = true; // 转置wuq
    param.transposeWuk = true; // 转置wuk
    // cache类型，输入的q拆分为qNope和qRope，输入的kvCache拆分为ctKV和kRope
    param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::KROPE_CTKV;
    param.quantMode = atb::infer::MlaPreprocessParam::QuantMode::PER_TENSOR_QUANT_ASYMM;
    return atb::CreateOperation(param, mlaPreprocessOp);
}

/**
 * @brief 进行MlaPreprocessOperation的循环调用
 * @param context context指针
 * @param stream stream
 * @param dtype 指定部分输入/输出vector数据类型
 * @param tokenNum 词元数
 * @param headNum 头数
 * @return atb::Status 错误码
 */
atb::Status RunDemo(atb::Context *context, void *stream, aclDataType dtype, int tokenNum, int headNum)
{
    // 创建op
    atb::Operation *mlaPreprocessOp = nullptr;
    CHECK_STATUS(CreateMlaPreprocessOperation(&mlaPreprocessOp));
    // 准备输入tensor
    atb::VariantPack variantPack;
    // 放入rmsNormQuant_0，matmul_0，rmsNormQuant_1输入tensor
    CHECK_STATUS(PrepareInTensor1(context, stream, dtype, tokenNum, &variantPack.inTensors));
    // 放入matmul_1，rmsNorm，rope，matmulEin，reshapeAndCache，quant输入tensor
    CHECK_STATUS(PrepareInTensor2(context, stream, dtype, tokenNum, headNum, &variantPack.inTensors));
    // 准备输出tensor
    atb::Tensor qOut0;
    CreateTensor(dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 512}, qOut0);
    atb::Tensor &kvCacheOut0 = variantPack.inTensors.at(19);
    atb::Tensor qOut1;
    CreateTensor(dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 64}, qOut1);
    atb::Tensor &kvCacheOut1 = variantPack.inTensors.at(20);
    variantPack.outTensors = {qOut0, kvCacheOut0, qOut1, kvCacheOut1}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(mlaPreprocessOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    for (size_t i = 0; i < 10; i++) {
        std::cout << "tokenNum: " << tokenNum << " headNum: " << headNum << " loop: " << i << std::endl;
        // mlaPreprocess执行
        CHECK_STATUS(mlaPreprocessOp->Execute(variantPack, workspacePtr, workspaceSize, context));
        CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    }
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
        for (atb::Tensor &outTensor : variantPack.outTensors) {
            if (outTensor.deviceData == inTensor.deviceData) {
                outTensor.deviceData = nullptr;
            }
        }
        inTensor.deviceData = nullptr;
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        if (outTensor.deviceData == nullptr)
            continue;
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    return atb::DestroyOperation(mlaPreprocessOp); // operation，对象概念，先释放
}


int main(int argc, char **argv)
{
    std::string dtypeStr;
    int tokenNum = 32;
    int headNum = 128;
    aclDataType dtype = ACL_BF16;
    if (argc == 4) {
        dtypeStr = argv[1];
        tokenNum = std::stoi(argv[2]);
        headNum = std::stoi(argv[3]);
    }
    if (dtypeStr == "bf16") {
        dtype = ACL_BF16;
    }
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);
    RunDemo(context, stream, dtype, tokenNum, headNum);
    // 释放资源
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "MlaPreprocess demo success!" << std::endl;
    return 0;
}
