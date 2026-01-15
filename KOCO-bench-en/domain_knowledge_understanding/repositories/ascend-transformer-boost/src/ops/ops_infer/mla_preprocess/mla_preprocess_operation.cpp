/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mla_preprocess_operation.h"
#include "mla_preprocess_aclnn_runner.h"
#include "mla_preprocess_ops_runner.h"
#include "mla_preprocess_ops_runner_split.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 24;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM_SPLIT = 4;
static const uint32_t INPUT_INDEX = 0;
static const uint32_t GAMMA0_INDEX = 1;
static const uint32_t BETA0_INDEX = 2;
static const uint32_t WDQKV_INDEX = 5;
static const uint32_t BIAS0_INDEX = 7;
static const uint32_t BIAS1_INDEX = 14;
static const uint32_t WUK_INDEX = 18;
static const uint32_t KVCACHE_INDEX = 19;
static const uint32_t KVCACHE_ROPE_INDEX = 20;
static const uint32_t QOUT1_INDEX = 2;
static const uint32_t KVCACHEOUT1_INDEX = 3;
static const uint32_t INNER_DIM_7168 = 7168;
static const uint32_t INNER_DIM_2112 = 2112;
static const uint32_t INNER_DIM_1536 = 1536;
static const uint32_t INNER_DIM_576 = 576;
static const uint32_t INNER_DIM_512 = 512;
static const uint32_t INNER_DIM_192 = 192;
static const uint32_t INNER_DIM_128 = 128;
static const uint32_t INNER_DIM_64 = 64;
static const int64_t MAX_TOKEN_NUM = 1024;
static const int64_t MAX_HIDDEN_SIZE = 8192;
static const int64_t MIN_HIDDEN_SIZE = 2048;
} // namespace

namespace atb {

template <> Status CreateOperation(const infer::MlaPreprocessParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    ATB_LOG(INFO) << "CreateOperation with MlaPreprocessParam: " << OpParamToJson(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2/A3 Inference Product";
        return ERROR_INVALID_PARAM;
    }

    if (opParam.cacheMode > infer::MlaPreprocessParam::CacheMode::NZCACHE) {
        ATB_LOG(ERROR) << "invalid cacheMode";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.quantMode == infer::MlaPreprocessParam::QuantMode::PER_TOKEN_QUANT_ASYMM ||
        opParam.quantMode == infer::MlaPreprocessParam::QuantMode::UNQUANT) {
        ATB_LOG(ERROR) << "invalid quantMode, doesn't support PER_TOKEN_ASYMM_QUANT and UNQUANT";
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new (std::nothrow) MlaPreprocessOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new MlaPreprocessOperation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

MlaPreprocessOperation::MlaPreprocessOperation(const infer::MlaPreprocessParam &param)
    : OperationBase("MlaPreprocessOperation"), param_(param)
{
    std::string opIrKeyStr;
    opIrKeyStr += "MlaPreprocessOperation";
    if (param.cacheMode == infer::MlaPreprocessParam::CacheMode::KROPE_CTKV) {
        opIrKeyStr += "Split";
    }
    if (param.cacheMode == infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE) {
        opIrKeyStr += "INT8NZ";
    }
    if (param.cacheMode == infer::MlaPreprocessParam::CacheMode::NZCACHE) {
        opIrKeyStr += "NZ";
    }
    if (param.quantMode == infer::MlaPreprocessParam::QuantMode::PER_TOKEN_QUANT_SYMM) {
        opIrKeyStr += "Pertoken";
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKeyStr);
}

MlaPreprocessOperation::~MlaPreprocessOperation() {}

uint32_t MlaPreprocessOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t MlaPreprocessOperation::GetOutputNum() const
{
    if (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        return OUT_TENSOR_NUM;
    }
    return OUT_TENSOR_NUM_SPLIT;
}

Status MlaPreprocessOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                              SVector<TensorDesc> &outTensorDescs) const
{
    bool isSplit = param_.cacheMode == infer::MlaPreprocessParam::CacheMode::KROPE_CTKV;
    bool isint8Nz = param_.cacheMode == infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE;
    bool isNz = param_.cacheMode == infer::MlaPreprocessParam::CacheMode::NZCACHE;
    int64_t tokenNum = inTensorDescs.at(0).shape.dims[0];
    int64_t headNum = inTensorDescs.at(WUK_INDEX).shape.dims[0];
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dimNum = 3;              // 3 : dimNum, [tokenNum, headNum, headDim]
    outTensorDescs.at(0).shape.dims[0] = tokenNum;      // 0 : dim 0
    outTensorDescs.at(0).shape.dims[1] = headNum;       // 1 : dim 1
    outTensorDescs.at(0).shape.dims[2] = INNER_DIM_576; // 2 : dim 2
    outTensorDescs.at(1) = inTensorDescs.at(KVCACHE_INDEX);
    if (isSplit) {
        outTensorDescs.at(0).shape.dims[2] = INNER_DIM_512; // 2 : dim 2
        outTensorDescs.at(QOUT1_INDEX) = outTensorDescs.at(0);
        outTensorDescs.at(QOUT1_INDEX).shape.dims[2] = INNER_DIM_64; // 2 : dim 2
        outTensorDescs.at(KVCACHEOUT1_INDEX) = inTensorDescs.at(KVCACHE_ROPE_INDEX);
    }
    if (isint8Nz) {
        outTensorDescs.at(0).shape.dims[2] = INNER_DIM_512; // 2 : dim 2
        outTensorDescs.at(QOUT1_INDEX) = outTensorDescs.at(0);
        outTensorDescs.at(QOUT1_INDEX).shape.dims[2] = INNER_DIM_64; // 2 : dim 2
        outTensorDescs.at(KVCACHEOUT1_INDEX) = inTensorDescs.at(KVCACHE_ROPE_INDEX);
        outTensorDescs.at(0).dtype = ACL_INT8;
    }
    if (isNz) {
        outTensorDescs.at(0).shape.dims[2] = INNER_DIM_512; // 2 : dim 2
        outTensorDescs.at(QOUT1_INDEX) = outTensorDescs.at(0);
        outTensorDescs.at(QOUT1_INDEX).shape.dims[2] = INNER_DIM_64; // 2 : dim 2
        outTensorDescs.at(KVCACHEOUT1_INDEX) = inTensorDescs.at(KVCACHE_ROPE_INDEX);
    }
    return NO_ERROR;
}

Status MlaPreprocessOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status MlaPreprocessOperation::OutTensorCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    int64_t tokenNum = inTensors.at(0).desc.shape.dims[0];
    int64_t headNum = inTensors.at(WUK_INDEX).desc.shape.dims[0];
    if (outTensors.at(0).desc.shape.dimNum != 3 || // 3 : qOut dimNum
        outTensors.at(1).desc.shape.dimNum != 4) { // 4 : kvCacheOut dimNum
        ATB_LOG(ERROR) << "outTensor0 dimNum : " << outTensors.at(0).desc.shape.dimNum << "expect 3"
                       << " outTensor1 dimNum : " << outTensors.at(1).desc.shape.dimNum << "expect 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (outTensors.at(0).desc.shape.dims[0] != tokenNum ||      // 0 : dim 0
        outTensors.at(0).desc.shape.dims[1] != headNum ||       // 1 : dim 1
        outTensors.at(0).desc.shape.dims[2] != INNER_DIM_576) { // 2 : dim 2
        ATB_LOG(ERROR) << "outTensor0 dims : [" << outTensors.at(0).desc.shape.dims[0] << ","
                       << outTensors.at(0).desc.shape.dims[1] << "," << outTensors.at(0).desc.shape.dims[2] // 2 : dim 2
                       << "], expect [tokenNum,headNum,576]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorUtil::TensorDescEqual(inTensors.at(KVCACHE_INDEX).desc, outTensors.at(1).desc)) {
        ATB_LOG(ERROR) << "kvCacheIn and kvCacheOut have different Desc";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MlaPreprocessOperation::OutTensorCheckSplit(const SVector<Tensor> &inTensors,
                                                   const SVector<Tensor> &outTensors) const
{
    int64_t tokenNum = inTensors.at(0).desc.shape.dims[0];
    int64_t headNum = inTensors.at(WUK_INDEX).desc.shape.dims[0];
    if (outTensors.at(0).desc.shape.dimNum != 3 || // 3: qOut dimNum
        outTensors.at(1).desc.shape.dimNum != 4 || // 4: kvCacheOut dimNum
        outTensors.at(2).desc.shape.dimNum != 3 || // 2: dim2 3: qOut dimNum
        outTensors.at(3).desc.shape.dimNum != 4) { // 3: dim3 4: kvCacheOut dimNum
        ATB_LOG(ERROR) << "outTensor0 dimNum : " << outTensors.at(0).desc.shape.dimNum << "expect 3"
                       << " outTensor1 dimNum : " << outTensors.at(1).desc.shape.dimNum << "expect 4"
                       << " outTensor2 dimNum : " << outTensors.at(2).desc.shape.dimNum << "expect 3"  // 2: dim2
                       << " outTensor3 dimNum : " << outTensors.at(3).desc.shape.dimNum << "expect 4"; // 3: dim3
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (outTensors.at(0).desc.shape.dims[0] != tokenNum ||      // 0: dim 0
        outTensors.at(0).desc.shape.dims[1] != headNum ||       // 1: dim 1
        outTensors.at(0).desc.shape.dims[2] != INNER_DIM_512) { // 2: dim 2
        ATB_LOG(ERROR) << "outTensor0 dims expect [tokenNum,headNum,512]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(1).desc.shape.dims[0] != inTensors.at(KVCACHE_INDEX).desc.shape.dims[0] || // 0: dim 0
        outTensors.at(1).desc.shape.dims[1] != inTensors.at(KVCACHE_INDEX).desc.shape.dims[1] || // 1: dim 1
        outTensors.at(1).desc.shape.dims[2] != 1 ||                                              // 2: dim 2
        outTensors.at(1).desc.shape.dims[3] != INNER_DIM_512) {                                  // 3: dim 3
        ATB_LOG(ERROR) << "outTensor1 dims expect [blockNum,blockSize,1,512]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(2).desc.shape.dims[0] != tokenNum ||     // 2: tensor2 0 : dim 0
        outTensors.at(2).desc.shape.dims[1] != headNum ||      // 2: tensor2 1 : dim 1
        outTensors.at(2).desc.shape.dims[2] != INNER_DIM_64) { // 2: tensor2 2 : dim 2
        ATB_LOG(ERROR) << "outTensor2 dims expect [tokenNum,headNum,64]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(3).desc.shape.dims[0] != inTensors.at(KVCACHE_INDEX).desc.shape.dims[0] || // 3: tensor3 0: dim 0
        outTensors.at(3).desc.shape.dims[1] != inTensors.at(KVCACHE_INDEX).desc.shape.dims[1] || // 3: tensor3 1: dim 1
        outTensors.at(3).desc.shape.dims[2] != 1 ||                                              // 3: tensor3 2: dim 2
        outTensors.at(3).desc.shape.dims[3] != INNER_DIM_64) {                                   // 3: tensor3 3: dim 3
        ATB_LOG(ERROR) << "outTensor3 dims expect [blockNum,blockSize,1,64]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MlaPreprocessOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetupCheckImpl";
    SVector<TensorDesc> inTensorDesc = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDesc);
    Status st = DimCheck(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::KROPE_CTKV) {
        st = OutTensorCheckSplit(inTensors, outTensors);
    } else if (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE) {
        st = NO_ERROR;
    } else if (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::NZCACHE) {
        st = NO_ERROR;
    } else {
        st = OutTensorCheck(inTensors, outTensors);
    }
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status MlaPreprocessOperation::TensorShapeCheck(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                                const std::vector<std::vector<int64_t>> &inTensorShapes) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "TensorShapeCheck start";
    if (inTensorDesc.size() != inTensorShapes.size()) {
        ATB_LOG(ERROR) << "inTensor num is not equal to the expect num : " << inTensorShapes.size();
        return ERROR_INVALID_IN_TENSOR_NUM;
    }
    for (size_t i = 0; i < inTensorDesc.size(); i++) {
        ATB_LOG(INFO) << GetLogPrefix() << "TensorShapeCheck index [" << i << "]";
        if (inTensorShapes.at(i).size() == 0 || inTensorDesc.at(i).shape.dimNum == 0) {
            continue;
        }
        if (inTensorDesc.at(i).shape.dimNum != inTensorShapes.at(i).size()) {
            ATB_LOG(ERROR) << "inTensor" << i << " dimnum " << inTensorDesc[i].shape.dimNum
                           << " is not equal to the expect dimnum : " << inTensorShapes[i].size();
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
        for (size_t j = 0; j < inTensorDesc[i].shape.dimNum; j++) {
            if (inTensorShapes[i][j] != -1 && inTensorDesc.at(i).shape.dims[j] != inTensorShapes[i][j]) {
                ATB_LOG(ERROR) << "inTensor" << i << " dim" << j << " size:" << inTensorDesc.at(i).shape.dims[j]
                               << " is not equal to the expect dim size : " << inTensorShapes[i][j];
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "TensorShapeCheck finished";
    return NO_ERROR;
}

void MlaPreprocessOperation::ModifyInTensorShapeTable(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                                      const std::vector<std::vector<int64_t>> &inTensorShapes,
                                                      const int64_t headNum) const
{
    (void)inTensorDesc;
    (void)inTensorShapes;
    (void)headNum;
}

Status MlaPreprocessOperation::DimCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "DimCheck start";
    int64_t tokenNum = inTensorDesc.at(0).shape.dims[0];
    int64_t headNum = inTensorDesc.at(WUK_INDEX).shape.dims[0];
    if (tokenNum <= 0 || headNum <= 0) {
        ATB_LOG(ERROR) << "tokenNum and headNum should > 0";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (tokenNum > MAX_TOKEN_NUM) {
        ATB_LOG(ERROR) << "tokenNum should <= " << MAX_TOKEN_NUM;
        return ERROR_INVALID_TENSOR_DIM;
    }
    Status st = BlockSizeCheck(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    st = CheckAclnnKernel(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    st = HiddenSizeCheck(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    std::vector<std::vector<int64_t>> inTensorShapes = {
        {},                        // input
        {},                        // gamma0
        {},                        // beta0
        {1},                       // quantScale0
        {1},                       // quantOffset0
        {},                        // wdqkv
        {INNER_DIM_2112},          // deScale0
        {},                        // bias0
        {INNER_DIM_1536},          // gamma1
        {INNER_DIM_1536},          // beta1
        {1},                       // quantScale1
        {1},                       // quantOffset1
        {},                        // wuq
        {headNum * INNER_DIM_192}, // deScale1
        {},                        // bias1
        {INNER_DIM_512},           // gamma2
        {tokenNum, INNER_DIM_64},  // cos
        {tokenNum, INNER_DIM_64},  // sin
        {},                        // wuk
        {},                        // ctkv
        {},                        // kRope
        {tokenNum},                // slotmapping
        {},                        // ctkvScale
        {},                        // qNopeScale
    };
    ModifyInTensorShapeTable(inTensorDesc, inTensorShapes, headNum);
    st = TensorShapeCheck(inTensorDesc, inTensorShapes);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "DimCheck finished";
    return NO_ERROR;
}

Status MlaPreprocessOperation::BlockSizeCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "BlockSizeCheck start";
    bool isNz = (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE ||
                 param_.cacheMode == infer::MlaPreprocessParam::CacheMode::NZCACHE);
    int64_t blockSize = isNz ? inTensorDesc.at(KVCACHE_INDEX).shape.dims[2] : // 2: blocksizeIdx
                               inTensorDesc.at(KVCACHE_INDEX).shape.dims[1];
    if (blockSize <= 0 || (blockSize > 128 && blockSize != 256)) { // 128 , 256 : blockSize ∈ [1, 128]∪{256}
        ATB_LOG(ERROR) << "blockSize should > 0 and <= 128 or == 256";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (isNz && blockSize != 128) { // 128: blocksize
        ATB_LOG(ERROR) << "blockSize should be 128 with INT8_NZCACHE or NZCACHE";
        return ERROR_INVALID_TENSOR_DIM;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "BlockSizeCheck finished";
    return NO_ERROR;
}

Status MlaPreprocessOperation::HiddenSizeCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    if (inTensorDesc.at(INPUT_INDEX).shape.dimNum != 2) { // 2: input [tokenNum, hiddenSize]
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dimNum of input should be 2, but got: " << inTensorDesc.at(INPUT_INDEX).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t hiddenSize = inTensorDesc.at(INPUT_INDEX).shape.dims[1]; // 1: input hiddenSize index
    if (useAclnnKernel_) {
        if (hiddenSize < MIN_HIDDEN_SIZE || hiddenSize > MAX_HIDDEN_SIZE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "expect hiddenSize of input to be in range [" << MIN_HIDDEN_SIZE << ","
                           << MAX_HIDDEN_SIZE << "], but got: " << hiddenSize;
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else if (hiddenSize != INNER_DIM_7168) {
        ATB_LOG(ERROR) << GetLogPrefix() << "expect hiddenSize of input to be" << INNER_DIM_7168
                       << ", but got: " << hiddenSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(GAMMA0_INDEX).shape.dimNum != 1) { // 1: input [hiddenSize]
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dimNum of gamma0 should be 1, but got: " << inTensorDesc.at(GAMMA0_INDEX).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDesc.at(GAMMA0_INDEX).shape.dims[0] != hiddenSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSize of gamma0 should be the same as input's, " << hiddenSize
                       << ", but got: " << inTensorDesc.at(GAMMA0_INDEX).shape.dims[0]; // gamma0 hiddenSize index
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(BETA0_INDEX).shape.dimNum != 1) { // 1: input [hiddenSize]
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dimNum of beta0 should be 1, but got: " << inTensorDesc.at(BETA0_INDEX).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDesc.at(BETA0_INDEX).shape.dims[0] != hiddenSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSize of beta0 should be the same as input's, " << hiddenSize
                       << ", but got: " << inTensorDesc.at(BETA0_INDEX).shape.dims[0]; // beta0 hiddenSize index
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MlaPreprocessOperation::CheckAclnnKernel(const SVector<TensorDesc> &inTensorDesc) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "CheckAclnnKernel start";
    // input's hiddenSize != 7168, generalize hiddenSize
    bool generalizedHiddenSize = inTensorDesc.at(INPUT_INDEX).shape.dims[1] != INNER_DIM_7168; // 1: hiddenSize
    // if wdqkv's dtype is the same as the input and is either float16/bf16, then do not do rmsNormQuant
    aclDataType inputDtype = inTensorDesc.at(INPUT_INDEX).dtype;
    doRmsNorm_ =
        !(inTensorDesc.at(WDQKV_INDEX).dtype == inputDtype && (inputDtype == ACL_FLOAT16 || inputDtype == ACL_BF16));
    if (!generalizedHiddenSize && doRmsNorm_) {
        ATB_LOG(INFO) << GetLogPrefix()
                      << "no need to use aclnn kernel for non-generalized hiddenSize and rmsNormQuant for input is on";
        useAclnnKernel_ = false;
        return NO_ERROR;
    }
    useAclnnKernel_ = true;
    Status ret = MlaPreprocessAclnnRunner::LoadMethod();
    ATB_LOG(INFO) << GetLogPrefix() << "MlaPreprocessAclnnRunner::LoadMethod() ret: " << ret;
    if (ret != NO_ERROR) {
        if (generalizedHiddenSize) {
            ATB_LOG(INFO) << GetLogPrefix()
                          << "Need to use aclnn kernel for generalized hiddenSize but load methods failed! Consider "
                             "change hiddenSize back to "
                          << INNER_DIM_7168 << " to use the atb kernel";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
        if (!doRmsNorm_) {
            ATB_LOG(INFO) << GetLogPrefix()
                          << "Need to use aclnn kernel while skipping rmsNormQuant for input but load methods failed!";
            return ret;
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn kernel is required and usable, generalizedHiddenSize: " << generalizedHiddenSize
                  << ", doRmsNorm: " << doRmsNorm_;
    return NO_ERROR;
}

std::shared_ptr<Runner> MlaPreprocessOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (useAclnnKernel_) {
        ATB_LOG(INFO) << GetLogPrefix() << "create MlaPreprocess AclnnRunner";
        bool doRmsNormParam = doRmsNorm_;
        useAclnnKernel_ = false;
        doRmsNorm_ = true;
        return std::make_shared<MlaPreprocessAclnnRunner>(param_, doRmsNormParam);
    }

    ATB_LOG(INFO) << GetLogPrefix() << "create MlaPreprocess OpsRunner";
    if (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        return std::make_shared<MlaPreprocessOpsRunner>(param_);
    }
    return std::make_shared<MlaPreprocessOpsRunnerSplit>(param_);
}

nlohmann::json MlaPreprocessOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
