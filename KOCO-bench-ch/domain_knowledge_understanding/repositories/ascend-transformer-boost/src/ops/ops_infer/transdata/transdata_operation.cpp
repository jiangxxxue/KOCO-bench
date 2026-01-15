/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transdata_operation.h"
#include "transdata_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t ALIGN_FLOAT16 = 16;
static const uint32_t ALIGN_INT8 = 32;
static const uint32_t ALIGN_BF16 = 16;
static constexpr int64_t DIM_1 = 1;
static constexpr int64_t DIM_2 = 2;
static constexpr int64_t DIM_3 = 3;
static constexpr int64_t DIM_4 = 4;
static const int64_t DEFAULT_ALIGN = 16;

namespace atb {
template <> Status CreateOperation(const infer::TransdataParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new TransdataOperation(opParam);
    return NO_ERROR;
}

TransdataOperation::TransdataOperation(const infer::TransdataParam &param)
    : OperationBase("TransdataOperation"), param_(param)
{
    std::string opIrKey;
    if (GetSingleton<Config>().Is310B()) {
        opIrKey = param_.transdataType == atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ ?
            "TransdataOperationNdToNzAtlas200I500A2" : "TransdataOperationNzToNdAtlas200I500A2";
    } else {
        opIrKey = param_.transdataType == atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ ?
            "TransdataOperationNdToNz" : "TransdataOperationNzToNd";
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey);
}

TransdataOperation::~TransdataOperation() {}

uint32_t TransdataOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t TransdataOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status TransdataOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    static std::map<aclDataType, uint32_t> alignTable = {
        {ACL_FLOAT16, ALIGN_FLOAT16},
        {ACL_INT8, ALIGN_INT8},
        {ACL_BF16, ALIGN_BF16},
    };
    aclDataType inTensorDType = inTensorDescs.at(0).dtype;
    std::map<aclDataType, uint32_t>::const_iterator it = alignTable.find(inTensorDType);
    if (it == alignTable.end()) {
        return ERROR_INVALID_PARAM;
    }
    int64_t align = static_cast<int64_t>(it->second);
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    auto inTensorDims = inTensorDescs.at(0).shape.dimNum;
    if (param_.transdataType == atb::infer::TransdataParam::ND_TO_FRACTAL_NZ) {
        outTensorDescs.at(0).format = ACL_FORMAT_FRACTAL_NZ;
        outTensorDescs.at(0).shape.dimNum = DIM_4;
        const bool dim3Flag = (inTensorDims == DIM_3);
        auto outTensorDim0 = dim3Flag ? inTensorDescs.at(0).shape.dims[0] : 1;
        int64_t outTensorDim1 =
            OperationUtil::RoundUp(inTensorDescs.at(0).shape.dims[dim3Flag ? DIM_2 : DIM_1], align) / align;
        auto outTensorDim2 =
            OperationUtil::RoundUp(inTensorDescs.at(0).shape.dims[dim3Flag ? DIM_1 : 0], DEFAULT_ALIGN);
        int64_t outTensorDim3 = align;
        outTensorDescs.at(0).shape.dims[0] = outTensorDim0;
        outTensorDescs.at(0).shape.dims[DIM_1] = outTensorDim1;
        outTensorDescs.at(0).shape.dims[DIM_2] = outTensorDim2;
        outTensorDescs.at(0).shape.dims[DIM_3] = outTensorDim3;
    } else {
        outTensorDescs.at(0).format = ACL_FORMAT_ND;
        outTensorDescs.at(0).shape.dimNum = DIM_3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        if (param_.outCrops[0] > 0 && param_.outCrops[1] > 0) {
            outTensorDescs.at(0).shape.dims[1] = param_.outCrops[0];
            outTensorDescs.at(0).shape.dims[DIM_2] = param_.outCrops[1];
        } else {
            ATB_LOG(ERROR) << "outCrops can not <= 0!";
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

Status TransdataOperation::OutCropsCheck(const TensorDesc &inTensorDesc) const
{
    int64_t outCrop0Lower = OperationUtil::RoundUp(inTensorDesc.shape.dims[DIM_2], ALIGN_FLOAT16) - ALIGN_FLOAT16;
    int64_t outCrop0Upper = OperationUtil::RoundUp(inTensorDesc.shape.dims[DIM_2], ALIGN_FLOAT16);
    int64_t outCrop1Lower =
        OperationUtil::RoundUp(inTensorDesc.shape.dims[1] * inTensorDesc.shape.dims[DIM_3], ALIGN_FLOAT16) -
        ALIGN_FLOAT16;
    int64_t outCrop1Upper =
        OperationUtil::RoundUp(inTensorDesc.shape.dims[1] * inTensorDesc.shape.dims[DIM_3], ALIGN_FLOAT16);
    if (param_.outCrops[0] > outCrop0Upper || param_.outCrops[0] <= outCrop0Lower ||
        param_.outCrops[1] > outCrop1Upper || param_.outCrops[1] <= outCrop1Lower) {
        ATB_LOG(ERROR) << "outCrops not in the valid range!";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status TransdataOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.transdataType == atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ) {
        if (inTensorDescs.at(0).shape.dimNum != DIM_2 && inTensorDescs.at(0).shape.dimNum != DIM_3) {
            ATB_LOG(ERROR) << GetLogPrefix() << "ND_TO_FRACTAL_NZ inTensor DimNum should be 2 or 3";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else if (param_.transdataType == atb::infer::TransdataParam::TransdataType::FRACTAL_NZ_TO_ND) {
        if (inTensorDescs.at(0).shape.dimNum != DIM_4) {
            ATB_LOG(ERROR) << GetLogPrefix() << "FRACTAL_NZ_TO_ND inTensor should be DimNum 4";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.outCrops.size() != DIM_2) {
            ATB_LOG(ERROR) << "outCrops size should be 2!";
            return ERROR_INVALID_PARAM;
        }
        return OutCropsCheck(inTensorDescs.at(0));
    } else {
        ATB_LOG(ERROR) << "transdataType is not support, only support ND_TO_FRACTAL_NZ and "
                          "FRACTAL_NZ_TO_ND";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status TransdataOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (param_.transdataType == atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ) {
        if ((inTensors.at(0).desc.shape.dimNum != DIM_2 && inTensors.at(0).desc.shape.dimNum != DIM_3) ||
            outTensors.at(0).desc.shape.dimNum != DIM_4) {
            ATB_LOG(ERROR) << GetLogPrefix() << "ND_TO_FRACTAL_NZ inTensor DimNum should be 2 or 3, "
                           << "outTensor DimNum should be 4.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (inTensors.at(0).desc.shape.dimNum != DIM_4 || outTensors.at(0).desc.shape.dimNum != DIM_3) {
            ATB_LOG(ERROR) << GetLogPrefix() << "FRACTAL_NZ_TO_ND inTensor should be DimNum 4, "
                           << "outTensor should be DimNum 3.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.outCrops.size() != DIM_2) {
            ATB_LOG(ERROR) << "outCrops size should be 2!";
            return ERROR_INVALID_PARAM;
        }
        return OutCropsCheck(inTensors.at(0).desc);
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> TransdataOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<TransdataOpsRunner>(param_);
}

nlohmann::json TransdataOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb