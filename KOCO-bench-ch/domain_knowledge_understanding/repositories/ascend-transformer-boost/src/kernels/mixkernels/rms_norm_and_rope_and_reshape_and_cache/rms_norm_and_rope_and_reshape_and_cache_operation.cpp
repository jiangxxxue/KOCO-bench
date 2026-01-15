/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"
 
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t DIM_5 = 5;
constexpr uint32_t DIM_6 = 6;
constexpr int64_t IN_TENSOR_NUM = 7;
constexpr int64_t OUT_TENSOR_NUM = 1;
constexpr uint32_t KEY_CACHE_IN_IDX = 6;
constexpr uint32_t KEY_CACHE_OUT_IDX = 0;
constexpr uint32_t TENSOR_INPUT_NUM = 4;
constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
constexpr uint32_t TENSOR_DIMS_SIZE_1 = 1;
constexpr uint32_t TENSOR_DIMS_SIZE_2 = 2;
constexpr uint32_t TENSOR_DIMS_SIZE_3 = 3;
constexpr uint32_t TENSOR_DIMS_SIZE_4 = 4;
constexpr uint32_t HEAD_DIM_CONVENTION_NUM = 16;
constexpr uint32_t CONST_ZERO = 0;
 
namespace AtbOps {
using namespace Mki;
class RmsNormAndRopeAndReshapeAndCacheOperation : public OperationBase {
public:
    explicit RmsNormAndRopeAndReshapeAndCacheOperation(const std::string &opName) noexcept : OperationBase(opName) {}
 
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RmsNormAndRopeAndReshapeAndCache),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("RmsNormAndRopeAndReshapeAndCacheKernel");
    }
 
    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return IN_TENSOR_NUM;
    }
 
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return OUT_TENSOR_NUM;
    }

    bool CheckNormRopeCache(const LaunchParam &launchParam) const
    {
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dims.size() == TENSOR_DIMS_SIZE_3, "dim size of inTensor0 is invalid", return false);
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(inTensor1.desc.dims.size() == TENSOR_DIMS_SIZE_1, "dim size of inTensor1 is invalid", return false);
        auto &inTensor2 = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(inTensor2.desc.dims.size() == TENSOR_DIMS_SIZE_2, "dim size of inTensor2 is invalid", return false);
        auto &inTensor3 = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(inTensor3.desc.dims.size() == TENSOR_DIMS_SIZE_2, "dim size of inTensor3 is invalid", return false);
        auto &inTensor4 = launchParam.GetInTensor(DIM_4);
        MKI_CHECK(inTensor4.desc.dims.size() == TENSOR_DIMS_SIZE_2, "dim size of inTensor4 is invalid", return false);
        auto &inTensor5 = launchParam.GetInTensor(DIM_5);
        MKI_CHECK(inTensor5.desc.dims.size() == TENSOR_DIMS_SIZE_1, "dim size of inTensor5 is invalid", return false);
        auto &inTensor6 = launchParam.GetInTensor(DIM_6);
        MKI_CHECK(inTensor6.desc.dims.size() == TENSOR_DIMS_SIZE_4, "dim size of inTensor6 is invalid", return false);
        MKI_CHECK(inTensor0.desc.dims[DIM_2] == inTensor1.desc.dims[DIM_0],
            "col of rmsNorm/gamma should be same", return false);
        MKI_CHECK(inTensor2.desc.dims[DIM_1] == inTensor3.desc.dims[DIM_1] &&
            inTensor3.desc.dims[DIM_1] == inTensor4.desc.dims[DIM_1],
            "col of keyrope/cos/sin should be same", return false);
        MKI_CHECK(inTensor0.desc.dims[DIM_0] == inTensor2.desc.dims[DIM_0] &&
            inTensor2.desc.dims[DIM_0] == inTensor3.desc.dims[DIM_0] &&
            inTensor3.desc.dims[DIM_0] == inTensor4.desc.dims[DIM_0] &&
            inTensor4.desc.dims[DIM_0] == inTensor5.desc.dims[DIM_0] &&
            inTensor3.desc.dims[DIM_1] == inTensor4.desc.dims[DIM_1],
            "row of rmsNorm/keyrope/cos/sin/slotmapping should be same", return false);
        for (size_t i = 0; i < TENSOR_INPUT_NUM; i++) {
            auto inTensor = launchParam.GetInTensor(i);
            size_t dimNum = inTensor.desc.dims.size();
            for (size_t dimIndex = DIM_0; dimIndex < dimNum; dimIndex++) {
                MKI_CHECK(inTensor.desc.dims[dimIndex] != CONST_ZERO,
                          "Any dimension of the input vector cannot be equal to zero.", return false);
            }
        }
        size_t headDim = inTensor0.desc.dims[DIM_1];
        MKI_CHECK(headDim == 1, "headDim is invalid. It must be 1.", return false);
        size_t rmsNormSize = inTensor0.desc.dims[DIM_2];
        size_t keyRopeSize = inTensor2.desc.dims[DIM_1];
        size_t cacheSize = inTensor6.desc.dims[DIM_3];
        MKI_CHECK(rmsNormSize % HEAD_DIM_CONVENTION_NUM == CONST_ZERO&&
            keyRopeSize % HEAD_DIM_CONVENTION_NUM == CONST_ZERO&&
            cacheSize % HEAD_DIM_CONVENTION_NUM == CONST_ZERO,
            "inputSize is invalid. It must be a multiple of 16.", return false);
        MKI_CHECK(rmsNormSize + keyRopeSize == cacheSize,
            "inputSize is invalid. rmsNormSize + keyRopeSize must be equle to cacheSize.", return false);
        return true;
    }
 
protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckNormRopeCache(launchParam), "Failed to check run info",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_LOG(INFO) << "RmsNormAndRopeAndReshapeAndCache InferShapeImpl launchParam.GetInTensor().desc="
                      << launchParam.GetInTensor(KEY_CACHE_IN_IDX).desc.ToString();
        outTensors[KEY_CACHE_OUT_IDX].desc = launchParam.GetInTensor(KEY_CACHE_IN_IDX).desc;
 
        return Status::OkStatus();
    }
};
REG_OPERATION(RmsNormAndRopeAndReshapeAndCacheOperation);
} // namespace AtbOps