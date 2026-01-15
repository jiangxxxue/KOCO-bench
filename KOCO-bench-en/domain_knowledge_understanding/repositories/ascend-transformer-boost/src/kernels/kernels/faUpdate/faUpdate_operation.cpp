/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "asdops/params/params.h"

namespace AsdOps {
using namespace Mki;

static const int32_t IN_TENSOR_NUM = 2;

class FaUpdateOperation : public OperationBase {
public:
    explicit FaUpdateOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        auto inDtype0 = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FaUpdate), "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::FaUpdate>(launchParam.GetParam());
        switch (param.faUpdateType) {
            case OpParam::FaUpdate::DECODE_UPDATE:
                switch (inDtype0) {
                    case TENSOR_DTYPE_FLOAT: return GetKernelByName("DecodeUpdateKernel");
                    default:
                        MKI_LOG(ERROR) << "No supported DecodeUpdate tactic for dtype " << inDtype0;
                        return nullptr;
                }
            default:
                MKI_LOG(ERROR) << "Unsupported FaUpdate tatic type INDEX_UNDEFINED";
                return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FaUpdate), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::FaUpdate>(specificParam);
        switch (param.faUpdateType) {
            case OpParam::FaUpdate::DECODE_UPDATE:
                return IN_TENSOR_NUM;
            default: return 0;
        }
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 1;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FaUpdate),
                     "no match param type", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::FaUpdate>(launchParam.GetParam());

        switch (param.faUpdateType) {
            case OpParam::FaUpdate::DECODE_UPDATE: {
                MKI_CHECK(CheckDecodeUpdateLaunchParam(launchParam), "Failed to Check DecoodeUpdateInferShape",
                             return Status::FailStatus(ERROR_INFERSHAPE_ERROR,
                                                       "Failed to Check DecoodeUpdateInferShape"));
                outTensors[0].desc = launchParam.GetInTensor(1).desc;
                const uint32_t bshc = launchParam.GetInTensor(0).desc.dims.at(1);
                const uint32_t hd = launchParam.GetInTensor(1).desc.dims.at(2);
                outTensors[0].desc.dims = SVector<int64_t>({bshc, hd});
                return Status::OkStatus();
            }
            default: return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported FaUpdate tatic type");
        }
    }

private:
    bool CheckDecodeUpdateLaunchParam(const LaunchParam &launchParam) const
    {
        const int64_t size0 = launchParam.GetInTensor(0).desc.Numel();
        const int64_t size1 = launchParam.GetInTensor(1).desc.Numel();
        const uint32_t sp = launchParam.GetInTensor(0).desc.dims.at(0);
        const uint32_t hd = launchParam.GetInTensor(1).desc.dims.at(2);

        auto param = AnyCast<OpParam::FaUpdate>(launchParam.GetParam());
        MKI_CHECK(param.sp == sp, "sizeof input0 SP should equal to param SP", return false);
        MKI_CHECK(size0 * hd == size1, "sizeof shape0 times hDim should equal to size of shape1", return false);
        return true;
    }
};
REG_OPERATION(FaUpdateOperation);
} // namespace AsdOps