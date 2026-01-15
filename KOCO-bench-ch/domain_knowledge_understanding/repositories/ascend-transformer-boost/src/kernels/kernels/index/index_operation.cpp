/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"

namespace AsdOps {
using namespace Mki;
class IndexOperation : public OperationBase {
public:
    explicit IndexOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        auto inDtype0 = launchParam.GetInTensor(0).desc.dtype;
        auto inDtype2 = launchParam.GetInTensor(2).desc.dtype;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Index), "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::Index>(launchParam.GetParam());
        switch (param.indexType) {
            case OpParam::Index::INDEX_ADD:
                switch (inDtype0) {
                    case TENSOR_DTYPE_FLOAT16: return GetKernelByName("IndexAddF16Kernel");
                    case TENSOR_DTYPE_BF16: return GetKernelByName("IndexAddBF16Kernel");
                    default:
                        MKI_LOG(ERROR) << "No supported INDEX_ADD kernel for dtype " << inDtype0;
                        return nullptr;
                }
            case OpParam::Index::INDEX_ADD_VALID:
                switch (inDtype2) {
                    case TENSOR_DTYPE_FLOAT16:
                    case TENSOR_DTYPE_BF16:
                        return GetKernelByName("IndexAddValidF16Kernel");
                    default:
                        MKI_LOG(ERROR) << "No supported INDEX_ADD_VALID kernel for dtype " << inDtype2;
                        return nullptr;
                }
            default:
                MKI_LOG(ERROR) << "Unsupported index tatic type INDEX_UNDEFINED";
                return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Index), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Index>(specificParam);
        switch (param.indexType) {
            case OpParam::Index::INDEX_ADD:
            case OpParam::Index::INDEX_ADD_VALID:
                return 4;   // index_add和index_add_valid均为4个输入tensor
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
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Index),
                  "no match param type", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::Index>(launchParam.GetParam());

        switch (param.indexType) {
            case OpParam::Index::INDEX_ADD: {
                outTensors[0].desc = launchParam.GetInTensor(0).desc;
                return Status::OkStatus();
            }
            case OpParam::Index::INDEX_ADD_VALID: {
                outTensors[0].desc = launchParam.GetInTensor(0).desc; // inTensor(3)对应var, 是原地做加法的零张量
                return Status::OkStatus();
            }
            default: return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported index tatic type");
        }
    }
};
REG_OPERATION(IndexOperation);
} // namespace AsdOps