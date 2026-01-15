/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
 
namespace AtbOps {
const int32_t Q_HEADDIM = 576;
const int32_t SPLIT_SIZE_ONE = 512;
const int32_t SPLIT_SIZE_TWO = 64;
using namespace Mki;
class MlaPreprocessOperation : public OperationBase {
    static constexpr int64_t NUM_INPUT_TENSORS = 25;
public:
    explicit MlaPreprocessOperation(const std::string &opName) noexcept : OperationBase(opName) {}
 
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MlaPreprocess), "Invalid Op Param.", return nullptr);
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        auto param = AnyCast<OpParam::MlaPreprocess>(launchParam.GetParam());
        bool deqOnTheFly = inDtype == TENSOR_DTYPE_FLOAT16 &&
                        param.quantMode == OpParam::MlaPreprocess::QuantMode::PER_TENSOR_ASYMM_QUANT;
        if (deqOnTheFly) {
            return GetKernelByName("MLAPreprocessKernel");
        }
        return GetKernelByName("MLAPreprocessBF16Kernel");
    }
 
    int64_t GetInputNum(const Any &specificParam) const override
    {
        return NUM_INPUT_TENSORS;
    }
    
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        return DIM_4;
    }
 
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorKEY = launchParam.GetInTensor(16);
        auto param = AnyCast<OpParam::MlaPreprocess>(launchParam.GetParam());
        if (param.cacheMode == 0) {
            outTensors[DIM_0].desc = tensorQ.desc;
            outTensors[DIM_0].desc.dims = {param.N, param.headNum, Q_HEADDIM};
            outTensors[DIM_1].desc = tensorKEY.desc;
        } else {
            outTensors[DIM_0].desc = tensorQ.desc;
            outTensors[DIM_0].desc.dims = {param.N, param.headNum, SPLIT_SIZE_ONE};
            outTensors[DIM_1].desc = tensorKEY.desc;
            outTensors[DIM_1].desc.dims[DIM_3] = SPLIT_SIZE_ONE;
            outTensors[DIM_2].desc = tensorQ.desc;
            outTensors[DIM_2].desc.dims = {param.N, param.headNum, SPLIT_SIZE_TWO};
            outTensors[DIM_3].desc = tensorKEY.desc;
            outTensors[DIM_3].desc.dims[DIM_3] = SPLIT_SIZE_TWO;
        }
        return Mki::Status::OkStatus();
    }
};
 
REG_OPERATION(MlaPreprocessOperation);
} //    namespace AtbOps