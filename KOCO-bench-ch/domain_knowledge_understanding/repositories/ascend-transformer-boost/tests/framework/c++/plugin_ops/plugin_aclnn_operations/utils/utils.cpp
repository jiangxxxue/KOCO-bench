/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "utils.h"
#include "atb/utils/log.h"

namespace atb {
void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < intensorDescs.size(); i++)
    {
        intensorDescs.at(i).dtype = ACL_FLOAT16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < inTensors.size(); i++)
    {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        std::vector<uint16_t> hostData(atb::Utils::GetTensorNumel(inTensors.at(i)), 2); // 一段全2的hostBuffer
        int ret = aclrtMalloc(
            &inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        if (ret != 0) {
            ATB_LOG(ERROR) << "alloc error!";
        }

        ret = aclrtMemcpy(inTensors.at(i).deviceData,
                          inTensors.at(i).dataSize,
                          hostData.data(),
                          hostData.size() * sizeof(uint16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE); // 拷贝CPU内存到NPU侧
        if (ret != 0) {
            ATB_LOG(ERROR) << "aclrtMemcpy error!";
        }
    }
}

void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++)
    {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            ATB_LOG(ERROR) << "aclrtMalloc error!";
        }
    }
}

void CreateTensorFromDesc(atb::Tensor &tensor, atb::TensorDesc &tensorDescs)
{
    tensor.desc = tensorDescs;
    tensor.dataSize = atb::Utils::GetTensorSize(tensor);
    int ret = aclrtMalloc(&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0) {
        ATB_LOG(ERROR) << "aclrtMalloc error!";
    }
}

void PrintOutTensorValue(atb::Tensor &outTensor)
{
    // 输出tensor拷贝回host侧并打印
    std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(outTensor));
    int ret = aclrtMemcpy(outBuffer.data(),
                          outBuffer.size() * sizeof(uint16_t),
                          outTensor.deviceData,
                          outTensor.dataSize,
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        ATB_LOG(ERROR) << "copy error!";
    }

    for (size_t i = 0; i < outBuffer.size(); i = i + 1)
    {
        ATB_LOG(INFO) << "out[" << std::to_string(i) << "] = " << std::to_string((uint32_t)outBuffer.at(i));
    }
}
}