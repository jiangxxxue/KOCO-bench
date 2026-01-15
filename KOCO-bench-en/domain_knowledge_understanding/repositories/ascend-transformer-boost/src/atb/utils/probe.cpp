/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/utils/probe.h"
#include <sstream>
#include <mki/types.h>
#include "atb/utils/log.h"
#include "atb/types.h"
#include "atb/utils/tensor_util.h"

namespace atb {
// 具体实现由ait工具进行打桩实现

bool Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &opType)
{
    if (ids.size() > 0 && Mki::LogCore::Instance().GetLogLevel() <= Mki::LogLevel::DEBUG) {
        std::stringstream ss;
        ss << "save ids:";
        for (size_t i = 0; i < ids.size(); i++) {
            ss << ids.at(i) << "_";
        }
        ss << " opType:" << opType;
        ATB_LOG(DEBUG) << ss.str();
    }
    return false;
}

bool Probe::IsSaveTensorData()
{
    return false;
}

bool Probe::IsSaveTensorDesc()
{
    return false;
}

bool Probe::IsExecuteCountInRange(const uint64_t executeCount)
{
    ATB_LOG(DEBUG) << "executeCount:" << executeCount;
    return false;
}

bool Probe::IsSaveTensorBefore()
{
    return false;
}

bool Probe::IsSaveTensorAfter()
{
    return false;
}

void Probe::SaveTensor(const std::string &format, const std::string &dtype, const std::string &dims,
                       const void *hostData, uint64_t dataSize, const std::string &filePath)
{
    if (hostData != nullptr) {
        ATB_LOG(DEBUG) << "format:" << format << " dtype:" << dtype << " dims:" << dims
                       << " dataSize:" << std::to_string(dataSize) << " filePath:" << filePath;
    }
}

bool Probe::IsSaveTiling()
{
    return false;
}

void Probe::SaveTiling(const uint8_t *tilingBuffer, uint64_t tilingBufferSize, const std::string &filePath)
{
    if (tilingBuffer != nullptr) {
        ATB_LOG(DEBUG) << "tilingBufferSize:" << tilingBufferSize << " filePath:" << filePath;
    }
}

bool Probe::ReportOperationIOTensorEnable()
{
    return false;
}

bool Probe::ReportOperationStatisticEnable()
{
    return false;
}

void Probe::ReportOperationSetupStatistic(const size_t executeCount, const std::string &opName, const std::string &st)
{
    ATB_LOG(DEBUG) << " executeCount: " << executeCount << "opName: " << opName << " statistics: " << st;
}

void Probe::ReportOperationExecuteStatistic(const size_t executeCount, const std::string &opName, const std::string &st)
{
    ATB_LOG(DEBUG) << " executeCount: " << executeCount << "opName: " << opName << " statistics: " << st;
}


bool Probe::ReportKernelIOTensorEnable()
{
    return false;
}

void Probe::ReportOperationIOTensor(const size_t executeCount, const std::string &opName, const std::string &opParam,
                                    const std::vector<Probe::Tensor> &inTensors,
                                    const std::vector<Probe::Tensor> &outTensors)
{
    ATB_LOG(DEBUG) << "Operation IO: " << opName << " executeCount: " << executeCount << " param: " << opParam
                   << " inTensorNum: " << inTensors.size() << " outTensorNum: " << outTensors.size();
}

bool Probe::ReportOperationGraphEnable()
{
    return false;
}

void Probe::ReportKernelIOTensor(const size_t executeCount, const std::string &opName, const std::string &opParam,
                                 const std::vector<Probe::Tensor> &inTensors,
                                 const std::vector<Probe::Tensor> &outTensors)
{
    ATB_LOG(DEBUG) << "kernel: " << opName << " executeCount: " << executeCount << " param: " << opParam
                   << " inTensorNum: " << inTensors.size() << " outTensorNum: " << outTensors.size();
}

void Probe::AsdopsTensorToProbeTensor(const Mki::Tensor &asdopsTensor, Probe::Tensor &probeTensor,
                                      const std::string &filePath)
{
    probeTensor.dtype = Mki::GetStrWithDType(asdopsTensor.desc.dtype);
    probeTensor.format = Mki::GetStrWithFormat(asdopsTensor.desc.format);
    probeTensor.shape = TensorUtil::AsdOpsDimsToString(asdopsTensor.desc.dims);
    probeTensor.filePath = filePath;
}

void Probe::AtbTensorToProbeTensor(const atb::Tensor &atbTensor, Probe::Tensor &probeTensor,
                                   const std::string &filePath)
{
    probeTensor.dtype = Mki::GetStrWithDType(atbTensor.desc.dtype);
    probeTensor.format = Mki::GetStrWithFormat(atbTensor.desc.format);
    probeTensor.shape = TensorUtil::ShapeToString(atbTensor.desc.shape);
    probeTensor.filePath = filePath;
}

void Probe::ReportOperationGraph(const std::string &opName, const std::string &graph)
{
    ATB_LOG(DEBUG) << "opName:" << opName << " graph json string:" << graph;
}

void Probe::SaveParam(const std::string &param, const std::string &filePath)
{
    (void)param;
    (void)filePath;
}

bool Probe::IsSaveParam()
{
    return false;
}

bool Probe::IsOverflowCheck()
{
    return false;
}

bool Probe::IsOverflowStop()
{
    return false;
}

void Probe::ReportOverflowKernel(const std::string &kernelPath)
{
    (void)kernelPath;
}
} // namespace atb