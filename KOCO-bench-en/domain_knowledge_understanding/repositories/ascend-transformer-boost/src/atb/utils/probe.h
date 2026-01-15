/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_PROBE_H
#define ATB_PROBE_H

#include <vector>
#include <string>
#include <mki/tensor.h>
#include <atb/types.h>

namespace atb {
class Probe {
public:
    struct Tensor {
        std::string dtype;
        std::string format;
        std::string shape;
        std::string filePath;
    };

public:
    static bool IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &opType);
    static bool IsSaveTensorData();
    static bool IsSaveTensorDesc();
    static bool IsExecuteCountInRange(const uint64_t executeCount);
    static bool IsSaveTensorBefore();
    static bool IsSaveTensorAfter();
    static void SaveTensor(const std::string &format, const std::string &dtype, const std::string &dims,
                           const void *hostData, uint64_t dataSize, const std::string &filePath);
    static bool IsSaveTiling();
    static void SaveTiling(const uint8_t *tilingBuffer, uint64_t tilingBufferSize, const std::string &filePath);
    static void SaveParam(const std::string &param, const std::string &filePath);
    static bool IsSaveParam();
    static bool ReportOperationIOTensorEnable();
    static void ReportOperationIOTensor(const size_t executeCount, const std::string &opName,
                                        const std::string &opParam, const std::vector<Probe::Tensor> &inTensors,
                                        const std::vector<Probe::Tensor> &outTensors);
    static bool ReportOperationStatisticEnable();
    static void ReportOperationSetupStatistic(const size_t executeCount, const std::string &opName,
                                              const std::string &st);
    static void ReportOperationExecuteStatistic(const size_t executeCount, const std::string &opName,
                                                const std::string &st);
    static bool ReportKernelIOTensorEnable();
    static void ReportKernelIOTensor(const size_t executeCount, const std::string &opName, const std::string &opParam,
                                     const std::vector<Probe::Tensor> &inTensors,
                                     const std::vector<Probe::Tensor> &outTensors);
    static void AsdopsTensorToProbeTensor(const Mki::Tensor &asdopsTensor, Probe::Tensor &probeTensor,
                                          const std::string &filePath);
    static void AtbTensorToProbeTensor(const atb::Tensor &atbTensor, Probe::Tensor &probeTensor,
                                       const std::string &filePath);
    static bool IsOverflowCheck();
    static bool IsOverflowStop();
    static void ReportOverflowKernel(const std::string &kernelPath);

public:
    static bool ReportOperationGraphEnable();
    static void ReportOperationGraph(const std::string &opName, const std::string &graph);
};
} // namespace atb
#endif
