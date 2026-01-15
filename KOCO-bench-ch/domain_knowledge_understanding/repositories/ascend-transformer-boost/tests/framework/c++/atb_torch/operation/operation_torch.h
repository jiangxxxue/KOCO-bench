/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OPERATION_TORCH_H
#define OPERATION_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "atb/operation.h"
#include "hosttensor_binders/hosttensor_binder.h"

class OperationTorch : public torch::CustomClassHolder {
public:
    explicit OperationTorch(std::string opName);
    ~OperationTorch();
    void SetName(std::string name);
    std::string SetParam(std::string param);
    std::string UpdateParam(std::string param);
    std::string GetParam(std::string param);
    void SetVaraintPackParam(std::string varaintPackParam);
    std::string InferShape(std::vector<torch::Tensor> atInTensors);
    std::string Setup(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors);
    std::string ExecuteSync(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
        int64_t workspaceSize);
    std::vector<torch::Tensor> ExecuteWithParam(std::vector<torch::Tensor> atInTensors, std::string varaintPackParam);
    void ExecuteOutWithParam(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                             std::string varaintPackParam);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors);
    std::vector<torch::Tensor> ExecuteSplitThread(std::vector<torch::Tensor> atInTensors);
    void ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors);
    void SetTensorList(std::vector<torch::Tensor> atInTensors, std::string tensorListName);
    c10::intrusive_ptr<OperationTorch> clone() const { return c10::make_intrusive<OperationTorch>(opName_); }

private:
    void CreateAtOutTensors(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void ExecuteOutImpl(std::vector<torch::Tensor> &inTensors, std::vector<torch::Tensor> &outTensor);
    void BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    std::string GetSaveTensorDir();
    atb::Status InferShapeOutTensorDesc(std::vector<torch::Tensor> &atInTensors,
                                        atb::SVector<atb::TensorDesc> &outTensorDescs);
    bool IsCopyStreamValid();

private:
    std::string opName_;
    uint64_t opId_ = 0;
    std::string nodeId_ = "0";
    std::string name_;
    std::string param_;
    std::unique_ptr<atb::Operation> operation_;
    std::unique_ptr<HostTensorBinder> hostTensorBinder_;
    uint64_t executeCount_ = 0;
    atb::VariantPack variantPack_;
};

#endif