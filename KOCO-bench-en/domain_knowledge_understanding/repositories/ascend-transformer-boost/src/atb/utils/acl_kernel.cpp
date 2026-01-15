/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/acl_kernel.h"
#include "acl/acl_op_compiler.h"
#include "atb/utils/log.h"

namespace atb {
AclKernel::AclKernel() {}

AclKernel::~AclKernel()
{
    try {
        Reset();
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << "An exception occurred when running Reset()." << "Traceback: \n" << e.what();
    }
}

aclError AclKernel::Run(const std::string &kernelName, const SVector<Tensor> &inTensors,
                        const SVector<Tensor> &outTensors, aclrtStream stream)
{
    Status st = Reset();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "AclKernel Reset failed! ret:" << st;
        return st;
    }
    opType_ = kernelName;
    stream_ = stream;
    inTensorsDesc_.resize(inTensors.size());
    inTensorsBuffer_.resize(inTensors.size());
    for (size_t i = 0; i < inTensors.size(); i++) {
        const TensorDesc desc = inTensors.at(i).desc;
        inTensorsDesc_.at(i) = aclCreateTensorDesc(desc.dtype, desc.shape.dimNum, desc.shape.dims, desc.format);
        inTensorsBuffer_.at(i) = aclCreateDataBuffer(inTensors.at(i).deviceData, inTensors.at(i).dataSize);
    }

    outTensorsDesc_.resize(outTensors.size());
    outTensorsBuffer_.resize(outTensors.size());
    for (size_t i = 0; i < outTensors.size(); i++) {
        const TensorDesc desc = outTensors.at(i).desc;
        outTensorsDesc_.at(i) = aclCreateTensorDesc(desc.dtype, desc.shape.dimNum, desc.shape.dims, desc.format);
        outTensorsBuffer_.at(i) = aclCreateDataBuffer(outTensors.at(i).deviceData, outTensors.at(i).dataSize);
    }
    st = AclKernelRun();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "AclKernel AclKernelRun failed!";
        return st;
    }
    return NO_ERROR;
}

aclError AclKernel::AclKernelRun()
{
    ATB_LOG(DEBUG) << "opType:" << opType_ << " inTensorNum:" << inTensorsDesc_.size()
                   << " inTensorsDesc:" << inTensorsDesc_.data() << " inTensorsBuffer:" << inTensorsBuffer_.data()
                   << " outTensorNum:" << outTensorsDesc_.size() << " outTensorsDesc:" << outTensorsDesc_.data()
                   << " outTensorsBuffer:" << outTensorsBuffer_.data() << " opAttr:" << opAttr_;

    Status st = aclopCompileAndExecuteV2(
        opType_.c_str(), inTensorsDesc_.size(), inTensorsDesc_.data(), inTensorsBuffer_.data(), outTensorsDesc_.size(),
        outTensorsDesc_.data(), outTensorsBuffer_.data(), opAttr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "aclopCompileAndExecuteV2 failed! ret:" << st;
    }
    return st;
}

Status AclKernel::Reset()
{
    opType_ = "";
    stream_ = nullptr;
    Status st = NO_ERROR;
    if (opAttr_ != nullptr) {
        aclopDestroyAttr(opAttr_);
    }
    opAttr_ = nullptr;

    for (size_t i = 0; i < inTensorsDesc_.size(); i++) {
        if (inTensorsDesc_.at(i) != nullptr) {
            aclDestroyTensorDesc(inTensorsDesc_.at(i));
        }
    }
    inTensorsDesc_.clear();

    for (size_t i = 0; i < inTensorsBuffer_.size(); i++) {
        if (inTensorsBuffer_.at(i) != nullptr) {
            st = aclDestroyDataBuffer(inTensorsBuffer_.at(i));
            if (st != NO_ERROR) {
                ATB_LOG(ERROR) << "aclDestroyDataBuffer failed! ret:" << st;
                return st;
            }
        }
    }
    inTensorsBuffer_.clear();

    for (size_t i = 0; i < outTensorsDesc_.size(); i++) {
        if (outTensorsDesc_.at(i) != nullptr) {
            aclDestroyTensorDesc(outTensorsDesc_.at(i));
        }
    }
    outTensorsDesc_.clear();

    for (size_t i = 0; i < outTensorsBuffer_.size(); i++) {
        if (outTensorsBuffer_.at(i) != nullptr) {
            st = aclDestroyDataBuffer(outTensorsBuffer_.at(i));
            if (st != NO_ERROR) {
                ATB_LOG(ERROR) << "aclDestroyDataBuffer failed! ret:" << st;
                return st;
            }
        }
    }
    outTensorsBuffer_.clear();
    return NO_ERROR;
}
} // namespace atb