/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include <string>
#include <functional>
#include <map>
#include <cpp-stub/src/stub.h>
#include <gtest/gtest.h>
#include "atb/runner/mki_node_implement.h"
#include "asdops/params/elewise.h"
#include "asdops/ops.h"
#include "mki/base/kernel_base.h"


class TestATBMKIErrorCode : public ::testing::Test {
protected:
    Mki::LaunchParam launchParam;
    Mki::TensorDesc desc;
    int64_t SIZE = 2;
    int64_t DATASIZE = 8;
    float SCALE = 1.0;
    int OFFSET = 0;
    uint8_t *addr = NULL;
    AsdOps::OpParam::Elewise opParam;
    Mki::Tensor A;
    Mki::Tensor B;
    void SetUp() override
    {
        desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
        desc.format = Mki::TensorFormat::TENSOR_FORMAT_ND;
        desc.dims.push_back(SIZE);
        desc.dims.push_back(SIZE);
        opParam.inputScale = SCALE;
        opParam.inputOffset = OFFSET;

        opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_ADD;
        launchParam.GetParam() = opParam;
        A.dataSize = DATASIZE;
        B.dataSize = DATASIZE;

        A.desc = desc;
        B.desc = desc;
        launchParam.AddInTensor(A);
        launchParam.AddInTensor(B);
        launchParam.AddOutTensor(B);
    }

    void TearDown() override {}
};

TEST_F(TestATBMKIErrorCode, TestHash)
{
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::NO_ERROR)->second, atb::ErrorType::NO_ERROR);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_INVALID_VALUE)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_OPERATION_NOT_EXIST)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_TACTIC_NOT_EXIST)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_KERNEL_NOT_EXIST)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_ATTR_NOT_EXIST)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_ATTR_INVALID_TYPE)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_LAUNCH_KERNEL_ERROR)->second,
              atb::ErrorType::ERROR_RT_FAIL);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_SYNC_STREAM_ERROR)->second,
              atb::ErrorType::ERROR_RT_FAIL);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_INFERSHAPE_ERROR)->second, atb::ErrorType::ERROR_RT_FAIL);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_NOT_CONSISTANT)->second,
              atb::ErrorType::ERROR_INVALID_PARAM);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_ALLOC_HOST)->second,
              atb::ErrorType::ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_MEMERY_COPY_ERROR)->second,
              atb::ErrorType::ERROR_COPY_HOST_MEMORY_FAIL);
    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType::ERROR_RUN_TIME_ERROR)->second, atb::ErrorType::ERROR_RT_FAIL);
}

errno_t memset_s_stub(void *dest, size_t destMax, int c, size_t count)
{
    return ERANGE;
}

TEST_F(TestATBMKIErrorCode, TestNewErrorCode)
{
    Mki::Operation *op = AsdOps::Ops::Instance().GetOperationByName("ElewiseOperation");
    Mki::Kernel *kernel = op->GetBestKernel(launchParam);

    Mki::Status status = kernel->Init(launchParam);
    Stub stub;
    stub.set(memset_s, memset_s_stub);
    Mki::RunInfo runInfo;
    status = kernel->Run(launchParam, runInfo);

    EXPECT_EQ(atb::ATB_MKI_ERROR_HASH.find(static_cast<Mki::ErrorType>(status.Code()))->second,
              atb::ErrorType::ERROR_COPY_HOST_MEMORY_FAIL);
}