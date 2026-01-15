/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include <atb/utils.h>
#include <iostream>
#include <fstream>
#include "atb/operation.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/operation_infra.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;
const size_t dimNum = 2;
const size_t dimX = 2;
const size_t dimY = 2;
const size_t inTensorNum = 2;
const size_t outTensorNum = 1;
const aclDataType dtype = ACL_FLOAT;
const float result= 10.f;
const size_t counter = 100;
const size_t fp16Size = 2;
const size_t fp32Size = sizeof(float);
const size_t intSize = sizeof(uint32_t);

static void CreateAddOperation( atb::Operation **operation)
{
    atb::infer::ElewiseParam addParam1;
    addParam1.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam1, operation);
}

static void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < intensorDescs.size(); i++) {
        intensorDescs.at(i).dtype = dtype;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = dimNum;
        intensorDescs.at(i).shape.dims[0] = dimX;
        intensorDescs.at(i).shape.dims[1] = dimY;
    }
}

static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    std::vector<float> zeroData(4, 5); // 一段全0的hostBuffer
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        auto ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, zeroData.data(), inTensors.at(i).dataSize, ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
    }
}

static char *ReadFile( const std::string& filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }
    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0 || size != bufferSize) {
        return nullptr;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    return static_cast<char *>(buffer);
}

static void WriteFile( const std::string& filePath, void *buffer, const size_t &bufferSize)
{
    std::ofstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        return ;
    }
    file.write(reinterpret_cast<const char*>(buffer), bufferSize);
    return ;
}

static void CreateInTensorsFromFile(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs, const std::string& path1, const std::string& path2)
{
    for (size_t i = 0; i < inTensors.size(); i++) {
        std::string path = (i == 0 ? path1 : path2);
        void *data = nullptr;
        // 
        size_t fileSize = 0;
        size_t dataSizeByte = dimX * dimY * fp32Size;
        aclrtMallocHost(&data, dataSizeByte);
        char *fileData = ReadFile(path, fileSize, data, dataSizeByte);
        if (fileData == nullptr) {
            std::cout <<"readfile failed!" << std::endl;
            return ;
        }
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        auto ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        // EXPECT_EQ(ret, atb::NO_ERROR);
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, data, inTensors.at(i).dataSize, ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
        // EXPECT_EQ(ret, atb::NO_ERROR);
    }
}

static void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
}


TEST(CreateMatMulAddGraphOperation, TestAddMultiExecute)
{
    if (!GetSingleton<Config>().Is910B() || !GetSingleton<Config>().Is310B()) {
        exit(0);  
    }
    uint32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    atb::Context *context = nullptr;
    Status st = atb::CreateContext(&context);
    ATB_LOG_IF(st != 0, ERROR) << "CreateContext fail";
    
    aclrtStream stream = nullptr;
    st = aclrtCreateStream(&stream);
    ATB_LOG_IF(st != 0, ERROR) << "aclrtCreateStream fail";
    context->SetExecuteStream(stream);

    atb::Operation *operation = nullptr;
    CreateAddOperation(&operation);

    // 准备输入输出tensor
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs1;
    atb::SVector<atb::TensorDesc> outtensorDescs1;

    pack.inTensors.resize(inTensorNum);
    pack.outTensors.resize(outTensorNum);
    intensorDescs1.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs1);
    CreateInTensors(pack.inTensors, intensorDescs1);
    // CreateInTensorsFromFile(pack.inTensors, intensorDescs1, "x.bin", "y.bin");

    outtensorDescs1.resize(outTensorNum);
    outtensorDescs1.at(0).dtype = dtype;
    outtensorDescs1.at(0).format = ACL_FORMAT_ND;
    outtensorDescs1.at(0).shape.dimNum = dimNum;
    outtensorDescs1.at(0).shape.dims[0] = dimX;
    outtensorDescs1.at(0).shape.dims[1] = dimY;
    CreateOutTensors(pack.outTensors, outtensorDescs1);

    // Setup
    uint64_t workspaceSize = 0;
    operation->Setup(pack, workspaceSize, context);
    void *workSpace = nullptr;
    int ret1 = 0;
    if (workspaceSize != 0) {
        ret1 = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    //Execute
    for (size_t i = 0; i < counter; i++) {
        st = operation->Execute(pack, (uint8_t *)workSpace, workspaceSize, context);
    }
    
    //流同步
    ret1 = aclrtSynchronizeStream(stream);
    if (ret1 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        size_t outSize =  atb::Utils::GetTensorSize(pack.outTensors.at(i));
        void *output0 = nullptr;
        aclrtMallocHost(&output0, outSize);
        aclrtMemcpy(output0, outSize, pack.outTensors.at(i).deviceData, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t j = 0; j < outSize / sizeof(float); j++) {
            // std::cout << "((float*)output0)[" << j <<"]" << (((float*)output0)[j] == result) << std::endl;
            EXPECT_EQ(((float*)output0)[j], result);
        }
        aclrtFreeHost(output0);
    }
    // 资源释放
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);

    
    aclrtFree(workSpace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    // aclFinalize();
    return 0;
}


// TEST(CreateMatMulAddGraphOperation, TestAddFile)
// {
//     if (!GetSingleton<Config>().Is910B() || !GetSingleton<Config>().Is310B()) {
//         exit(0);  
//     }
//     uint32_t deviceId = 0;
//     aclrtSetDevice(deviceId);
//     atb::Context *context = nullptr;
//     Status st = atb::CreateContext(&context);
//     ATB_LOG_IF(st != 0, ERROR) << "CreateContext fail";
    
//     aclrtStream stream = nullptr;
//     st = aclrtCreateStream(&stream);
//     ATB_LOG_IF(st != 0, ERROR) << "aclrtCreateStream fail";
//     context->SetExecuteStream(stream);

//     atb::Operation *operation = nullptr;
//     CreateAddOperation(&operation);

//     // 准备输入输出tensor
//     atb::VariantPack pack;
//     atb::SVector<atb::TensorDesc> intensorDescs1;
//     atb::SVector<atb::TensorDesc> outtensorDescs1;

//     pack.inTensors.resize(inTensorNum);
//     pack.outTensors.resize(outTensorNum);
//     intensorDescs1.resize(inTensorNum);

//     CreateInTensorDescs(intensorDescs1);
//     // CreateInTensors(pack.inTensors, intensorDescs1);
//     CreateInTensorsFromFile(pack.inTensors, intensorDescs1, "x.bin", "y.bin");

//     outtensorDescs1.resize(outTensorNum);
//     outtensorDescs1.at(0).dtype = dtype;
//     outtensorDescs1.at(0).format = ACL_FORMAT_ND;
//     outtensorDescs1.at(0).shape.dimNum = dimNum;
//     outtensorDescs1.at(0).shape.dims[0] = dimX;
//     outtensorDescs1.at(0).shape.dims[1] = dimY;
//     CreateOutTensors(pack.outTensors, outtensorDescs1);

//     // Setup
//     uint64_t workspaceSize = 0;
//     operation->Setup(pack, workspaceSize, context);
//     void *workSpace = nullptr;
//     int ret1 = 0;
//     if (workspaceSize != 0) {
//         ret1 = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
//         if (ret1 != 0) {
//             std::cout << "alloc error!";
//             exit(0);
//         }
//     }

//     //Execute
//     for (size_t i = 0; i < counter; i++) {
//         st = operation->Execute(pack, (uint8_t *)workSpace, workspaceSize, context);
//         // EXPECT_EQ(st, atb::NO_ERROR);
//     }
    
//     //流同步
//     ret1 = aclrtSynchronizeStream(stream);
//     // EXPECT_EQ(ret1, atb::NO_ERROR);
//     if (ret1 != 0) {
//         std::cout << "sync error!";
//         exit(0);
//     }

//     for (size_t i = 0; i < pack.outTensors.size(); i++) {
//         size_t outSize =  atb::Utils::GetTensorSize(pack.outTensors.at(i));
//         void *output0 = nullptr;
//         aclrtMallocHost(&output0, outSize);
//         aclrtMemcpy(output0, outSize, pack.outTensors.at(i).deviceData, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
//         WriteFile("out.bin", output0, outSize);
//         aclrtFreeHost(output0);
//     }

//     // 资源释放
//     for (size_t i = 0; i < pack.inTensors.size(); i++) {
//         aclrtFree(pack.inTensors.at(i).deviceData);
//     }
//     for (size_t i = 0; i < pack.outTensors.size(); i++) {
//         aclrtFree(pack.outTensors.at(i).deviceData);
//     }
//     atb::DestroyOperation(operation);
//     atb::DestroyContext(context);

    
//     aclrtFree(workSpace);
//     aclrtDestroyStream(stream);
//     aclrtResetDevice(deviceId);
// }
