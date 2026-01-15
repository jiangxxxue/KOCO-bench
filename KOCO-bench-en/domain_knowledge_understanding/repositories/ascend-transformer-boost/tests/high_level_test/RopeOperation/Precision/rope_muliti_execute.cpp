#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/types.h"
#include <atb/atb_infer.h>
#include "demo_util.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"

using namespace atb;

const aclDataType dtype = ACL_FLOAT16;
const aclDataType dtypeInt32 = ACL_INT32;
const size_t dimNum = 4;
const size_t dimNum2 = 4;
const size_t dimNum4 = 4;
const size_t dimX = 16;
const size_t dimY = 32;
const size_t dimZ = 16;
const size_t dimQ = 16;
const size_t dimK = 2;
const size_t inTensorNum = 5;
const size_t outTensorNum = 1;
const size_t counter= 100;
void CreateRopeOperation( atb::Operation **operation)
{
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 4; // 4: 设置旋转系数
    CreateOperation(ropeParam, operation);
}

void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs)
{
    size_t i = 0;
    for (; i < 2; i++) {
        intensorDescs.at(i).dtype = dtype;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = dimNum;
        intensorDescs.at(i).shape.dims[0] = dimX;
        intensorDescs.at(i).shape.dims[1] = dimY;
        intensorDescs.at(i).shape.dims[2] = dimZ;
        intensorDescs.at(i).shape.dims[3] = dimQ;
    }
    for (; i < 4; i++) {
        intensorDescs.at(i).dtype = dtype;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = dimNum2;
        intensorDescs.at(i).shape.dims[0] = dimX;
        intensorDescs.at(i).shape.dims[1] = dimY;
    }
    

    intensorDescs.at(4).dtype = dtypeInt32;
    intensorDescs.at(4).format = ACL_FORMAT_ND;
    intensorDescs.at(4).shape.dimNum = dimNum4;
    intensorDescs.at(4).shape.dims[0] = dimK;
}

// 设置各个intensor并且为各个intensor分配内存空间，此处的intensor为手动设置，工程实现上可以使用torchTensor转换或者其他简单数据结构转换的方式
void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs, uint32_t value)
{
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        std::vector<uint16_t> hostData(atb::Utils::GetTensorNumel(inTensors.at(i)), value);   // 一段全2的hostBuffer
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        // ASSERT_EQ(ret, 0);
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, hostData.data(), hostData.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
        // ASSERT_EQ(ret, 0);
    }
}

// 设置各个outtensor并且为outtensor分配内存空间，同intensor设置
void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        // ASSERT_EQ(ret, 0);
    }
}

char *ReadFile( const std::string& filePath, size_t &fileSize, void *buffer, size_t bufferSize)
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

void WriteFile( const std::string& filePath, void *buffer, const size_t &bufferSize)
{
    std::ofstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        return ;
    }
    file.write(reinterpret_cast<const char*>(buffer), bufferSize);
    return ;
}

TEST(CreateMatMulAddGraphOperation, TestRopeMultiExecute)
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
    CreateRopeOperation(&operation);

    // 准备输入输出tensor
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
    size_t inTensorNum = operation->GetInputNum();
    size_t outTensorNum = operation->GetOutputNum();

    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    CreateInTensorDescs(intensorDescs);


    outtensorDescs.resize(outTensorNum);

    pack.outTensors.resize(outTensorNum);
    operation->InferShape(intensorDescs, outtensorDescs);
    CreateInTensors(pack.inTensors, intensorDescs, 2);
    CreateOutTensors(pack.outTensors, outtensorDescs);

    // Setup
    uint64_t workspaceSize = 0;

    operation->Setup(pack, workspaceSize, context);
    uint8_t *workSpace = nullptr;
    int ret1 = 0;
    if (workspaceSize != 0) {
        ret1 = aclrtMalloc((void **)(&workSpace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
    operation->Execute(pack, workSpace, workspaceSize, context);

    ret1 = aclrtSynchronizeStream(stream); // 流同步，等待device侧任务计算完成
    
    size_t outCounter = pack.outTensors.size();
    std::vector<void *> output0(outCounter, nullptr);
    std::vector<void *> output1(outCounter, nullptr);
    size_t outSize =  atb::Utils::GetTensorSize(pack.outTensors.at(0));
    for (size_t i = 0; i < outCounter; i++) {
        aclrtMallocHost(&output0[i], outSize);
        aclrtMallocHost(&output1[i], outSize);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
           aclrtMemcpy(output0[i], outSize, pack.outTensors.at(i).deviceData, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    for (size_t i = 0; i < counter; i++) {
       operation->Execute(pack, workSpace, workspaceSize, context);
       ret1 = aclrtSynchronizeStream(stream); // 流同步，等待device侧任务计算完成
       for (size_t j = 0; j < pack.outTensors.size(); j++) {
           aclrtMemcpy(output1[j], outSize, pack.outTensors.at(j).deviceData, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
        //    WriteFile("rope_out_multi_" + std::to_string(j) + ".bin", output0, outSize);
    }
    }
    for (size_t i = 0; i < outCounter; i++) {
        for (size_t j = 0; j < outSize / sizeof(char); j++) {
            EXPECT_EQ(((char*)output0[i])[j], ((char*)output1[i])[j]);
    }
    }
    for (size_t i = 0; i < outCounter; i++) {
        aclrtFreeHost(output0[i]);
        aclrtFreeHost(output1[i]);
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
    return 0;
}