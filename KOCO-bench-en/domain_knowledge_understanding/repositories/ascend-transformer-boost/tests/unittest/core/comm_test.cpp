/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/comm.h"
#include <gtest/gtest.h>
#include <iostream>

namespace atb {
void TestCreateHcclComm(int rank, int rankRoot, int rankSize)
{
    aclrtSetDevice(rank);
    char commName[128] = "";
    HcclComm hcclComm = nullptr;
    hcclComm = atb::Comm::CreateHcclComm(rank, rankRoot, rankSize, commName);
    std::cout << "commName: " << commName << std::endl;
    EXPECT_NE(hcclComm, nullptr) << "HcclComm should not be null";
    EXPECT_NE(commName, "") << "CommName should not be null";
    auto ret = atb::Comm::DestoryHcclComm(hcclComm);
    aclrtResetDevice(rank);
    std::cout << "destoryHcclComm  ret: " << ret << std::endl;
}

void TestCreateHcclCommByRankFile(int rank, int rankSize)
{
    aclrtSetDevice(rank);
    char commName[128] = "";
    char *rankTableFile = "/home/zmq/ranktableFile.json";
    HcclComm hcclComm = nullptr;
    hcclComm = atb::Comm::CreateHcclCommByRankTableFile(rank, rankSize, rankTableFile, commName);
    std::cout << "rankTableFile commName: " << commName << std::endl;
    EXPECT_NE(hcclComm, nullptr) << "rankTableFile HcclComm should not be null";
    EXPECT_NE(commName, "") << "rankTableFile CommName should not be null";
    auto ret = atb::Comm::DestoryHcclComm(hcclComm);
    aclrtResetDevice(rank);
    std::cout << "destoryHcclComm  ret: " << ret << std::endl;
}

void TestCreateHcclCrossMulitComm(int realRank, int subCommRankId, std::vector<uint32_t> rankIds, uint64_t subCommId)
{
    std::cout << "realRank: " << realRank << std::endl;
    aclrtSetDevice(realRank);
    char *rankTableFile = "/data/ranktableFile.json";
    uint32_t hcclBufferSize = 200;
    char commName[129] = "";
    HcclComm hcclComm = nullptr;
    hcclComm = atb::Comm::CreateHcclCrossMulitComm(rankTableFile, subCommRankId, rankIds, subCommId, hcclBufferSize, commName);
    std::cout << "commName: " << commName << std::endl;
    EXPECT_NE(hcclComm, nullptr) << "HcclComm should not be null";
    EXPECT_NE(commName, "") << "CommName should not be null";
    auto ret = atb::Comm::DestoryHcclComm(hcclComm);
    aclrtResetDevice(realRank);
    std::cout << "destoryHcclComm  ret: " << ret << std::endl;
}
} // namespace atb

void RunTestInProcess(int rank, int rankSize)
{
    int rankRoot = 0;
    atb::TestCreateHcclComm(rank, rankRoot, rankSize);
    // atb::TestCreateHcclCommByRankFile(rank, rankSize);
}

void RunCreateHcclCrossMulitComm()
{
    // 创建两个子通信域
    const int processCount = 4;
    pid_t pids[processCount];

    std::vector<uint32_t> rankIds1 = {0, 1};
    int subCommId1 = 1;
    pids[0] = fork();
    if (pids[0] < 0) {
        std::cout << "fork failed ! " << pids[0] << std::endl;
    } else if (pids[0] == 0) {
        atb::TestCreateHcclCrossMulitComm(0, 0, rankIds1, subCommId1);
        exit(0);
    }
    pids[1] = fork();
    if (pids[1] < 0) {
        std::cout << "fork failed ! " << pids[1] << std::endl;
    } else if (pids[1] == 0) {
        atb::TestCreateHcclCrossMulitComm(1, 1, rankIds1, subCommId1);
        exit(0);
    }

    std::vector<uint32_t> rankIds2 = {2, 3};
    int subCommId2 = 2;
    pids[2] = fork();
    if (pids[2] < 0) {
        std::cout << "fork failed ! " << pids[2] << std::endl;
    } else if (pids[2] == 0) {
        atb::TestCreateHcclCrossMulitComm(2, 0, rankIds2, subCommId2);
        exit(0);
    }
    pids[3] = fork();
    if (pids[3] < 0) {
        std::cout << "fork failed ! " << pids[3] << std::endl;
    } else if (pids[3] == 0) {
        atb::TestCreateHcclCrossMulitComm(3, 1, rankIds2, subCommId2);
        exit(0);
    }

    // 父进程等待所有子进程完成
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], NULL, 0);
    }
}

TEST(CreateCommTest, CreateHcclCommMultiProcess)
{
    // 所需的进程数量
    const int processCount = 2;
    pid_t pids[processCount];

    for (int i = 0; i < processCount; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            // 子进程，完成任务后退出
            RunTestInProcess(i, processCount);
            exit(0);
        }
    }

    // 父进程等待所有子进程完成
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], NULL, 0);
    }
}

// TEST(CreateCommTest, CreateHcclCrossMulitComm)
// {
//     RunCreateHcclCrossMulitComm1();
// }
