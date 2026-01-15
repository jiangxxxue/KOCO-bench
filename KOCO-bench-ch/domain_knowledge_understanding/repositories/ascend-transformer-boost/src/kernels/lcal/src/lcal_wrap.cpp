/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <lcal_api.h>
#include <vector>
#include <thread>
#include <atomic>
#include "mki/utils/log/log.h"
#include "lcal.h"
#include "tools/socket/lcal_sock_exchange.h"

using namespace std;
using namespace Lcal;

int LcalCommInitRankLocal(int rankSize, int rank, LcalCommPtr *comm)
{
    MKI_LOG(INFO) << "using lcal c++ api! rank" << rank;
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm ptr is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    auto *c = new (std::nothrow) LcalComm(rank, rankSize);
    if (c == nullptr) {
        MKI_LOG(ERROR) << "LcalComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return LCAL_ERROR_INTERNAL;
    }
    *comm = c;
    int ret = c->Init();
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "lccl init failed!";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalGetUniqueId(LcalUniqueId *uniqueId, int commDomain)
{
    if (uniqueId == nullptr) {
        MKI_LOG(ERROR) << "uniqueId is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    int res = BootstrapGetUniqueId(reinterpret_cast<struct LcalBootstrapHandle &>(uniqueId), commDomain);
    if (res != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "lcal BootstrapGetUniqueId failed!";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalCommInitRank(LcalUniqueId commId, int rankSize, int rank, LcalCommPtr *comm)
{
    MKI_LOG(INFO) << "using lcal c++ api! rank" << rank;
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm ptr is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    auto *c = new (std::nothrow) LcalComm(rank, rankSize, commId);
    if (c == nullptr) {
        MKI_LOG(ERROR) << "LcalComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return LCAL_ERROR_INTERNAL;
    }
    *comm = c;
    int ret = c->Init();
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "lccl init failed!";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, LcalCommPtr *comm,
    const bool isEnableAutoMagicNum)
{
    MKI_LOG(INFO) << "using lcal c++ api! rank : " << rank << ", rankSize : " << rankSize << ", commDomain:" <<
        commDomain << ", bufferSize:" << bufferSize << ", isEnableAutoMagicNum:" << isEnableAutoMagicNum;
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm ptr is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }

    constexpr int minBufferSize = LCAL_COMM_BUFFER_SIZE;
    if (bufferSize < minBufferSize) {
        MKI_LOG(ERROR) << "lcal comm buffer size " << bufferSize << " MBytes should not be less than " <<
            minBufferSize << " MBytes!";
        return LCAL_ERROR_INTERNAL;
    }

    auto *c = new (std::nothrow) LcalComm(rank, rankSize, commDomain, bufferSize, isEnableAutoMagicNum);
    if (c == nullptr) {
        MKI_LOG(ERROR) << "LcalComm create failed. rank : " << rank << ", rankSize : " << rankSize << ", commDomain:" <<
            commDomain << ", bufferSize:" << bufferSize << ", isEnableAutoMagicNum:" << isEnableAutoMagicNum;
        return LCAL_ERROR_INTERNAL;
    }
    *comm = c;
    int ret = c->Init();
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "lccl init failed!";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalCommInitRankWithDomain(int commDomain, int rankSize, int rank, LcalCommPtr *comm)
{
    constexpr int minBufferSize = LCAL_COMM_BUFFER_SIZE;
    return LcalCommInitRankWithCustDomainSize(commDomain, minBufferSize, rankSize, rank, comm);
}

int LcalGetCommArgsDev(LcalCommPtr comm, GM_ADDR &commArgsPtr)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    auto *lccl = static_cast<LcalComm *>(comm);
    commArgsPtr = lccl->GetCommArgsPtr();
    return LCAL_SUCCESS;
}

int LcalGetCommArgsHost(LcalCommPtr comm, Lcal::CommArgs *&commArgsPtr)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    auto *c = static_cast<LcalComm *>(comm);
    commArgsPtr = c->GetCommArgs();
    return LCAL_SUCCESS;
}

void LcalPrintDFX2Log(LcalCommPtr comm)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm is nullptr!";
        return;
    }
    auto *lcal = static_cast<LcalComm *>(comm);
    MKI_LOG(INFO) << lcal->PrintDFX();
}

int LcalCommInit(int rank, int rankSize, LcalCommPtr *comms)
{
    if (comms == nullptr) {
        MKI_LOG(ERROR) << "lcal comms is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    *comms = new (std::nothrow) LcalComm(rank, rankSize);
    if (*comms == nullptr) {
        MKI_LOG(ERROR) << "LcalComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalCommInitAll(uint32_t ndev, int32_t *devices, LcalCommPtr *comms)
{
    if (comms == nullptr) {
        MKI_LOG(ERROR) << "lcal comms is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    if (devices == nullptr) {
        MKI_LOG(ERROR) << "lcal devices is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    static int commDomain = 0;
    commDomain++;
    for (uint32_t i = 0; i < ndev; ++i) {
        comms[i] = new (std::nothrow) LcalComm(i, ndev, commDomain, LCAL_COMM_BUFFER_SIZE, false);
        if (comms[i] == nullptr) {
            MKI_LOG(ERROR) << "LcalComm create failed. dev : " << i << ", ndev : " << ndev;
            return LCAL_ERROR_INTERNAL;
        }
    }
    static atomic<int> uid;
    uid++;
    vector<unique_ptr<thread>> threads;
    int error = LCAL_SUCCESS;
    for (uint32_t r = 0; r < ndev; r++) {
        threads.emplace_back(make_unique<thread>(
            [&](int rank) {
                aclrtSetDevice(devices[rank]);
                auto *c = static_cast<LcalComm *>(comms[rank]);
                int ret = c->InitThread("uid" + to_string(uid));
                if (ret != LCAL_SUCCESS) {
                    error = ret;
                }
            },
            r));
    }
    for (auto &t : threads) {
        t->join();
    }
    threads.clear();
    return error;
}

int LcalCommInitThread(int rank, int rankSize, const char *uid, LcalCommPtr *comms)
{
    if (uid == nullptr) {
        MKI_LOG(ERROR) << "lcal uid is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    if (comms == nullptr) {
        MKI_LOG(ERROR) << "lcal comms is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    if (rank >= rankSize) {
        MKI_LOG(ERROR) << "lcal rank : " << rank << " rankSize : " << rankSize;
        return LCAL_ERROR_INTERNAL;
    }
    *comms = new (std::nothrow) LcalComm(rank, rankSize);
    if (*comms == nullptr) {
        MKI_LOG(ERROR) << "LcalComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return LCAL_ERROR_INTERNAL;
    }
    auto *c = static_cast<LcalComm *>(*comms);
    return c->InitThread(string(uid));
}

int LcclAllReduce(void *sendBuf, void *recvBuf, int64_t count, HcclDataType dataType, HcclReduceOp op,
                  LcalCommPtr comm, aclrtStream stream)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "LcclAllReduce comm is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    Lccl lccl(static_cast<LcalComm *>(comm));
    return lccl.AllReduce(sendBuf, recvBuf, count, dataType, op, stream);
}

int LcclAllGather(void *sendBuf, void *recvBuf, int64_t sendCount, HcclDataType dataType, LcalCommPtr comm,
                  aclrtStream stream)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "LcclAllGather comm is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    Lccl lccl(static_cast<LcalComm *>(comm));
    return lccl.AllGather(sendBuf, recvBuf, sendCount, dataType, stream);
}

int LcclReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                      LcalCommPtr comm, aclrtStream stream)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "LcclReduceScatter comm is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    Lccl lccl(static_cast<LcalComm *>(comm));
    return lccl.ReduceScatter(sendBuf, recvBuf, recvCount, dataType, op, stream);
}

int LcclBroadcast(void *buf, int64_t count, HcclDataType dataType, int root, LcalCommPtr comm,
                  aclrtStream stream)
{
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "LcclBroadcast comm is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }
    Lccl lccl(static_cast<LcalComm *>(comm));
    return lccl.Broadcast(buf, count, dataType, root, stream);
}

int LcclCommDestroy(LcalCommPtr comm)
{
    if (comm == nullptr) {
        return LCAL_INVALID_VALUE;
    }
    auto *c = static_cast<LcalComm *>(comm);
    delete c;
    return LCAL_SUCCESS;
}