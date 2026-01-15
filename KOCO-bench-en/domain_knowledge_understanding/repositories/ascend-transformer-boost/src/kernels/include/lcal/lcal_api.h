/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_API_H
#define LCAL_API_H

#include <string>
#include <hccl.h>
#include <comm_args.h>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *LcalCommPtr;
#define LCAL_UNIQUE_ID_BYTES 128
typedef struct { char internal[LCAL_UNIQUE_ID_BYTES]; } LcalUniqueId;

int LcalGetUniqueId(LcalUniqueId *uniqueId, int commDomain);

int LcalCommInitRankLocal(int rankSize, int rank, LcalCommPtr *comm);

int LcalCommInitRank(LcalUniqueId commId, int rankSize, int rank, LcalCommPtr *comm);

int LcalCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, LcalCommPtr *comm,
                                       const bool isEnableAutoMagicNum = false);

int LcalCommInitRankWithDomain(int commDomain, int rankSize, int rank, LcalCommPtr *comm);

int LcalGetCommArgsDev(LcalCommPtr comm, GM_ADDR &commArgsPtr);

int LcalGetCommArgsHost(LcalCommPtr comm, Lcal::CommArgs *&commArgsPtr);

void LcalPrintDFX2Log(LcalCommPtr comm);

int LcalCommInit(int rank, int rankSize, LcalCommPtr *comms);

int LcalCommInitAll(uint32_t ndev, int32_t* devices, LcalCommPtr *comms);

int LcalCommInitThread(int rank, int rankSize, const char *uid, LcalCommPtr *comms);

int LcclAllReduce(void *sendBuf, void *recvBuf, int64_t count, HcclDataType dataType, HcclReduceOp op,
                  LcalCommPtr comm, aclrtStream stream);

int LcclAllGather(void *sendBuf, void *recvBuf, int64_t sendCount, HcclDataType dataType, LcalCommPtr comm,
                  aclrtStream stream);

int LcclReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                      LcalCommPtr comm, aclrtStream stream);
                             
int LcclBroadcast(void *buf, int64_t count, HcclDataType dataType, int root, LcalCommPtr comm,
                  aclrtStream stream);

int LcclCommDestroy(LcalCommPtr comm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // LCAL_API_H
