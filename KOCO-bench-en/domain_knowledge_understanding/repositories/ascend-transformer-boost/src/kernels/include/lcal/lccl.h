/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_LCCL_H
#define LCAL_LCCL_H

#include <lcal_comm.h>


namespace Lcal {
class Lccl {
public:
    Lccl() = delete;
    explicit Lccl(LcalComm *comm);
    explicit Lccl(LcalComm &comm);
    ~Lccl();
    uint32_t GetBlockNum(LcalType cclType, uint32_t rankSize, int64_t dataSize, int localRankSize, uint32_t extraFlag)
        const;
    int AllReduce(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType,
        HcclReduceOp op = HCCL_REDUCE_SUM, aclrtStream stream = nullptr,
        HcclDataType outputDataType = HCCL_DATA_TYPE_RESERVED, const void *scale = nullptr, int64_t scaleCount = 0,
        const void *offset = nullptr) const;
    int ReduceScatter(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType,
        HcclReduceOp op = HCCL_REDUCE_SUM, aclrtStream stream = nullptr) const;
    int AllGather(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, aclrtStream stream) const;
    int All2All(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, aclrtStream stream) const;
    int All2All(void *sendBuff, void *recvBuff, int64_t count, int burstLen,
        int stride, HcclDataType dataType, aclrtStream stream) const;
    int All2AllVC(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, aclrtStream stream) const;

    int Broadcast(void *buff, int64_t count, HcclDataType dataType, int32_t root, aclrtStream stream) const;
    int BandwidthTest(const void *buff, void *recvBuff, int64_t count, HcclDataType dataType,
        int32_t root, aclrtStream stream) const;
    friend class LcclTest;

private:
    bool CheckDataType(const HcclDataType &dataType) const;
    bool CheckBuff(const void *sendBuff, const void *recvBuff) const;
    int LoopBack(const void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType) const;

private:
    LcalComm *comm_ = nullptr;
    int rank_ = 0;
    int rankSize_ = 0;
};
}
#endif // LCAL_LCCL_H
