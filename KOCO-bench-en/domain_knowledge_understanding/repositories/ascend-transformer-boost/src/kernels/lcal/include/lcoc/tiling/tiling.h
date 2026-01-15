/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_TILING_H
#define LCAL_TILING_H

#include <iostream>
#include <array>
#include "tiling_args.h"
#include "lcal_types.h"
#include "lcal_comm.h"
#include "lcoc.h"

namespace Lcal {
class CoCTilingFunc {
public:
    CoCTilingFunc(const CoCTilingFunc &) = delete;
    CoCTilingFunc &operator = (const CoCTilingFunc &) = delete;
    CoCTilingFunc() {}
    virtual ~CoCTilingFunc() {}
    CoCTilingData GenerateTiling(const TaskParam &taskParam, const CoCTiling &tiling);

    virtual bool CheckTiling(const TaskParam &taskParam);
    virtual void GetDefaultTiling(const TaskParam &taskParam);

protected:
    CoCTilingData cocTilingData = {};
};

class CoCMatmulAllReduceTilingFunc : public CoCTilingFunc {
public:
    CoCMatmulAllReduceTilingFunc(const CoCMatmulAllReduceTilingFunc &) = delete;
    CoCMatmulAllReduceTilingFunc &operator = (const CoCMatmulAllReduceTilingFunc &) = delete;
    CoCMatmulAllReduceTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};

class CoCMatmulAllReduceDeterTilingFunc : public CoCMatmulAllReduceTilingFunc {
public:
    CoCMatmulAllReduceDeterTilingFunc(const CoCMatmulAllReduceDeterTilingFunc &) = delete;
    CoCMatmulAllReduceDeterTilingFunc &operator = (const CoCMatmulAllReduceDeterTilingFunc &) = delete;
    CoCMatmulAllReduceDeterTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};

class CoCMatmulReduceScatterTilingFunc : public CoCMatmulAllReduceTilingFunc {
public:
    CoCMatmulReduceScatterTilingFunc(const CoCMatmulReduceScatterTilingFunc &) = delete;
    CoCMatmulReduceScatterTilingFunc &operator = (const CoCMatmulReduceScatterTilingFunc &) = delete;
    CoCMatmulReduceScatterTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};

class CoCAllGatherMatmulTilingFunc : public CoCTilingFunc {
public:
    CoCAllGatherMatmulTilingFunc(const CoCAllGatherMatmulTilingFunc &) = delete;
    CoCAllGatherMatmulTilingFunc &operator = (const CoCAllGatherMatmulTilingFunc &) = delete;
    CoCAllGatherMatmulTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};

class CoCAllGatherMatmulV2TilingFunc : public CoCTilingFunc {
public:
    CoCAllGatherMatmulV2TilingFunc(const CoCAllGatherMatmulV2TilingFunc &) = delete;
    CoCAllGatherMatmulV2TilingFunc &operator = (const CoCAllGatherMatmulV2TilingFunc &) = delete;
    CoCAllGatherMatmulV2TilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};

class CoCAllgatherMatmulReduceScatterTilingFunc : public CoCTilingFunc {
public:
    CoCAllgatherMatmulReduceScatterTilingFunc(const CoCAllgatherMatmulReduceScatterTilingFunc &) = delete;
    CoCAllgatherMatmulReduceScatterTilingFunc &operator = (const CoCAllgatherMatmulReduceScatterTilingFunc &) = delete;
    CoCAllgatherMatmulReduceScatterTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};
class CoCAllToAllAllGatherMatmulTilingFunc : public CoCAllGatherMatmulTilingFunc {
public:
    CoCAllToAllAllGatherMatmulTilingFunc(const CoCAllToAllAllGatherMatmulTilingFunc &) = delete;
    CoCAllToAllAllGatherMatmulTilingFunc &operator = (const CoCAllToAllAllGatherMatmulTilingFunc &) = delete;
    CoCAllToAllAllGatherMatmulTilingFunc() {}
    bool CheckTiling(const TaskParam &tilingInfo) override;
    void GetDefaultTiling(const TaskParam &tilingInfo) override;
};
class CoCAllToAllAllGatherMatmulHiddenTilingFunc : public CoCAllGatherMatmulTilingFunc {
public:
    CoCAllToAllAllGatherMatmulHiddenTilingFunc(const CoCAllToAllAllGatherMatmulHiddenTilingFunc &) = delete;
    CoCAllToAllAllGatherMatmulHiddenTilingFunc &operator = (
        const CoCAllToAllAllGatherMatmulHiddenTilingFunc &) = delete;
    CoCAllToAllAllGatherMatmulHiddenTilingFunc() {}
    bool CheckTiling(const TaskParam &tilingInfo) override;
    void GetDefaultTiling(const TaskParam &tilingInfo) override;
};

class CoCMatmulReduceScatterAllToAllHiddenTilingFunc : public CoCMatmulReduceScatterTilingFunc {
public:
    CoCMatmulReduceScatterAllToAllHiddenTilingFunc(const CoCMatmulReduceScatterAllToAllHiddenTilingFunc &) = delete;
    CoCMatmulReduceScatterAllToAllHiddenTilingFunc &operator = (
        const CoCMatmulReduceScatterAllToAllHiddenTilingFunc &) = delete;
    CoCMatmulReduceScatterAllToAllHiddenTilingFunc() {}
    bool CheckTiling(const TaskParam &tilingInfo) override;
    void GetDefaultTiling(const TaskParam &tilingInfo) override;
};

}
#endif // LCAL_TILING_H
