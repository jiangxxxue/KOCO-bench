/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_FUSION_TILING_DATA
#define ATBOPS_FUSION_TILING_DATA

#include <cstdint>

namespace AtbOps {

struct FusionTilingData {
    int64_t key{0};
    int64_t mTile{0};
    int64_t nTile{0};
    int64_t kTile{0};
    int64_t processM{0};
    int64_t processN{0};
    int64_t processK{0};
    int64_t splitKSlices{1};
    int64_t swizzleDir{0};
    int64_t swizzleCnt{1};
    int64_t shuffleKType{0};
    int64_t workspaceReuse{0};
    int64_t workspaceBufferNum{2};
    int64_t pTiles{0};
    int64_t ubMaxBitSize{0};
};

struct KernelArgs {
    void *xDevice;
    void *xDeviceDup;
    int64_t offsetX = 0;
    int64_t sizeX0 = -1;
    int64_t sizeX1 = -1;
    int64_t strideX0 = -1;
    int64_t strideX1 = -1;
    void *yDevice;
    void *yDeviceDup;
    int64_t offsetY = 0;
    int64_t sizeY0 = -1;
    int64_t sizeY1 = -1;
    int64_t strideY0 = -1;
    int64_t strideY1 = -1;
    void *vDevice;
    void *vDeviceDup;
    int64_t offsetV = 0;
    int64_t sizeV0 = -1;
    int64_t sizeV1 = -1;
    int64_t strideV0 = -1;
    int64_t strideV1 = -1;
    void *oDevice;
    void *oDeviceDup;
    int64_t offsetO =  -1;
    int64_t sizeO0 =  -1;
    int64_t sizeO1 =  -1;
    int64_t strideO0 = -1;
    int64_t strideO1 = -1;
    void *tilingDevice;
    void *tilingDeviceDup;
    int64_t offsetTiling = 0;
    int64_t sizeTiling = sizeof(FusionTilingData);
    int64_t strideTiling = 1;
};

struct KernelArgsForInferShapeWorkspaceWithTiling {
    void *xDevice;
    void *xDeviceDup;
    int64_t offsetX = 0;
    int64_t sizeX0 = -1;
    int64_t sizeX1 = -1;
    int64_t strideX0 = -1;
    int64_t strideX1 = 1;
    void *yDevice;
    void *yDeviceDup;
    int64_t offsetY = 0;
    int64_t sizeY0 = -1;
    int64_t sizeY1 = -1;
    int64_t strideY0 = -1;
    int64_t strideY1 = 1;
    void *vDevice;
    void *vDeviceDup;
    int64_t offsetV = 0;
    int64_t sizeV0 = -1;
    int64_t sizeV1 = -1;
    int64_t strideV0 = -1;
    int64_t strideV1 = 1;
    void *tilingDevice;
    void *tilingDeviceDup;
    int64_t offsetTiling = 0;
    int64_t sizeTiling = sizeof(FusionTilingData);
    int64_t strideTiling = 1;
};

typedef void (*TILING_FUNC_GET)(void *);
typedef int64_t (*INFER_WORKSPACE_T)(void *);

} // namespace AtbOps
#endif