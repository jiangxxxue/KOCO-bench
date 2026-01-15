/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_LCOC_H
#define LCAL_LCOC_H

#include <lcal_comm.h>
#include <hccl.h>
#include "lcoc_args.h"
#include "tiling_args.h"

namespace Lcal {
class Lcoc {
public:
    Lcoc() = delete;
    explicit Lcoc(LcalComm &comm);
    explicit Lcoc(LcalComm *comm);
    ~Lcoc();
    int SetParam(LcalType lcalType, const CoCTiling &tiling, const CoCParamDesc &paramDesc);
    int AllGatherMatmul(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream = nullptr);
    int AllGatherMatmulV2(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream = nullptr);
    int MatmulReduceScatter(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
                            aclrtStream stream = nullptr);
    int MatmulAllReduce(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream = nullptr);
    int PureMatmul(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream = nullptr);
    int AllGatherMatmulReduceScatter(CoCInputPkg inputPkg, CoCOutputPkg outputPkg,
                                     void *workspace, aclrtStream stream = nullptr);
    int AllToAllVAllGatherMatmul(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
        aclrtStream stream = nullptr);
    int AllToAllVAllGatherMatmulHidden(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
        aclrtStream stream = nullptr);
    int MatmulReduceScatterAllToAllVHidden(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
        aclrtStream stream = nullptr);
    int64_t GetWorkspaceSize();
    LcalComm *GetComm();
    MatMulInfo &GetMatMulInfo();
    void GetTiling(CoCTiling &tiling);

private:
    int LaunchOperator(CoCInputPkg &inputPkg, CoCOutputPkg &outputPkg, void *workspace, aclrtStream stream);
    bool CheckBasic(const CoCInputPkg &inputPkg, const CoCOutputPkg &outputPkg, LcalType lcalType) const;
    bool CheckInputParam(LcalType lcalType, const CoCTiling &tiling, const CoCParamDesc &paramDesc) const;
    void SetLcocParam(LcalType lcalType, const CoCParamDesc &paramDesc);
    void SetTaskParam(LcalType lcalType, const CoCParamDesc &paramDesc, const LcalComm &comm);

private:
    LcalComm *comm_ = nullptr;
    CoCTilingData tiling_ = {};
    TaskParam taskParam_ = {};
    bool tilingSuccess_ = false;
};
}
#endif  // LCAL_LCOC_H
