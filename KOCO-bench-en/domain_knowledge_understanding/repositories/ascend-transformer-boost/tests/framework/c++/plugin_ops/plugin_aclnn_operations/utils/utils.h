/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_TEST_PLUGIN_OP_UTILS_H
#define ATB_TEST_PLUGIN_OP_UTILS_H

#include <string>
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"

namespace atb {
// 设置各个intensor的属性
void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs);

// 设置各个输入tensor并且为各个输入tensor分配内存空间，此处的输入tensor为手动设置，工程实现上可以使用torchTensor转换或者其他简单数据结构转换的方式
void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs);

// 设置各个outtensor并且为outtensor分配内存空间，同输入tensor设置
void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs);

void CreateTensorFromDesc(atb::Tensor &tensor, atb::TensorDesc &tensorDescs);

// 输出打印
void PrintOutTensorValue(atb::Tensor &outTensor);

// 创建图算子
atb::Status CreateGraphOperation(atb::Operation **operation);
}
#endif