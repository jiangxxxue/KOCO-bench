/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_UTILS_H
#define ATB_UTILS_H
#include <cstdint>
#include "atb/types.h"

//!
//! \file utils.h
//!
//! \brief 定义加速库公共数据接口类
//!

namespace atb {

//!
//! \class Utils.
//!
//! \brief 加速库公共工具接口类.
//!
//! 该接口类定义了一系列的公共接口
//!
class Utils {
public:
    //!
    //! \brief 获取加速库版本信息。
    //!
    //! \return 返回字符串类型.
    //!
    static std::string GetAtbVersion();

    //!
    //! \brief 返回Tensor对象的数据存储大小。
    //!
    //! \param tensor 传入Tensor
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorSize(const Tensor &tensor);

    //!
    //! \brief 返回Tensor对象的数据存储大小。
    //!
    //! \param tensorDesc 传入TensorDesc
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorSize(const TensorDesc &tensorDesc);

    //!
    //! \brief 返回Tensor对象的数据个数。
    //!
    //! \param tensor 传入Tensor
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorNumel(const Tensor &tensor);

    //!
    //! \brief 返回Tensor对象的数据个数。
    //!
    //! \param tensorDesc 传入TensorDesc
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorNumel(const TensorDesc &tensorDesc);

    //!
    //! \brief 量化场景使用。float数组转成uint64数组，实现逻辑是复制float到uint64的后32位，uint64的前32位置0。
    //!
    //! \param src 输入float数组
    //! \param dest 转化得到的uint64数组
    //! \param itemCount 数组元素个数
    //!
    static void QuantParamConvert(const float *src, uint64_t *dest, uint64_t itemCount);
};
} // namespace atb
#endif