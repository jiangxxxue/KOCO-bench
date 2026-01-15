/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_attr_value.h"

#include "graph/utils/graph_utils.h"

#define UNUSED_VALUE(x) (void)(x)
namespace ge {
void NamedAttrs::SetName(const std::string &name) { name_ = name; }

std::string NamedAttrs::GetName() const { return name_; }

AnyValue NamedAttrs::GetItem(const std::string &key) const
{
    AnyValue value;
    (void)GetAttr(key, value);
    return value;
}

ProtoAttrMap &NamedAttrs::MutableAttrMap() { return attrs_; }

ConstProtoAttrMap &NamedAttrs::GetAttrMap() const { return attrs_; }

bool AttrUtils::HasAttr(ConstAttrHolderAdapter &&obj, const std::string &name)
{
    if (!obj) {
        return false;
    }
    return obj->HasAttr(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      int32_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      uint32_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc)
{
    UNUSED_VALUE(org_op_desc);
    return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc)
{
    UNUSED_VALUE(org_op_desc);
    return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<int64_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<int64_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetInt(AttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      const int64_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      int64_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      uint64_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetFloat(AttrHolderAdapter &&obj,
                                                                        const std::string &name,
                                                                        const float32_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetFloat(ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name,
                                                                        float32_t &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListFloat(AttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            const std::vector<float32_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListFloat(ConstAttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            std::vector<float32_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetBool(AttrHolderAdapter &&obj,
                                                                       const std::string &name,
                                                                       const bool &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetBool(ConstAttrHolderAdapter &&obj,
                                                                       const std::string &name,
                                                                       bool &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListBool(AttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           const std::vector<bool> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListBool(ConstAttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           std::vector<bool> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetStr(AttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      const std::string &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetStr(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name,
                                                                      std::string &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::string *AttrUtils::GetStr(ConstAttrHolderAdapter &&obj,
                                                                                    const std::string &name)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    static std::string str;
    return &str;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListStr(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<std::string> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListStr(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<std::string> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensorDesc(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const GeTensorDesc &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetTensorDesc(ConstAttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             GeTensorDesc &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensorDesc(AttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 const std::vector<GeTensorDesc> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListTensorDesc(ConstAttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 std::vector<GeTensorDesc> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetNamedAttrs(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const NamedAttrs &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetNamedAttrs(ConstAttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             NamedAttrs &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListNamedAttrs(AttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 const std::vector<NamedAttrs> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListNamedAttrs(ConstAttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 std::vector<NamedAttrs> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetDataType(AttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           const DataType &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetDataType(ConstAttrHolderAdapter &&obj,
                                                                           const std::string &name,
                                                                           DataType &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListDataType(AttrHolderAdapter &&obj,
                                                                               const std::string &name,
                                                                               const std::vector<DataType> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListDataType(ConstAttrHolderAdapter &&obj,
                                                                               const std::string &name,
                                                                               std::vector<DataType> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListListInt(AttrHolderAdapter &&obj,
    const std::string &name,
    const std::vector<std::vector<int64_t>> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListListInt(ConstAttrHolderAdapter &&obj,
                                                                              const std::string &name,
                                                                              std::vector<std::vector<int64_t>> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListListFloat(AttrHolderAdapter &&obj,
    const std::string &name,
    const std::vector<std::vector<float32_t>> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListListFloat(ConstAttrHolderAdapter &&obj,
    const std::string &name,
    std::vector<std::vector<float32_t>> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<uint32_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrUtils::AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          const std::vector<int32_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListInt(AttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::initializer_list<int64_t> &&value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<int32_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj,
                                                                          const std::string &name,
                                                                          std::vector<uint32_t> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensor(AttrUtils::AttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         const GeTensor &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensor(AttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         const GeTensorPtr &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetTensor(AttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         const ConstGeTensorPtr &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(AttrUtils::AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const std::vector<GeTensor> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const std::vector<GeTensorPtr> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             const std::vector<ConstGeTensorPtr> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj,
    const std::string &name,
    std::initializer_list<ConstGeTensorPtr> &&value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

// 所有权UT测试，不能把属性上的GeTensor给错误释放了
// 而且这里的行为与老版本是不一样的，老版本中，即使属性的owner生命周期结束析构了，通过本接口获取的value仍然是可用的
// 但是新接口中，owner没有转移，owner析构后，value指向的内存就被释放了，这里需要排查
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::MutableTensor(AttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             GeTensorPtr &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetTensor(ConstAttrHolderAdapter &&obj,
                                                                         const std::string &name,
                                                                         ConstGeTensorPtr &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListTensor(ConstAttrHolderAdapter &&obj,
                                                                             const std::string &name,
                                                                             std::vector<ConstGeTensorPtr> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::MutableListTensor(AttrHolderAdapter &&obj,
                                                                                 const std::string &name,
                                                                                 std::vector<GeTensorPtr> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetGraph(AttrUtils::AttrHolderAdapter &&obj,
                                                                        const std::string &name,
                                                                        const ComputeGraphPtr &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListGraph(AttrUtils::AttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            const std::vector<ComputeGraphPtr> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetGraph(AttrUtils::ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name, ComputeGraphPtr &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListGraph(AttrUtils::ConstAttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            std::vector<ComputeGraphPtr> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetBytes(AttrUtils::AttrHolderAdapter &&obj,
                                                                        const std::string &name, const Buffer &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetBytes(ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name, Buffer &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetListBytes(AttrUtils::AttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            const std::vector<Buffer> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetListBytes(AttrUtils::ConstAttrHolderAdapter &&obj,
                                                                            const std::string &name,
                                                                            std::vector<Buffer> &value)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetZeroCopyBytes(AttrHolderAdapter &&obj,
                                                                                const std::string &name,
                                                                                Buffer &&buffer)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(buffer);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetZeroCopyBytes(ConstAttrHolderAdapter &&obj,
                                                                                const std::string &name,
                                                                                Buffer &buffer)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(buffer);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetZeroCopyListBytes(AttrHolderAdapter &&obj,
                                                                                    const std::string &name,
                                                                                    std::vector<Buffer> &list_buffer)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(list_buffer);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj,
                                                                                    const std::string &name,
                                                                                    std::vector<Buffer> &list_buffer)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(name);
    UNUSED_VALUE(list_buffer);
    return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::map<std::string, AnyValue>
AttrUtils::GetAllAttrs(ConstAttrHolderAdapter &&obj)
{
    UNUSED_VALUE(obj);
    const std::map<std::string, AnyValue> empty;
    return empty;
}

std::string AttrUtils::GetAttrsStrAfterRid(ConstAttrHolderAdapter &&obj, const std::set<std::string> &un_compute_attrs)
{
    UNUSED_VALUE(obj);
    UNUSED_VALUE(un_compute_attrs);
    return "";
}
std::string AttrUtils::GetAllAttrsStr(ConstAttrHolderAdapter &&obj)
{
    UNUSED_VALUE(obj);
    return "";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::ClearAllAttrs(AttrHolderAdapter &&obj)
{
    UNUSED_VALUE(obj);
    return true;
}
} // namespace ge
