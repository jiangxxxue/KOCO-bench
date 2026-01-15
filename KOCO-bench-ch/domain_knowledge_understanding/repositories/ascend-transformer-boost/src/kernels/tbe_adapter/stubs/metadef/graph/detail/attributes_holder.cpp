/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/detail/attributes_holder.h"

#define UNUSED_VALUE(x) (void)(x)
namespace ge {
void AttrHolder::CopyAttrsFrom(const AttrHolder &holder)
{
    UNUSED_VALUE(holder);
}
void AttrHolder::CopyFrom(const AttrHolder &holder)
{
    UNUSED_VALUE(holder);
}

graphStatus AttrHolder::SetAttr(const std::string &name, const AnyValue &value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return GRAPH_SUCCESS;
}
graphStatus AttrHolder::TrySetAttr(const std::string &name, const AnyValue &value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return GRAPH_SUCCESS;
}
graphStatus AttrHolder::AddRequiredAttr(const std::string &name)
{
    UNUSED_VALUE(name);
    return GRAPH_SUCCESS;
}

graphStatus AttrHolder::GetAttr(const std::string &name, AnyValue &value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(value);
    return GRAPH_SUCCESS;
}

bool AttrHolder::HasAttr(const std::string &name) const
{
    UNUSED_VALUE(name);
    return true;
}

graphStatus AttrHolder::DelAttr(const std::string &name)
{
    UNUSED_VALUE(name);
    return GRAPH_SUCCESS;
}

const std::map<std::string, AnyValue> AttrHolder::GetAllAttrs() const
{
    static std::map<std::string, AnyValue> all;
    return all;
}

const std::set<std::string> AttrHolder::GetAllAttrNames() const
{
    static std::set<std::string> all;
    return all;
}

template <> void GeIrProtoHelper<proto::AttrDef>::InitDefault() {}

template <> void GeIrProtoHelper<proto::TensorDef>::InitDefault() {}

template <> void GeIrProtoHelper<proto::TensorDescriptor>::InitDefault() {}

template <> void GeIrProtoHelper<proto::ShapeDef>::InitDefault() {}

template <> void GeIrProtoHelper<proto::NamedAttrs>::InitDefault() {}

template <> void GeIrProtoHelper<proto::ModelDef>::InitDefault() {}

template <> void GeIrProtoHelper<proto::OpDef>::InitDefault() {}

template <> void GeIrProtoHelper<proto::GraphDef>::InitDefault() {}
} // namespace ge
