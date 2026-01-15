/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_ATTR_GROUP_SERIALIZE_H
#define INC_GRAPH_ATTR_GROUP_SERIALIZE_H

#include "graph/attr_store.h"
#include "graph/ge_error_codes.h"

namespace ge {
namespace proto {
class AttrGroups;
}
class AttrGroupSerialize {
public:
    static graphStatus SerializeAllAttr(proto::AttrGroups &attrGroups, const AttrStore &attrStore);
    static graphStatus DeserializeAllAttr(const proto::AttrGroups &attrGroup, AttrStore &attrStore);

private:
    static graphStatus OtherGroupSerialize(proto::AttrGroups &attrGroups, const AttrStore &attrStore);
    static graphStatus OtherGroupDeserialize(const proto::AttrGroups &attrGroups, AttrStore &attrStore);
};
} // namespace ge

#endif // INC_GRAPH_ATTR_GROUP_SERIALIZE_H
