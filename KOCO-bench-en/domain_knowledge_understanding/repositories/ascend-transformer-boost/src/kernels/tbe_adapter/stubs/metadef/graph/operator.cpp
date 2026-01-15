/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "external/graph/operator.h"

#include <cstdint>
#include <algorithm>
#include <mutex>
#include <queue>
#include <set>

#include "debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"

#define UNUSED_VALUE(x) (void)(x)
namespace ge {
Graph::Graph(const char_t *name)
{
    UNUSED_VALUE(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromNode(ge::ConstNodePtr node_ptr)
{
    UNUSED_VALUE(node_ptr);
    return Operator("default");
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDescUtils::CopyOperators(
    const ComputeGraphPtr &dst_compute_graph, const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
    const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
    const std::map<std::string, ge::Operator> &src_op_list, std::map<std::string, ge::Operator> &dst_op_list)
{
    UNUSED_VALUE(dst_compute_graph);
    UNUSED_VALUE(node_old_2_new);
    UNUSED_VALUE(op_desc_old_2_new);
    UNUSED_VALUE(src_op_list);
    UNUSED_VALUE(dst_op_list);
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDescUtils::CopyOperatorLinks(
    const std::map<std::string, ge::Operator> &src_op_list, std::map<std::string, ge::Operator> &dst_op_list)
{
    UNUSED_VALUE(src_op_list);
    UNUSED_VALUE(dst_op_list);
    return GRAPH_SUCCESS;
}

Operator::Operator(const std::string &type)
{
    UNUSED_VALUE(type);
}

Operator::Operator(const char_t *type)
{
    UNUSED_VALUE(type);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromOpDesc(OpDescPtr op_desc)
{
    UNUSED_VALUE(op_desc);
    return Operator("default");
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator &oprt)
{
    UNUSED_VALUE(oprt);
    return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstNodePtr NodeUtilsEx::GetNodeFromOperator(const Operator &op)
{
    UNUSED_VALUE(op);
    return nullptr;
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const std::string &name, const std::string &type)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(type);
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const AscendString &name, const AscendString &type)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(type);
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const char_t *name, const char_t *type)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(type);
}

Operator::Operator(ge::OperatorImplPtr &&op_impl)
{
    UNUSED_VALUE(op_impl);
}

bool Operator::IsEmpty() const { return true; }

std::string Operator::GetName() const { return ""; }

graphStatus Operator::GetName(AscendString &name) const
{
    UNUSED_VALUE(name);
    return GRAPH_SUCCESS;
}

GE_FUNC_HOST_VISIBILITY Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(src_oprt);
    return *this;
}

GE_FUNC_HOST_VISIBILITY Operator &Operator::SetInput(const char_t *dst_name, const ge::Operator &src_oprt)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(src_oprt);
    return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::OutHandler &out_handler)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(out_handler);
    return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, const ge::OutHandler &out_handler)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(out_handler);
    return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt, const std::string &name)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(name);
    return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, const ge::Operator &src_oprt, const char_t *name)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(name);
    return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt, uint32_t index)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(index);
    return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, const ge::Operator &src_oprt, uint32_t index)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(index);
    return *this;
}

Operator &Operator::SetInput(uint32_t dst_index, const Operator &src_oprt, uint32_t src_index)
{
    UNUSED_VALUE(dst_index);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(src_index);
    return *this;
}

Operator &Operator::AddControlInput(const Operator &src_oprt)
{
    UNUSED_VALUE(src_oprt);
    return *this;
}

graphStatus Operator::GetInputConstData(const std::string &dst_name, Tensor &data) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(data);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstData(const char_t *dst_name, Tensor &data) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(data);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstDataOut(const std::string &dst_name, Tensor &data) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(data);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstDataOut(const char_t *dst_name, Tensor &data) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(data);
    return GRAPH_SUCCESS;
}

std::shared_ptr<const Node> Operator::GetNode() const { return nullptr; }

TensorDesc Operator::GetInputDesc(const std::string &name) const
{
    UNUSED_VALUE(name);
    return TensorDesc();
}

TensorDesc Operator::GetInputDescByName(const char_t *name) const
{
    UNUSED_VALUE(name);
    return TensorDesc();
}

void Operator::SetInferenceContext(const InferenceContextPtr &inference_context)
{
    UNUSED_VALUE(inference_context);
}

InferenceContextPtr Operator::GetInferenceContext() const { return nullptr; }

TensorDesc Operator::GetInputDesc(uint32_t index) const
{
    UNUSED_VALUE(index);
    return TensorDesc();
}

graphStatus Operator::TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::TryGetInputDesc(const char_t *name, TensorDesc &tensor_desc) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateInputDesc(const std::string &name, const ge::TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateInputDesc(const char_t *name, const ge::TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

OutHandler Operator::GetOutput(const std::string &name) const
{
    return GetOutput(name.c_str());
}

OutHandler Operator::GetOutput(const char_t *name) const
{
    UNUSED_VALUE(name);
    return nullptr;
}

OutHandler Operator::GetOutput(uint32_t index) const
{
    UNUSED_VALUE(index);
    return nullptr;
}

TensorDesc Operator::GetOutputDesc(const std::string &name) const
{
    UNUSED_VALUE(name);
    return TensorDesc();
}

TensorDesc Operator::GetOutputDescByName(const char_t *name) const
{
    UNUSED_VALUE(name);
    return TensorDesc();
}

TensorDesc Operator::GetOutputDesc(uint32_t index) const
{
    UNUSED_VALUE(index);
    return TensorDesc();
}

graphStatus Operator::UpdateOutputDesc(const std::string &name, const ge::TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateOutputDesc(const char_t *name, const ge::TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

TensorDesc Operator::GetDynamicInputDesc(const std::string &name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return TensorDesc();
}

TensorDesc Operator::GetDynamicInputDesc(const char_t *name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return TensorDesc();
}

graphStatus Operator::UpdateDynamicInputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateDynamicInputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

TensorDesc Operator::GetDynamicOutputDesc(const std::string &name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return TensorDesc();
}

TensorDesc Operator::GetDynamicOutputDesc(const char_t *name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return TensorDesc();
}

graphStatus Operator::UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateDynamicOutputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    UNUSED_VALUE(tensor_desc);
    return GRAPH_SUCCESS;
}

graphStatus Operator::InferShapeAndType() { return GRAPH_SUCCESS; }

graphStatus Operator::VerifyAllAttr(bool disable_common_verifier)
{
    UNUSED_VALUE(disable_common_verifier);
    return GRAPH_SUCCESS;
}

GE_FUNC_HOST_VISIBILITY size_t Operator::GetInputsSize() const { return 0UL; }

GE_FUNC_HOST_VISIBILITY size_t Operator::GetOutputsSize() const { return 0UL; }

const std::map<std::string, std::string> Operator::GetAllAttrNamesAndTypes() const
{
    static std::map<std::string, std::string> attr_types;
    return attr_types;
}

graphStatus Operator::GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const
{
    UNUSED_VALUE(attr_name_types);
    return GRAPH_SUCCESS;
}

void Operator::InputRegister(const std::string &name)
{
    UNUSED_VALUE(name);
}

void Operator::InputRegister(const char_t *name)
{
    UNUSED_VALUE(name);
}

void Operator::InputRegister(const char_t *name, const char_t *datatype_symbol)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(datatype_symbol);
}

void Operator::OptionalInputRegister(const std::string &name)
{
    UNUSED_VALUE(name);
}

void Operator::OptionalInputRegister(const char_t *name)
{
    UNUSED_VALUE(name);
}

void Operator::OptionalInputRegister(const char_t *name, const char_t *datatype_symbol)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(datatype_symbol);
}

void Operator::InferFuncRegister(const std::function<graphStatus(Operator &)> &func)
{
    UNUSED_VALUE(func);
}

void Operator::InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func)
{
    UNUSED_VALUE(func);
}

void Operator::VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func)
{
    UNUSED_VALUE(func);
}

void Operator::OutputRegister(const std::string &name)
{
    UNUSED_VALUE(name);
}

void Operator::OutputRegister(const char_t *name)
{
    UNUSED_VALUE(name);
}

void Operator::OutputRegister(const char_t *name, const char_t *datatype_symbol)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(datatype_symbol);
}

void Operator::DynamicInputRegister(const std::string &name, const uint32_t num, bool is_push_back)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(is_push_back);
}

void Operator::DynamicInputRegister(const char_t *name, const uint32_t num, bool is_push_back)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(is_push_back);
}

void Operator::DynamicInputRegisterByIndex(const std::string &name, const uint32_t num, size_t index)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(index);
}

void Operator::DynamicInputRegisterByIndex(const char_t *name, const uint32_t num, size_t index)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(index);
}

int32_t Operator::GetDynamicInputNum(const std::string &name) const
{
    UNUSED_VALUE(name);
    return 0;
}

int32_t Operator::GetDynamicInputNum(const char_t *name) const
{
    UNUSED_VALUE(name);
    return 0;
}

void Operator::DynamicOutputRegister(const std::string &name, const uint32_t num, bool is_push_back)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(is_push_back);
}

void Operator::DynamicOutputRegister(const char_t *name, const uint32_t num, bool is_push_back)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(is_push_back);
}

int32_t Operator::GetDynamicOutputNum(const std::string &name) const
{
    UNUSED_VALUE(name);
    return 0;
}

int32_t Operator::GetDynamicOutputNum(const char_t *name) const
{
    UNUSED_VALUE(name);
    return 0;
}

void Operator::RequiredAttrRegister(const std::string &name)
{
    UNUSED_VALUE(name);
}

void Operator::RequiredAttrRegister(const char_t *name)
{
    UNUSED_VALUE(name);
}

void Operator::DataTypeRegister(const char_t *datatype_symbol, const TensorType &type_range)
{
    UNUSED_VALUE(datatype_symbol);
    UNUSED_VALUE(type_range);
}

void Operator::DataTypeRegister(const char_t *datatype_symbol, const ListTensorType &list_type_range)
{
    UNUSED_VALUE(datatype_symbol);
    UNUSED_VALUE(list_type_range);
}

graphStatus Operator::VerifyAll() { return GRAPH_SUCCESS; }

std::string Operator::GetOpType() const { return ""; }

graphStatus Operator::GetOpType(AscendString &type) const
{
    UNUSED_VALUE(type);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetInput(const std::string &dst_name, uint32_t dst_index, const ge::Operator &src_oprt)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(dst_index);
    UNUSED_VALUE(src_oprt);
    return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, uint32_t dst_index, const ge::Operator &src_oprt)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(dst_index);
    UNUSED_VALUE(src_oprt);
    return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, uint32_t dst_index, const ge::Operator &src_oprt,
                             const std::string &name)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(dst_index);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(name);
    return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, uint32_t dst_index, const ge::Operator &src_oprt,
                             const char_t *name)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(dst_index);
    UNUSED_VALUE(src_oprt);
    UNUSED_VALUE(name);
    return *this;
}

OperatorImplPtr Operator::GetOperatorImplPtr() const { return nullptr; }

void Operator::BreakConnect() const {}

void Operator::AttrRegister(const std::string &name, const std::string &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const std::vector<std::string> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, int64_t attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const vector<int64_t> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, float32_t attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const vector<float32_t> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const char_t *attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, bool attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const vector<bool> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const vector<vector<int64_t>> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const NamedAttrs &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const vector<NamedAttrs> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const AscendString &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const AscendString &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const std::vector<AscendString> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const std::vector<AscendString> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

Operator &Operator::SetAttr(const std::string &name, const std::string &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, std::string &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, int32_t &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, bool &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, int64_t &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, float32_t &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<long> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<float> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<int> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const std::vector<std::string> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, std::vector<std::string> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char_t *name, const char_t *attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetInputAttr(const int32_t index, const char_t *name, const char_t *attr_value)
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetInputAttr(const char_t *dst_name, const char_t *name, const char_t *attr_value)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetInputAttr(const int32_t index, const char_t *name, const std::vector<AscendString> &attr_value)
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetOutputAttr(const int32_t index, const char_t *name, const std::vector<AscendString> &attr_value)
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetInputAttr(const char_t *dst_name, const char_t *name,
                                 const std::vector<AscendString> &attr_value)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetOutputAttr(const char_t *dst_name, const char_t *name,
                                  const std::vector<AscendString> &attr_value)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetInputAttr(const char_t *dst_name, const char_t *name,
                                   std::vector<AscendString> &attr_value) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetOutputAttr(const char_t *dst_name, const char_t *name,
                                    std::vector<AscendString> &attr_value) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputAttr(const int32_t index, const char_t *name, std::vector<AscendString> &attr_value) const
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetOutputAttr(const int32_t index, const char_t *name,
                                    std::vector<AscendString> &attr_value) const
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetOutputAttr(const int32_t index, const char_t *name, const char_t *attr_value)
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetOutputAttr(const char_t *dst_name, const char_t *name, const char_t *attr_value)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetInputAttr(const int32_t index, const char_t *name, const AscendString &attr_value)
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetInputAttr(const char_t *dst_name, const char_t *name, const AscendString &attr_value)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetOutputAttr(const int32_t index, const char_t *name, const AscendString &attr_value)
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetOutputAttr(const char_t *dst_name, const char_t *name, const AscendString &attr_value)
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetOutputAttr(const char_t *dst_name, const char_t *name, AscendString &attr_value) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputAttr(const char_t *dst_name, const char_t *name, AscendString &attr_value) const
{
    UNUSED_VALUE(dst_name);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputAttr(const int32_t index, const char_t *name, AscendString &attr_value) const
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetOutputAttr(const int32_t index, const char_t *name, AscendString &attr_value) const
{
    UNUSED_VALUE(index);
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char_t *name, const AscendString &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const char_t *name, AscendString &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char_t *name, const std::vector<AscendString> &attr_values)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_values);
    return *this;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<AscendString> &attr_values) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_values);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const Tensor &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const char_t *name, const Tensor &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const std::string &name, const std::vector<Tensor> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const char_t *name, const std::vector<Tensor> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, Tensor &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, Tensor &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const std::string &name, std::vector<Tensor> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<Tensor> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const OpBytes &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const char_t *name, const OpBytes &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, OpBytes &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, OpBytes &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, ge::AttrValue &&attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const char_t *name, ge::AttrValue &&attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, ge::AttrValue &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, ge::AttrValue &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const std::vector<ge::DataType> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const char_t *name, const std::vector<ge::DataType> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, std::vector<ge::DataType> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<ge::DataType> &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const ge::DataType &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

Operator &Operator::SetAttr(const char_t *name, const ge::DataType &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, ge::DataType &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, ge::DataType &attr_value) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
    return GRAPH_SUCCESS;
}

void Operator::AttrRegister(const std::string &name, const std::vector<ge::DataType> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const std::vector<ge::DataType> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const ge::DataType &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const ge::DataType &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const Tensor &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const Tensor &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const std::vector<Tensor> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const vector<Tensor> &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const std::string &name, const OpBytes &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::AttrRegister(const char_t *name, const OpBytes &attr_value)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(attr_value);
}

void Operator::SubgraphRegister(const std::string &ir_name, bool dynamic)
{
    UNUSED_VALUE(ir_name);
    UNUSED_VALUE(dynamic);
}

void Operator::SubgraphRegister(const char_t *ir_name, bool dynamic)
{
    UNUSED_VALUE(ir_name);
    UNUSED_VALUE(dynamic);
}

void Operator::SubgraphCountRegister(const std::string &ir_name, uint32_t count)
{
    UNUSED_VALUE(ir_name);
    UNUSED_VALUE(count);
}

void Operator::SubgraphCountRegister(const char_t *ir_name, uint32_t count)
{
    UNUSED_VALUE(ir_name);
    UNUSED_VALUE(count);
}

void Operator::SetSubgraphBuilder(const std::string &ir_name, uint32_t index, const SubgraphBuilder &builder)
{
    UNUSED_VALUE(ir_name);
    UNUSED_VALUE(index);
    UNUSED_VALUE(builder);
}

void Operator::SetSubgraphBuilder(const char_t *ir_name, uint32_t index, const SubgraphBuilder &builder)
{
    UNUSED_VALUE(ir_name);
    UNUSED_VALUE(index);
    UNUSED_VALUE(builder);
}

std::vector<std::string> Operator::GetSubgraphNames() const
{
    static std::vector<std::string> names;
    return names;
}

graphStatus Operator::GetSubgraphNames(std::vector<AscendString> &names) const
{
    UNUSED_VALUE(names);
    return GRAPH_SUCCESS;
}

SubgraphBuilder Operator::GetDynamicSubgraphBuilder(const std::string &name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return nullptr;
}

SubgraphBuilder Operator::GetDynamicSubgraphBuilder(const char_t *name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return nullptr;
}

SubgraphBuilder Operator::GetSubgraphBuilder(const std::string &name) const
{
    UNUSED_VALUE(name);
    return nullptr;
}

SubgraphBuilder Operator::GetSubgraphBuilder(const char_t *name) const
{
    UNUSED_VALUE(name);
    return nullptr;
}

Graph Operator::GetSubgraphImpl(const std::string &name) const
{
    UNUSED_VALUE(name);
    return Graph();
}

Graph Operator::GetSubgraphImpl(const char_t *name) const
{
    UNUSED_VALUE(name);
    return Graph();
}

Graph Operator::GetSubgraph(const std::string &name) const
{
    UNUSED_VALUE(name);
    return Graph();
}

Graph Operator::GetSubgraph(const char_t *name) const
{
    UNUSED_VALUE(name);
    return Graph();
}

Graph Operator::GetDynamicSubgraph(const std::string &name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return Graph();
}

Graph Operator::GetDynamicSubgraph(const char_t *name, uint32_t index) const
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(index);
    return Graph();
}

size_t Operator::GetSubgraphNamesCount() const
{
    return 0UL;
}

void Operator::DynamicInputRegister(const char_t *name, const uint32_t num, const char_t *datatype_symbol,
                                    bool is_push_back)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(datatype_symbol);
    UNUSED_VALUE(is_push_back);
}

void Operator::DynamicOutputRegister(const char_t *name, const uint32_t num, const char_t *datatype_symbol,
                                     bool is_push_back)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(num);
    UNUSED_VALUE(datatype_symbol);
    UNUSED_VALUE(is_push_back);
}

static inline bool HasSameNameNode(const ComputeGraphPtr &compute_graph)
{
    UNUSED_VALUE(compute_graph);
    return true;
}

ComputeGraphPtr GraphUtilsEx::CreateGraphFromOperator(const std::string &name, const std::vector<ge::Operator> &inputs)
{
    UNUSED_VALUE(name);
    UNUSED_VALUE(inputs);
    return nullptr;
}

void GraphUtilsEx::BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos)
{
    UNUSED_VALUE(all_nodes_infos);
}
} // namespace ge

