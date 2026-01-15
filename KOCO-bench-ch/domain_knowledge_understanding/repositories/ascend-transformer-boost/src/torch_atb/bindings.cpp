/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#pragma GCC diagnostic pop
#include <sstream>
#include <atb/utils/param_to_json.h>
#include "graph_operation_builder.h"
#include "enger_graph_builder.h"
#include "graph_node.h"
#include "resource/memory_manager.h"
#include "prof/prof_stats.h"
#include "operation_wrapper.h"

namespace py = pybind11;
using namespace atb;
using namespace atb::infer;

template <typename Container, typename MethodType>
void AddElements(Container &container, MethodType &method)
{
    for (const auto &element : container) {
        method.push_back(element);
    }
}

PYBIND11_MODULE(_C, m)
{
    m.doc() = "Python bindings for torch_atb";

    m.def("set_buffer_size", static_cast<void(*)(uint64_t)>(&TorchAtb::MemoryManager::SetBufferSize),
          py::arg("bytes"), "Set default workspace buffer size (bytes)");

    py::class_<TorchAtb::ProfStats>(m, "Prof")
        .def_static("get_prof_stats", &TorchAtb::ProfStats::GetProfStats, py::return_value_policy::reference)
        .def("get_run_time_stats", &TorchAtb::ProfStats::GetRunTimeStats);

    py::class_<TorchAtb::OperationWrapper>(m, "Operation")
        .def(py::init<const LayerNormParam &>())
        .def(py::init<const ElewiseParam &>())
        .def(py::init<const LinearParam &>())
        .def(py::init<const SoftmaxParam &>())
        .def(py::init<const SelfAttentionParam &>())
        .def(py::init<const PagedAttentionParam &>())
        .def(py::init<const RopeParam &>())
        .def(py::init<const SplitParam &>())
        .def(py::init<const GatherParam &>())
        .def(py::init<const ActivationParam &>())
        .def(py::init<const RmsNormParam &>())
        .def(py::init<const AllGatherParam &>())
        .def(py::init<const AsStridedParam &>())
        .def(py::init<const CumsumParam &>())
        .def(py::init<const DynamicNTKParam &>())
        .def(py::init<const MultinomialParam &>())
        .def(py::init<const ConcatParam &>())
        .def(py::init<const SliceParam &>())
        .def(py::init<const TransposeParam &>())
        .def(py::init<const GatingParam &>())
        .def(py::init<const ReshapeAndCacheParam &>())
        .def(py::init<const FillParam &>())
        .def(py::init<const RazorFusionAttentionParam &>())
        .def(py::init<const AllReduceParam &>())
        .def(py::init<const BroadcastParam &>())
        .def(py::init<const ReduceScatterParam &>())
        .def(py::init<const ReduceScatterVParam &>())
        .def(py::init<const FaUpdateParam &>())
        .def(py::init<const LinearParallelParam &>())
        .def(py::init<const LinearSparseParam &>())
        .def(py::init<const RelayAttentionParam &>())
        .def(py::init<const TopkToppSamplingParam &>())
        .def(py::init<const AllToAllParam &>())
        .def(py::init<const GraphParam &>())
        .def_property_readonly("name", &TorchAtb::OperationWrapper::GetName)
        .def_property_readonly("input_num", &TorchAtb::OperationWrapper::GetInputNum)
        .def_property_readonly("output_num", &TorchAtb::OperationWrapper::GetOutputNum)
        .def("forward", &TorchAtb::OperationWrapper::Forward)
        .def("__repr__", [](const TorchAtb::OperationWrapper &opWrapper) {
            std::stringstream ss;
            ss << "op name: " << opWrapper.GetName() << ", input_num: " << opWrapper.GetInputNum()
               << ", output_num: " << opWrapper.GetOutputNum();
            return ss.str();
        });

    py::class_<TorchAtb::GraphNode>(m, "Node")
        .def("set_op", &TorchAtb::GraphNode::SetOperation)
        .def("get_output", &TorchAtb::GraphNode::GetOutput)
        .def("set_stream_id", &TorchAtb::GraphNode::SetStreamId)
        .def("get_stream_id", &TorchAtb::GraphNode::GetStreamId)
        .def_readonly("inTensorIds", &TorchAtb::GraphNode::inTensorIds)
        .def_readonly("outTensorIds", &TorchAtb::GraphNode::outTensorIds)
        .def_readonly("operation", &TorchAtb::GraphNode::operation);
        
    py::class_<TorchAtb::GraphBuilder>(m, "Builder")
        .def(py::init<const std::string &>())
        .def("add_input", &TorchAtb::GraphBuilder::AddInput)
        .def("add_node",
             py::overload_cast<const std::vector<std::string> &, atb::Operation *>(&TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::LinearParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::LayerNormParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::ElewiseParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::SoftmaxParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::SelfAttentionParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::RopeParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::SplitParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::GatherParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::ActivationParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::RmsNormParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::AllGatherParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::AsStridedParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::CumsumParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::DynamicNTKParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::MultinomialParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::ConcatParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::SliceParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::TransposeParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::GatingParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::ReshapeAndCacheParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::FillParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node",
             py::overload_cast<const std::vector<std::string> &, const atb::infer::RazorFusionAttentionParam &>(
                 &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::AllReduceParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::BroadcastParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::ReduceScatterParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::ReduceScatterVParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::LinearParallelParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::LinearSparseParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::RelayAttentionParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::TopkToppSamplingParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, const atb::infer::AllToAllParam &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("add_node", py::overload_cast<const std::vector<std::string> &, TorchAtb::OperationWrapper &>(
                             &TorchAtb::GraphBuilder::AddNode))
        .def("reshape", &TorchAtb::GraphBuilder::Reshape)
        .def("mark_output", &TorchAtb::GraphBuilder::MarkOutput)
        .def("set_execute_streams", &TorchAtb::GraphBuilder::SetExecuteStreams)
        .def("build", &TorchAtb::GraphBuilder::Build);

    py::class_<TorchAtb::GraphOperationBuilder>(m, "GraphBuilder")
        .def(py::init<const std::string &>())
        .def("set_input_output", &TorchAtb::GraphOperationBuilder::SetInputOutput)
        .def("reshape", &TorchAtb::GraphOperationBuilder::Reshape)
        .def("add_operation", &TorchAtb::GraphOperationBuilder::AddOperation)
        .def("build", &TorchAtb::GraphOperationBuilder::Build);

    py::enum_<aclDataType>(m, "AclDataType")
        .value("ACL_DT_UNDEFINED", aclDataType::ACL_DT_UNDEFINED)
        .value("ACL_FLOAT", aclDataType::ACL_FLOAT)
        .value("ACL_FLOAT16", aclDataType::ACL_FLOAT16)
        .value("ACL_INT8", aclDataType::ACL_INT8)
        .value("ACL_INT32", aclDataType::ACL_INT32)
        .value("ACL_UINT8", aclDataType::ACL_UINT8)
        .value("ACL_INT16", aclDataType::ACL_INT16)
        .value("ACL_UINT16", aclDataType::ACL_UINT16)
        .value("ACL_UINT32", aclDataType::ACL_UINT32)
        .value("ACL_INT64", aclDataType::ACL_INT64)
        .value("ACL_UINT64", aclDataType::ACL_UINT64)
        .value("ACL_DOUBLE", aclDataType::ACL_DOUBLE)
        .value("ACL_BOOL", aclDataType::ACL_BOOL)
        .value("ACL_STRING", aclDataType::ACL_STRING)
        .value("ACL_COMPLEX64", aclDataType::ACL_COMPLEX64)
        .value("ACL_COMPLEX128", aclDataType::ACL_COMPLEX128)
        .value("ACL_BF16", aclDataType::ACL_BF16)
        .value("ACL_INT4", aclDataType::ACL_INT4)
        .value("ACL_UINT1", aclDataType::ACL_UINT1)
        .value("ACL_COMPLEX32", aclDataType::ACL_COMPLEX32)
        .export_values();

    py::enum_<aclFormat>(m, "AclFormat")
        .value("ACL_FORMAT_UNDEFINED", ACL_FORMAT_UNDEFINED)
        .value("ACL_FORMAT_NCHW", ACL_FORMAT_NCHW)
        .value("ACL_FORMAT_NHWC", ACL_FORMAT_NHWC)
        .value("ACL_FORMAT_ND", ACL_FORMAT_ND)
        .value("ACL_FORMAT_NC1HWC0", ACL_FORMAT_NC1HWC0)
        .value("ACL_FORMAT_FRACTAL_Z", ACL_FORMAT_FRACTAL_Z)
        .value("ACL_FORMAT_NC1HWC0_C04", ACL_FORMAT_NC1HWC0_C04)
        .value("ACL_FORMAT_HWCN", ACL_FORMAT_HWCN)
        .value("ACL_FORMAT_NDHWC", ACL_FORMAT_NDHWC)
        .value("ACL_FORMAT_FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ)
        .value("ACL_FORMAT_NCDHW", ACL_FORMAT_NCDHW)
        .value("ACL_FORMAT_NDC1HWC0", ACL_FORMAT_NDC1HWC0)
        .value("ACL_FRACTAL_Z_3D", ACL_FRACTAL_Z_3D)
        .export_values();

    py::enum_<InputLayout>(m, "InputLayout")
        .value("TYPE_BSND", InputLayout::TYPE_BSND)
        .value("TYPE_BNSD", InputLayout::TYPE_BNSD)
        .export_values();

    py::enum_<QuantType>(m, "QuantType")
        .value("QUANT_UNQUANT", QuantType::QUANT_UNQUANT)
        .value("QUANT_INT4", QuantType::QUANT_INT4)
        .value("QUANT_INT8", QuantType::QUANT_INT8)
        .value("QUANT_INT16", QuantType::QUANT_INT16)
        .value("QUANT_FLOAT8", QuantType::QUANT_FLOAT8)
        .value("QUANT_FLOAT16", QuantType::QUANT_FLOAT16)
        .export_values();

    py::enum_<DynamicQuantType>(m, "DynamicQuantType")
        .value("DYNAMIC_QUANT_UNDEFINED", DynamicQuantType::DYNAMIC_QUANT_UNDEFINED)
        .value("DYNAMIC_QUANT_SYMMETRIC", DynamicQuantType::DYNAMIC_QUANT_SYMMETRIC)
        .value("DYNAMIC_QUANT_ASYMMETRIC", DynamicQuantType::DYNAMIC_QUANT_ASYMMETRIC)
        .export_values();

    py::enum_<ActivationType>(m, "ActivationType")
        .value("ACTIVATION_UNDEFINED", ACTIVATION_UNDEFINED)
        .value("ACTIVATION_RELU", ACTIVATION_RELU)
        .value("ACTIVATION_GELU", ACTIVATION_GELU)
        .value("ACTIVATION_FAST_GELU", ACTIVATION_FAST_GELU)
        .value("ACTIVATION_SWISH", ACTIVATION_SWISH)
        .value("ACTIVATION_LOG", ACTIVATION_LOG)
        .value("ACTIVATION_SWIGLU_FORWARD", ACTIVATION_SWIGLU_FORWARD)
        .value("ACTIVATION_SWIGLU_BACKWARD", ACTIVATION_SWIGLU_BACKWARD)
        .value("ACTIVATION_SIGMOID", ACTIVATION_SIGMOID)
        .value("ACTIVATION_FASTER_GELU_FORWARD", ACTIVATION_FASTER_GELU_FORWARD)
        .value("ACTIVATION_MAX", ACTIVATION_MAX)
        .export_values();

    py::enum_<CommMode>(m, "CommMode")
        .value("COMM_UNDEFINED", COMM_UNDEFINED)
        .value("COMM_MULTI_PROCESS", COMM_MULTI_PROCESS)
        .value("COMM_MULTI_THREAD", COMM_MULTI_THREAD)
        .export_values();

    py::class_<LayerNormParam> layerNorm(m, "LayerNormParam");

    py::enum_<LayerNormParam::LayerNormType>(layerNorm, "LayerNormType")
        .value("LAYER_NORM_UNDEFINED", LayerNormParam::LayerNormType::LAYER_NORM_UNDEFINED)
        .value("LAYER_NORM_NORM", LayerNormParam::LayerNormType::LAYER_NORM_NORM)
        .value("LAYER_NORM_PRENORM", LayerNormParam::LayerNormType::LAYER_NORM_PRENORM)
        .value("LAYER_NORM_POSTNORM", LayerNormParam::LayerNormType::LAYER_NORM_POSTNORM)
        .value("LAYER_NORM_MAX", LayerNormParam::LayerNormType::LAYER_NORM_MAX);

    py::class_<LayerNormParam::NormParam>(layerNorm, "NormParam")
        .def(py::init<QuantType, float, int32_t, int32_t, DynamicQuantType>(),
             py::arg("quant_type") = QuantType::QUANT_UNQUANT,
             py::arg("epsilon") = 1e-5,
             py::arg("begin_norm_axis") = 0,
             py::arg("begin_params_axis") = 0,
             py::arg("dynamic_quant_type") = DynamicQuantType::DYNAMIC_QUANT_UNDEFINED)
        .def_readwrite("quant_type", &LayerNormParam::NormParam::quantType)
        .def_readwrite("epsilon", &LayerNormParam::NormParam::epsilon)
        .def_readwrite("begin_norm_axis", &LayerNormParam::NormParam::beginNormAxis)
        .def_readwrite("begin_params_axis", &LayerNormParam::NormParam::beginParamsAxis)
        .def_readwrite("dynamic_quant_type", &LayerNormParam::NormParam::dynamicQuantType);

    py::class_<LayerNormParam::PreNormParam>(layerNorm, "PreNormParam")
        .def(py::init<QuantType, float, uint64_t, float>(),
             py::arg("quant_type") = QuantType::QUANT_UNQUANT,
             py::arg("epsilon") = 1e-5,
             py::arg("op_mode") = 0,
             py::arg("zoom_scale_value") = 1.0f)
        .def_readwrite("quant_type", &LayerNormParam::PreNormParam::quantType)
        .def_readwrite("epsilon", &LayerNormParam::PreNormParam::epsilon)
        .def_readwrite("op_mode", &LayerNormParam::PreNormParam::opMode)
        .def_readwrite("zoom_scale_value", &LayerNormParam::PreNormParam::zoomScaleValue);

    py::class_<LayerNormParam::PostNormParam>(layerNorm, "PostNormParam")
        .def(py::init<QuantType, float, uint64_t, float>(),
             py::arg("quant_type") = QuantType::QUANT_UNQUANT,
             py::arg("epsilon") = 1e-5,
             py::arg("op_mode") = 0,
             py::arg("zoom_scale_value") = 1.0f)
        .def_readwrite("quant_type", &LayerNormParam::PostNormParam::quantType)
        .def_readwrite("epsilon", &LayerNormParam::PostNormParam::epsilon)
        .def_readwrite("op_mode", &LayerNormParam::PostNormParam::opMode)
        .def_readwrite("zoom_scale_value", &LayerNormParam::PostNormParam::zoomScaleValue);

    layerNorm
        .def(py::init<LayerNormParam::LayerNormType, LayerNormParam::NormParam, LayerNormParam::PreNormParam,
                      LayerNormParam::PostNormParam>(),
             py::arg("layer_type") = LayerNormParam::LayerNormType::LAYER_NORM_UNDEFINED,
             py::arg("norm_param") = LayerNormParam::NormParam(),
             py::arg("pre_norm_param") = LayerNormParam::PreNormParam(),
             py::arg("post_norm_param") = LayerNormParam::PostNormParam())
        .def_readwrite("layer_type", &LayerNormParam::layerType)
        .def_readwrite("norm_param", &LayerNormParam::normParam)
        .def_readwrite("pre_norm_param", &LayerNormParam::preNormParam)
        .def_readwrite("post_norm_param", &LayerNormParam::postNormParam)
        .def("__repr__", [](const LayerNormParam &param) { return "LayerNormParam: " + OpParamToJson(param).dump(); });

    py::class_<ElewiseParam> elewise(m, "ElewiseParam");

    py::enum_<ElewiseParam::ElewiseType>(elewise, "ElewiseType")
        .value("ELEWISE_UNDEFINED", ElewiseParam::ElewiseType::ELEWISE_UNDEFINED)
        .value("ELEWISE_CAST", ElewiseParam::ElewiseType::ELEWISE_CAST)
        .value("ELEWISE_MULS", ElewiseParam::ElewiseType::ELEWISE_MULS)
        .value("ELEWISE_COS", ElewiseParam::ElewiseType::ELEWISE_COS)
        .value("ELEWISE_SIN", ElewiseParam::ElewiseType::ELEWISE_SIN)
        .value("ELEWISE_NEG", ElewiseParam::ElewiseType::ELEWISE_NEG)
        .value("ELEWISE_QUANT", ElewiseParam::ElewiseType::ELEWISE_QUANT)
        .value("ELEWISE_LOGICAL_NOT", ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT)
        .value("ELEWISE_ADD", ElewiseParam::ElewiseType::ELEWISE_ADD)
        .value("ELEWISE_MUL", ElewiseParam::ElewiseType::ELEWISE_MUL)
        .value("ELEWISE_REALDIV", ElewiseParam::ElewiseType::ELEWISE_REALDIV)
        .value("ELEWISE_LOGICAL_AND", ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND)
        .value("ELEWISE_LOGICAL_OR", ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR)
        .value("ELEWISE_LESS", ElewiseParam::ElewiseType::ELEWISE_LESS)
        .value("ELEWISE_GREATER", ElewiseParam::ElewiseType::ELEWISE_GREATER)
        .value("ELEWISE_SUB", ElewiseParam::ElewiseType::ELEWISE_SUB)
        .value("ELEWISE_EQUAL", ElewiseParam::ElewiseType::ELEWISE_EQUAL)
        .value("ELEWISE_QUANT_PER_CHANNEL", ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL)
        .value("ELEWISE_DEQUANT_PER_CHANNEL", ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL)
        .value("ELEWISE_DYNAMIC_QUANT", ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT)
        .value("ELEWISE_TANH", ElewiseParam::ElewiseType::ELEWISE_TANH);

    py::class_<ElewiseParam::QuantParam>(elewise, "QuantParam")
        .def(py::init<float, bool, int>(),
             py::arg("input_scale") = 1.0f,
             py::arg("asymmetric") = false,
             py::arg("input_offset") = 0)
        .def_readwrite("input_scale", &ElewiseParam::QuantParam::inputScale)
        .def_readwrite("asymmetric", &ElewiseParam::QuantParam::asymmetric)
        .def_readwrite("input_offset", &ElewiseParam::QuantParam::inputOffset);

    py::class_<ElewiseParam::MulsParam>(elewise, "MulsParam")
        .def(py::init<float>(),
             py::arg("var_attr") = 0.0f)
        .def_readwrite("var_attr", &ElewiseParam::MulsParam::varAttr);

    elewise
        .def(py::init<ElewiseParam::ElewiseType, ElewiseParam::QuantParam, ElewiseParam::MulsParam, aclDataType>(),
             py::arg("elewise_type") = ElewiseParam::ElewiseType::ELEWISE_UNDEFINED,
             py::arg("quant_param") = ElewiseParam::QuantParam(),
             py::arg("muls_param") = ElewiseParam::MulsParam(),
             py::arg("out_tensor_type") = aclDataType::ACL_DT_UNDEFINED)
        .def_readwrite("elewise_type", &ElewiseParam::elewiseType)
        .def_readwrite("quant_param", &ElewiseParam::quantParam)
        .def_readwrite("muls_param", &ElewiseParam::mulsParam)
        .def_readwrite("out_tensor_type", &ElewiseParam::outTensorType)
        .def("__repr__", [](const ElewiseParam &param) { return "ElewiseParam: " + OpParamToJson(param).dump(); });

    py::class_<LinearParam> linear(m, "LinearParam");

    py::enum_<LinearParam::MatmulType>(linear, "MatmulType")
        .value("MATMUL_UNDEFINED", LinearParam::MatmulType::MATMUL_UNDEFINED)
        .value("MATMUL_EIN_SUM", LinearParam::MatmulType::MATMUL_EIN_SUM);
    py::enum_<LinearParam::QuantMode>(linear, "QuantMode")
        .value("QUANT_UNDEFINED", LinearParam::QuantMode::QUANT_UNDEFINED)
        .value("PER_CHANNEL", LinearParam::QuantMode::PER_CHANNEL)
        .value("PER_TOKEN", LinearParam::QuantMode::PER_TOKEN);

    linear
        .def(py::init<bool, bool, bool, aclDataType, bool, LinearParam::MatmulType, LinearParam::QuantMode>(),
             py::arg("transpose_a") = false,
             py::arg("transpose_b") = true,
             py::arg("has_bias") = true,
             py::arg("out_data_type") = aclDataType::ACL_DT_UNDEFINED,
             py::arg("en_accum") = false,
             py::arg("matmul_type") = LinearParam::MatmulType::MATMUL_UNDEFINED,
             py::arg("quant_mode") = LinearParam::QuantMode::QUANT_UNDEFINED)
        .def_readwrite("transpose_a", &LinearParam::transposeA)
        .def_readwrite("transpose_b", &LinearParam::transposeB)
        .def_readwrite("has_bias", &LinearParam::hasBias)
        .def_readwrite("out_data_type", &LinearParam::outDataType)
        .def_readwrite("en_accum", &LinearParam::enAccum)
        .def_readwrite("matmul_type", &LinearParam::matmulType)
        .def_readwrite("quant_mode", &LinearParam::quantMode)
        .def("__repr__", [](const LinearParam &param) { return "LinearParam: " + OpParamToJson(param).dump(); });

    py::class_<SoftmaxParam>(m, "SoftmaxParam")
        .def(py::init([](const std::vector<int64_t> &axes) {
                 SoftmaxParam param;
                 AddElements(axes, param.axes);
                 return param;
                 }),
             py::arg("axes") = py::list())
        .def_readwrite("axes", &SoftmaxParam::axes)
        .def_property(
            "axes",
            [](SoftmaxParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> axisVec;
                axisVec.resize(param.axes.size());
                for (size_t i = 0; i < param.axes.size(); i++) {
                    axisVec.at(i) = param.axes.at(i);
                }
                return axisVec;
            },
            [](SoftmaxParam &param, const std::vector<int64_t> &axes) {
                param.axes.clear();
                AddElements(axes, param.axes);
            })
        .def("__repr__", [](const SoftmaxParam &param) { return "SoftmaxParam: " + OpParamToJson(param).dump(); });

    py::class_<SelfAttentionParam> selfAttention(m, "SelfAttentionParam");

    py::enum_<SelfAttentionParam::CalcType>(selfAttention, "CalcType")
        .value("UNDEFINED", SelfAttentionParam::UNDEFINED)
        .value("ENCODER", SelfAttentionParam::ENCODER)
        .value("DECODER", SelfAttentionParam::DECODER)
        .value("PA_ENCODER", SelfAttentionParam::PA_ENCODER)
        .value("PREFIX_ENCODER", SelfAttentionParam::PREFIX_ENCODER);

    py::enum_<SelfAttentionParam::KernelType>(selfAttention, "KernelType")
        .value("KERNELTYPE_DEFAULT", SelfAttentionParam::KERNELTYPE_DEFAULT)
        .value("KERNELTYPE_HIGH_PRECISION", SelfAttentionParam::KERNELTYPE_HIGH_PRECISION);

    py::enum_<SelfAttentionParam::ClampType>(selfAttention, "ClampType")
        .value("CLAMP_TYPE_UNDEFINED", SelfAttentionParam::CLAMP_TYPE_UNDEFINED)
        .value("CLAMP_TYPE_MIN_MAX", SelfAttentionParam::CLAMP_TYPE_MIN_MAX);

    py::enum_<SelfAttentionParam::MaskType>(selfAttention, "MaskType")
        .value("MASK_TYPE_UNDEFINED", SelfAttentionParam::MASK_TYPE_UNDEFINED)
        .value("MASK_TYPE_NORM", SelfAttentionParam::MASK_TYPE_NORM)
        .value("MASK_TYPE_ALIBI", SelfAttentionParam::MASK_TYPE_ALIBI)
        .value("MASK_TYPE_NORM_COMPRESS", SelfAttentionParam::MASK_TYPE_NORM_COMPRESS)
        .value("MASK_TYPE_ALIBI_COMPRESS", SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS)
        .value("MASK_TYPE_ALIBI_COMPRESS_SQRT", SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT)
        .value("MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN", SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN)
        .value("MASK_TYPE_SLIDING_WINDOW_NORM", SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM)
        .value("MASK_TYPE_SLIDING_WINDOW_COMPRESS", SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS);

    py::enum_<SelfAttentionParam::KvCacheCfg>(selfAttention, "KvCacheCfg")
        .value("K_CACHE_V_CACHE", SelfAttentionParam::K_CACHE_V_CACHE)
        .value("K_BYPASS_V_BYPASS", SelfAttentionParam::K_BYPASS_V_BYPASS);

    py::enum_<SelfAttentionParam::ScaleType>(selfAttention, "ScaleType")
        .value("SCALE_TYPE_TOR", SelfAttentionParam::SCALE_TYPE_TOR)
        .value("SCALE_TYPE_LOGN", SelfAttentionParam::SCALE_TYPE_LOGN)
        .value("SCALE_TYPE_MAX", SelfAttentionParam::SCALE_TYPE_MAX);

    py::enum_<SelfAttentionParam::QuantType>(selfAttention, "QuantType")
        .value("TYPE_QUANT_UNQUANT", SelfAttentionParam::TYPE_QUANT_UNQUANT)
        .value("TYPE_DEQUANT_FUSION", SelfAttentionParam::TYPE_DEQUANT_FUSION)
        .value("TYPE_QUANT_QKV_OFFLINE", SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE)
        .value("TYPE_QUANT_QKV_ONLINE", SelfAttentionParam::TYPE_QUANT_QKV_ONLINE);
    
    py::enum_<SelfAttentionParam::CacheType>(selfAttention, "CacheType")
        .value("CACHE_TYPE_NORM", SelfAttentionParam::CACHE_TYPE_NORM)
        .value("CACHE_TYPE_SWA", SelfAttentionParam::CACHE_TYPE_SWA);

    selfAttention
        .def(py::init<SelfAttentionParam::QuantType, aclDataType, int32_t, int32_t, float, float, bool, uint32_t,
                      SelfAttentionParam::CalcType, SelfAttentionParam::KernelType, SelfAttentionParam::ClampType,
                      float, float, SelfAttentionParam::MaskType, SelfAttentionParam::KvCacheCfg,
                      SelfAttentionParam::ScaleType, InputLayout, uint32_t, SelfAttentionParam::CacheType, uint32_t>(),
             py::arg("quant_type") = SelfAttentionParam::QuantType::TYPE_QUANT_UNQUANT,
             py::arg("out_data_type") = aclDataType::ACL_DT_UNDEFINED,
             py::arg("head_num") = 0,
             py::arg("kv_head_num") = 0,
             py::arg("q_scale") = 1,
             py::arg("qk_scale") = 1,
             py::arg("batch_run_status_enable") = false,
             py::arg("is_triu_mask") = 0,
             py::arg("calc_type") = SelfAttentionParam::CalcType::UNDEFINED,
             py::arg("kernel_type") = SelfAttentionParam::KernelType::KERNELTYPE_DEFAULT,
             py::arg("clamp_type") = SelfAttentionParam::ClampType::CLAMP_TYPE_UNDEFINED,
             py::arg("clamp_min") = 0,
             py::arg("clamp_max") = 0,
             py::arg("mask_type") = SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED,
             py::arg("kvcache_cfg") = SelfAttentionParam::KvCacheCfg::K_CACHE_V_CACHE,
             py::arg("scale_type") = SelfAttentionParam::ScaleType::SCALE_TYPE_TOR,
             py::arg("input_layout") = InputLayout::TYPE_BSND,
             py::arg("mla_v_head_size") = 0,
             py::arg("cache_type") = SelfAttentionParam::CacheType::CACHE_TYPE_NORM,
             py::arg("window_size") = 0)
        .def_readwrite("quant_type", &SelfAttentionParam::quantType)
        .def_readwrite("out_data_type", &SelfAttentionParam::outDataType)
        .def_readwrite("head_num", &SelfAttentionParam::headNum)
        .def_readwrite("kv_head_num", &SelfAttentionParam::kvHeadNum)
        .def_readwrite("q_scale", &SelfAttentionParam::qScale)
        .def_readwrite("qk_scale", &SelfAttentionParam::qkScale)
        .def_readwrite("batch_run_status_enable", &SelfAttentionParam::batchRunStatusEnable)
        .def_readwrite("is_triu_mask", &SelfAttentionParam::isTriuMask)
        .def_readwrite("calc_type", &SelfAttentionParam::calcType)
        .def_readwrite("kernel_type", &SelfAttentionParam::kernelType)
        .def_readwrite("clamp_type", &SelfAttentionParam::clampType)
        .def_readwrite("clamp_min", &SelfAttentionParam::clampMin)
        .def_readwrite("clamp_max", &SelfAttentionParam::clampMax)
        .def_readwrite("mask_type", &SelfAttentionParam::maskType)
        .def_readwrite("kvcache_cfg", &SelfAttentionParam::kvcacheCfg)
        .def_readwrite("scale_type", &SelfAttentionParam::scaleType)
        .def_readwrite("input_layout", &SelfAttentionParam::inputLayout)
        .def_readwrite("mla_v_head_size", &SelfAttentionParam::mlaVHeadSize)
        .def_readwrite("cache_type", &SelfAttentionParam::cacheType)
        .def_readwrite("window_size", &SelfAttentionParam::windowSize)
        .def("__repr__",
             [](const SelfAttentionParam &param) { return "SelfAttentionParam: " + OpParamToJson(param).dump(); });

    py::class_<PagedAttentionParam> pageAttention(m, "PagedAttentionParam");

    py::enum_<PagedAttentionParam::CalcType>(pageAttention, "CalcType")
        .value("CALC_TYPE_UNDEFINED", PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED)
        .value("CALC_TYPE_SPEC", PagedAttentionParam::CalcType::CALC_TYPE_SPEC);

    py::enum_<PagedAttentionParam::CompressType>(pageAttention, "CompressType")
        .value("COMPRESS_TYPE_UNDEFINED", PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED)
        .value("COMPRESS_TYPE_KVHEAD", PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD)
        .value("COMPRESS_TYPE_KVHEAD_ROPE", PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE)
        .value("COMPRESS_TYPE_MAX", PagedAttentionParam::CompressType::COMPRESS_TYPE_MAX);

    py::enum_<PagedAttentionParam::MaskType>(pageAttention, "MaskType")
        .value("UNDEFINED", PagedAttentionParam::MaskType::UNDEFINED)
        .value("MASK_TYPE_NORM", PagedAttentionParam::MaskType::MASK_TYPE_NORM)
        .value("MASK_TYPE_ALIBI", PagedAttentionParam::MaskType::MASK_TYPE_ALIBI)
        .value("MASK_TYPE_SPEC", PagedAttentionParam::MaskType::MASK_TYPE_SPEC);

    py::enum_<PagedAttentionParam::ScaleType>(pageAttention, "ScaleType")
        .value("SCALE_TYPE_TOR", PagedAttentionParam::ScaleType::SCALE_TYPE_TOR)
        .value("SCALE_TYPE_LOGN", PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN)
        .value("SCALE_TYPE_MAX", PagedAttentionParam::ScaleType::SCALE_TYPE_MAX);

    py::enum_<PagedAttentionParam::QuantType>(pageAttention, "QuantType")
        .value("TYPE_QUANT_UNQUANT", PagedAttentionParam::QuantType::TYPE_QUANT_UNQUANT)
        .value("TYPE_DEQUANT_FUSION", PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION)
        .value("TYPE_QUANT_QKV_OFFLINE", PagedAttentionParam::QuantType::TYPE_QUANT_QKV_OFFLINE)
        .value("TYPE_QUANT_QKV_ONLINE", PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE);

    pageAttention
        .def(py::init<int32_t, float, int32_t, PagedAttentionParam::MaskType, bool, PagedAttentionParam::QuantType,
                      aclDataType, bool, PagedAttentionParam::CompressType, PagedAttentionParam::CalcType,
                      PagedAttentionParam::ScaleType, InputLayout>(),
             py::arg("head_num") = 0,
             py::arg("qk_scale") = 1.0,
             py::arg("kv_head_num") = 0,
             py::arg("mask_type") = PagedAttentionParam::MaskType::UNDEFINED,
             py::arg("batch_run_status_enable") = false,
             py::arg("quant_type") = PagedAttentionParam::QuantType::TYPE_QUANT_UNQUANT,
             py::arg("out_data_type") = aclDataType::ACL_DT_UNDEFINED,
             py::arg("has_quant_offset") = false,
             py::arg("compress_type") = PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED,
             py::arg("calc_type") = PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED,
             py::arg("scale_type") = PagedAttentionParam::ScaleType::SCALE_TYPE_TOR,
             py::arg("input_layout") = InputLayout::TYPE_BSND)
        .def_readwrite("head_num", &PagedAttentionParam::headNum)
        .def_readwrite("qk_scale", &PagedAttentionParam::qkScale)
        .def_readwrite("kv_head_num", &PagedAttentionParam::kvHeadNum)
        .def_readwrite("mask_type", &PagedAttentionParam::maskType)
        .def_readwrite("batch_run_status_enable", &PagedAttentionParam::batchRunStatusEnable)
        .def_readwrite("quant_type", &PagedAttentionParam::quantType)
        .def_readwrite("out_data_type", &PagedAttentionParam::outDataType)
        .def_readwrite("has_quant_offset", &PagedAttentionParam::hasQuantOffset)
        .def_readwrite("compress_type", &PagedAttentionParam::compressType)
        .def_readwrite("calc_type", &PagedAttentionParam::calcType)
        .def_readwrite("scale_type", &PagedAttentionParam::scaleType)
        .def_readwrite("input_layout", &PagedAttentionParam::inputLayout)
        .def("__repr__",
             [](const PagedAttentionParam &param) { return "PagedAttentionParam: " + OpParamToJson(param).dump(); });

    py::class_<RopeParam>(m, "RopeParam")
        .def(py::init<int32_t, int32_t>(),
             py::arg("rotary_coeff") = 4,
             py::arg("cos_format") = 0)
        .def_readwrite("rotary_coeff", &RopeParam::rotaryCoeff)
        .def_readwrite("cos_format", &RopeParam::cosFormat)
        .def("__repr__", [](const RopeParam &param) { return "RopeParam: " + OpParamToJson(param).dump(); });

    py::class_<SplitParam>(m, "SplitParam")
        .def(py::init([](int32_t splitDim, int32_t splitNum, const std::vector<int32_t> &splitSizes) {
                 SplitParam param;
                 param.splitDim = splitDim;
                 param.splitNum = splitNum;
                 AddElements(splitSizes, param.splitSizes);
                 return param;
                 }),
             py::arg("split_dim") = 0,
             py::arg("split_num") = 2,
             py::arg("split_sizes") = py::list())
        .def_readwrite("split_dim", &SplitParam::splitDim)
        .def_readwrite("split_num", &SplitParam::splitNum)
        .def_property(
            "split_sizes",
            [](SplitParam &param) -> std::vector<int32_t> {
                std::vector<int32_t> vec;
                vec.resize(param.splitSizes.size());
                for (size_t i = 0; i < param.splitSizes.size(); i++) {
                    vec.at(i) = param.splitSizes.at(i);
                }
                return vec;
            },
            [](SplitParam &param, const std::vector<int32_t> &splitSizes) {
                param.splitSizes.clear();
                AddElements(splitSizes, param.splitSizes);
            })
        .def("__repr__", [](const SplitParam &param) { return "SplitParam: " + OpParamToJson(param).dump(); });

    py::class_<GatherParam>(m, "GatherParam")
        .def(py::init<int64_t, int64_t>(),
             py::arg("axis") = 0,
             py::arg("batch_dims") = 0)
        .def_readwrite("axis", &GatherParam::axis)
        .def_readwrite("batch_dims", &GatherParam::batchDims)
        .def("__repr__", [](const GatherParam &param) { return "GatherParam: " + OpParamToJson(param).dump(); });

    py::class_<ActivationParam> activation(m, "ActivationParam");
        
    py::enum_<ActivationParam::GeLUMode>(activation, "GeLUMode")
        .value("TANH_MODE", ActivationParam::TANH_MODE)
        .value("NONE_MODE", ActivationParam::NONE_MODE);

    activation
        .def(py::init<ActivationType, float, int32_t, ActivationParam::GeLUMode>(),
             py::arg("activation_type") = ActivationType::ACTIVATION_UNDEFINED,
             py::arg("scale") = 1.0f,
             py::arg("dim") = -1,
             py::arg("gelu_mode") = ActivationParam::GeLUMode::TANH_MODE)
        .def_readwrite("activation_type", &ActivationParam::activationType)
        .def_readwrite("scale", &ActivationParam::scale)
        .def_readwrite("dim", &ActivationParam::dim)
        .def_readwrite("gelu_mode", &ActivationParam::geluMode)
        .def("__repr__",
             [](const ActivationParam &param) { return "ActivationParam: " + OpParamToJson(param).dump(); });

    py::class_<RmsNormParam> rmsNorm(m, "RmsNormParam");

    py::enum_<RmsNormParam::RmsNormType>(rmsNorm, "RmsNormType")
        .value("RMS_NORM_UNDEFINED", RmsNormParam::RMS_NORM_UNDEFINED)
        .value("RMS_NORM_NORM", RmsNormParam::RMS_NORM_NORM)
        .value("RMS_NORM_PRENORM", RmsNormParam::RMS_NORM_PRENORM)
        .value("RMS_NORM_POSTNORM", RmsNormParam::RMS_NORM_POSTNORM);

    py::enum_<RmsNormParam::PrecisionMode>(rmsNorm, "PrecisionMode")
        .value("HIGH_PRECISION_MODE", RmsNormParam::HIGH_PRECISION_MODE)
        .value("HIGH_PERFORMANCE_MODE", RmsNormParam::HIGH_PERFORMANCE_MODE);

    py::enum_<RmsNormParam::ModelType>(rmsNorm, "ModelType")
        .value("LLAMA_MODEL", RmsNormParam::LLAMA_MODEL)
        .value("GEMMA_MODEL", RmsNormParam::GEMMA_MODEL);

    py::class_<RmsNormParam::NormParam>(rmsNorm, "NormParam")
        .def(py::init<QuantType, float, double, bool, RmsNormParam::PrecisionMode, RmsNormParam::ModelType,
                      DynamicQuantType>(),
             py::arg("quant_type") = QuantType::QUANT_UNQUANT,
             py::arg("epsilon") = 1e-5,
             py::arg("layer_norm_eps") = 1e-5,
             py::arg("rstd") = false,
             py::arg("precision_mode") = RmsNormParam::PrecisionMode::HIGH_PRECISION_MODE,
             py::arg("model_type") = RmsNormParam::ModelType::LLAMA_MODEL,
             py::arg("dynamic_quant_type") = DynamicQuantType::DYNAMIC_QUANT_UNDEFINED)
        .def_readwrite("quant_type", &RmsNormParam::NormParam::quantType)
        .def_readwrite("epsilon", &RmsNormParam::NormParam::epsilon)
        .def_readwrite("layer_norm_eps", &RmsNormParam::NormParam::layerNormEps)
        .def_readwrite("rstd", &RmsNormParam::NormParam::rstd)
        .def_readwrite("precision_mode", &RmsNormParam::NormParam::precisionMode)
        .def_readwrite("model_type", &RmsNormParam::NormParam::modelType)
        .def_readwrite("dynamic_quant_type", &RmsNormParam::NormParam::dynamicQuantType);

    py::class_<RmsNormParam::PreNormParam>(rmsNorm, "PreNormParam")
        .def(py::init<QuantType, float, bool>(),
             py::arg("quant_type") = QuantType::QUANT_UNQUANT,
             py::arg("epsilon") = 1e-5,
             py::arg("has_bias") = false)
        .def_readwrite("quant_type", &RmsNormParam::PreNormParam::quantType)
        .def_readwrite("epsilon", &RmsNormParam::PreNormParam::epsilon)
        .def_readwrite("has_bias", &RmsNormParam::PreNormParam::hasBias);

    py::class_<RmsNormParam::PostNormParam>(rmsNorm, "PostNormParam")
        .def(py::init<QuantType, float, bool>(),
             py::arg("quant_type") = QuantType::QUANT_UNQUANT,
             py::arg("epsilon") = 1e-5,
             py::arg("has_bias") = false)
        .def_readwrite("quant_type", &RmsNormParam::PostNormParam::quantType)
        .def_readwrite("epsilon", &RmsNormParam::PostNormParam::epsilon)
        .def_readwrite("has_bias", &RmsNormParam::PostNormParam::hasBias);

    rmsNorm
        .def(py::init<RmsNormParam::RmsNormType, RmsNormParam::NormParam, RmsNormParam::PreNormParam,
                      RmsNormParam::PostNormParam>(),
             py::arg("layer_type") = RmsNormParam::RmsNormType::RMS_NORM_UNDEFINED,
             py::arg("norm_param") = RmsNormParam::NormParam(),
             py::arg("pre_norm_param") = RmsNormParam::PreNormParam(),
             py::arg("post_norm_param") = RmsNormParam::PostNormParam())
        .def_readwrite("layer_type", &RmsNormParam::layerType)
        .def_readwrite("norm_param", &RmsNormParam::normParam)
        .def_readwrite("pre_norm_param", &RmsNormParam::preNormParam)
        .def_readwrite("post_norm_param", &RmsNormParam::postNormParam)
        .def("__repr__", [](const RmsNormParam &param) { return "RmsNormParam: " + OpParamToJson(param).dump(); });

    py::class_<AllGatherParam>(m, "AllGatherParam")
        .def(py::init<int, int, int, std::string, HcclComm, CommMode, std::string, std::string>(),
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("backend") = "hccl",
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("rank_table_file") = "",
             py::arg("comm_domain") = "")
        .def_readwrite("rank", &AllGatherParam::rank)
        .def_readwrite("rank_size", &AllGatherParam::rankSize)
        .def_readwrite("rank_root", &AllGatherParam::rankRoot)
        .def_readwrite("backend", &AllGatherParam::backend)
        .def_readwrite("hccl_comm", &AllGatherParam::hcclComm)
        .def_readwrite("comm_mode", &AllGatherParam::commMode)
        .def_readwrite("rank_table_file", &AllGatherParam::rankTableFile)
        .def_readwrite("comm_domain", &AllGatherParam::commDomain)
        .def("__repr__", [](const AllGatherParam &param) { return "AllGatherParam: " + OpParamToJson(param).dump(); });

    py::class_<AsStridedParam>(m, "AsStridedParam")
        .def(py::init([](const std::vector<int64_t> &size,
                         const std::vector<int64_t> &stride,
                         const std::vector<int64_t> &offset) {
                AsStridedParam param;
                AddElements(size, param.size);
                AddElements(stride, param.stride);
                AddElements(offset, param.offset);
                return param;
                }),
             py::arg("size") = py::list(),
             py::arg("stride") = py::list(),
             py::arg("offset") = py::list())
        .def_readwrite("size", &AsStridedParam::size)
        .def_readwrite("stride", &AsStridedParam::stride)
        .def_readwrite("offset", &AsStridedParam::offset)
        .def_property(
            "size",
            [](AsStridedParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> sizeVec;
                sizeVec.resize(param.size.size());
                for (size_t i = 0; i < param.size.size(); i++) {
                    sizeVec.at(i) = param.size.at(i);
                }
                return sizeVec;
            },
            [](AsStridedParam &param, const std::vector<int64_t> &size) {
                param.size.clear();
                AddElements(size, param.size);
            })
        .def_property(
            "stride",
            [](AsStridedParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> strideVec;
                strideVec.resize(param.stride.size());
                for (size_t i = 0; i < param.stride.size(); i++) {
                    strideVec.at(i) = param.stride.at(i);
                }
                return strideVec;
            },
            [](AsStridedParam &param, const std::vector<int64_t> &stride) {
                param.stride.clear();
                AddElements(stride, param.stride);
            })
        .def_property(
            "offset",
            [](AsStridedParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> offsetVec;
                offsetVec.resize(param.offset.size());
                for (size_t i = 0; i < param.offset.size(); i++) {
                    offsetVec.at(i) = param.offset.at(i);
                }
                return offsetVec;
            },
            [](AsStridedParam &param, const std::vector<int64_t> &offset) {
                param.offset.clear();
                AddElements(offset, param.offset);
            })
        .def("__repr__", [](const AsStridedParam &param) { return "AsStridedParam: " + OpParamToJson(param).dump(); });

    py::class_<CumsumParam>(m, "CumsumParam")
        .def(py::init([](const std::vector<int64_t> &axes, bool exclusive, bool reverse) {
                 CumsumParam param;
                 AddElements(axes, param.axes);
                 param.exclusive = exclusive;
                 param.reverse = reverse;
                 return param;
                 }),
             py::arg("axes") = py::list(),
             py::arg("exclusive") = false,
             py::arg("reverse") = false)
        .def_readwrite("axes", &CumsumParam::axes)
        .def_readwrite("exclusive", &CumsumParam::exclusive)
        .def_readwrite("reverse", &CumsumParam::reverse)
        .def_property(
            "axes",
            [](CumsumParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> vec;
                vec.resize(param.axes.size());
                for (size_t i = 0; i < param.axes.size(); i++) {
                    vec.at(i) = param.axes.at(i);
                }
                return vec;
            },
            [](CumsumParam &param, const std::vector<int64_t> &axes) {
                param.axes.clear();
                AddElements(axes, param.axes);
            })
        .def("__repr__", [](const CumsumParam &param) { return "CumsumParam: " + OpParamToJson(param).dump(); });

    py::class_<DynamicNTKParam>(m, "DynamicNTKParam")
        .def(py::init<aclDataType>(),
             py::arg("out_data_type") = aclDataType::ACL_DT_UNDEFINED)
        .def_readwrite("out_data_type", &DynamicNTKParam::outDataType)
        .def("__repr__",
             [](const DynamicNTKParam &param) { return "DynamicNTKParam: " + OpParamToJson(param).dump(); });

    py::class_<MultinomialParam>(m, "MultinomialParam")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("num_samples") = 1,
             py::arg("rand_seed") = 0)
        .def_readwrite("num_samples", &MultinomialParam::numSamples)
        .def_readwrite("rand_seed", &MultinomialParam::randSeed)
        .def("__repr__",
             [](const MultinomialParam &param) { return "MultinomialParam: " + OpParamToJson(param).dump(); });

    py::class_<ConcatParam>(m, "ConcatParam")
        .def(py::init<int>(),
             py::arg("concat_dim") = 0)
        .def_readwrite("concat_dim", &ConcatParam::concatDim)
        .def("__repr__", [](const ConcatParam &param) { return "ConcatParam: " + OpParamToJson(param).dump(); });

    py::class_<SliceParam>(m, "SliceParam")
        .def(py::init([](const std::vector<int64_t> &offsets, const std::vector<int64_t> &size) {
                 SliceParam param;
                 AddElements(offsets, param.offsets);
                 AddElements(size, param.size);
                 return param;
                 }),
             py::arg("offsets") = py::list(),
             py::arg("size") = py::list())
        .def_readwrite("offsets", &SliceParam::offsets)
        .def_readwrite("size", &SliceParam::size)
        .def_property(
            "offsets",
            [](SliceParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> offsetsVec;
                offsetsVec.resize(param.offsets.size());
                for (size_t i = 0; i < param.offsets.size(); i++) {
                    offsetsVec.at(i) = param.offsets.at(i);
                }
                return offsetsVec;
            },
            [](SliceParam &param, const std::vector<int64_t> &offsets) {
                param.offsets.clear();
                AddElements(offsets, param.offsets);
            })
        .def_property(
            "size",
            [](SliceParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> sizeVec;
                sizeVec.resize(param.size.size());
                for (size_t i = 0; i < param.size.size(); i++) {
                    sizeVec.at(i) = param.size.at(i);
                }
                return sizeVec;
            },
            [](SliceParam &param, const std::vector<int64_t> &size) {
                param.size.clear();
                AddElements(size, param.size);
            })
        .def("__repr__", [](const SliceParam &param) { return "SliceParam: " + OpParamToJson(param).dump(); });

    py::class_<TransposeParam>(m, "TransposeParam")
        .def(py::init([](const std::vector<int32_t> &perm) {
                 TransposeParam param;
                 AddElements(perm, param.perm);
                 return param;
                 }),
             py::arg("perm") = py::list())
        .def_readwrite("perm", &TransposeParam::perm)
        .def_property(
            "perm",
            [](TransposeParam &param) -> std::vector<int32_t> {
                std::vector<int32_t> vec;
                vec.resize(param.perm.size());
                for (size_t i = 0; i < param.perm.size(); i++) {
                    vec.at(i) = param.perm.at(i);
                }
                return vec;
            },
            [](TransposeParam &param, const std::vector<int32_t> &perm) {
                param.perm.clear();
                AddElements(perm, param.perm);
            })
        .def("__repr__", [](const TransposeParam &param) { return "TransposeParam: " + OpParamToJson(param).dump(); });

    py::class_<GatingParam>(m, "GatingParam")
        .def(py::init([](int32_t topkExpertNum, int32_t cumSumNum, bool cumSumInt64,
                         const std::vector<int32_t> &deviceExpert) {
                GatingParam param;
                param.topkExpertNum = topkExpertNum;
                param.cumSumNum = cumSumNum;
                param.cumSumInt64 = cumSumInt64;
                AddElements(deviceExpert, param.deviceExpert);
                return param;
                }),
             py::arg("topk_expert_num") = 1,
             py::arg("cum_sum_num") = 0,
             py::arg("cum_sum_int64") = false,
             py::arg("device_expert") = py::list())
        .def_readwrite("topk_expert_num", &GatingParam::topkExpertNum)
        .def_readwrite("cum_sum_num", &GatingParam::cumSumNum)
        .def_readwrite("cum_sum_int64", &GatingParam::cumSumInt64)
        .def_readwrite("device_expert", &GatingParam::deviceExpert)
        .def("__repr__", [](const GatingParam &param) { return "GatingParam: " + OpParamToJson(param).dump(); });

    py::class_<ReshapeAndCacheParam> reshapeAndCache(m, "ReshapeAndCacheParam");

    py::enum_<ReshapeAndCacheParam::CompressType>(reshapeAndCache, "CompressType")
        .value("COMPRESS_TYPE_UNDEFINED", ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED)
        .value("COMPRESS_TYPE_KVHEAD", ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD)
        .value("COMPRESS_TYPE_KVHEAD_ROPE", ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD_ROPE);

    py::enum_<ReshapeAndCacheParam::KvCacheCfg>(reshapeAndCache, "KvCacheCfg")
        .value("K_CACHE_V_CACHE", ReshapeAndCacheParam::K_CACHE_V_CACHE)
        .value("K_CACHE_V_BYPASS", ReshapeAndCacheParam::K_CACHE_V_BYPASS);

    reshapeAndCache
        .def(py::init<ReshapeAndCacheParam::CompressType, ReshapeAndCacheParam::KvCacheCfg>(),
             py::arg("compress_type") = ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED,
             py::arg("kv_cache_cfg") = ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE)
        .def_readwrite("compress_type", &ReshapeAndCacheParam::compressType)
        .def_readwrite("kv_cache_cfg", &ReshapeAndCacheParam::kvCacheCfg)
        .def("__repr__",
             [](const ReshapeAndCacheParam &param) { return "ReshapeAndCacheParam: " + OpParamToJson(param).dump(); });

    py::class_<FillParam>(m, "FillParam")
        .def(py::init([](bool withMask, const std::vector<float> &value, const std::vector<int64_t> &outDim) {
                 FillParam param;
                 param.withMask = withMask;
                 AddElements(value, param.value);
                 AddElements(outDim, param.outDim);
                 return param;
                 }),
             py::arg("with_mask") = true,
             py::arg("value") = py::list(),
             py::arg("out_dim") = py::list())
        .def_readwrite("with_mask", &FillParam::withMask)
        .def_readwrite("value", &FillParam::value)
        .def_readwrite("out_dim", &FillParam::outDim)
        .def_property(
            "value",
            [](FillParam &param) -> std::vector<float> {
                std::vector<float> valueVec;
                valueVec.resize(param.value.size());
                for (size_t i = 0; i < param.value.size(); i++) {
                    valueVec.at(i) = param.value.at(i);
                }
                return valueVec;
            },
            [](FillParam &param, const std::vector<float> &value) {
                param.value.clear();
                AddElements(value, param.value);
            })
        .def_property(
            "out_dim",
            [](FillParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> outDimVec;
                outDimVec.resize(param.outDim.size());
                for (size_t i = 0; i < param.outDim.size(); i++) {
                    outDimVec.at(i) = param.outDim.at(i);
                }
                return outDimVec;
            },
            [](FillParam &param, const std::vector<float> &outDim) {
                param.outDim.clear();
                AddElements(outDim, param.outDim);
            })
        .def("__repr__", [](const FillParam &param) { return "FillParam: " + OpParamToJson(param).dump(); });

    py::class_<RazorFusionAttentionParam>(m, "RazorFusionAttentionParam")
        .def(py::init<int32_t, int32_t, float, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>(),
             py::arg("head_num") = 1,
             py::arg("kv_head_num") = 1,
             py::arg("qk_scale") = 1,
             py::arg("razor_len") = 0,
             py::arg("pre_tokens") = 0,
             py::arg("next_tokens") = 0,
             py::arg("tile_q") = 0,
             py::arg("tile_kv") = 0,
             py::arg("text_q_len") = 0,
             py::arg("text_kv_len") = 0)
        .def_readwrite("head_num", &RazorFusionAttentionParam::headNum)
        .def_readwrite("kv_head_num", &RazorFusionAttentionParam::kvHeadNum)
        .def_readwrite("qk_scale", &RazorFusionAttentionParam::qkScale)
        .def_readwrite("razor_len", &RazorFusionAttentionParam::razorLen)
        .def_readwrite("pre_tokens", &RazorFusionAttentionParam::preTokens)
        .def_readwrite("next_tokens", &RazorFusionAttentionParam::nextTokens)
        .def_readwrite("tile_q", &RazorFusionAttentionParam::tileQ)
        .def_readwrite("tile_kv", &RazorFusionAttentionParam::tileKv)
        .def_readwrite("text_q_len", &RazorFusionAttentionParam::textQLen)
        .def_readwrite("text_kv_len", &RazorFusionAttentionParam::textKvLen)
        .def("__repr__",
             [](const RazorFusionAttentionParam &param) {
                 return "RazorFusionAttentionParam: " + OpParamToJson(param).dump();
             });

    py::class_<AllReduceParam> allReduce(m, "AllReduceParam");

    py::enum_<AllReduceParam::QuantType>(reshapeAndCache, "QuantType")
        .value("QUANT_TYPE_UNQUANT", AllReduceParam::QuantType::QUANT_TYPE_UNQUANT)
        .value("QUANT_TYPE_UNDEFINED", AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED)
        .value("QUANT_TYPE_PER_TENSOR", AllReduceParam::QuantType::QUANT_TYPE_PER_TENSOR)
        .value("QUANT_TYPE_PER_CHANNEL", AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL)
        .value("QUANT_TYPE_MAX", AllReduceParam::QuantType::QUANT_TYPE_MAX);

    allReduce
        .def(py::init<int, int, int, std::string, std::string, HcclComm, CommMode, std::string, std::string,
                      AllReduceParam::QuantType, aclDataType>(),
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("all_reduce_type") = "sum",
             py::arg("backend") = "hccl",
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("rank_table_file") = "",
             py::arg("comm_domain") = "",
             py::arg("quant_type") = AllReduceParam::QuantType::QUANT_TYPE_UNQUANT,
             py::arg("out_data_type") = aclDataType::ACL_DT_UNDEFINED)
        .def_readwrite("rank", &AllReduceParam::rank)
        .def_readwrite("rank_size", &AllReduceParam::rankSize)
        .def_readwrite("rank_root", &AllReduceParam::rankRoot)
        .def_readwrite("all_reduce_type", &AllReduceParam::allReduceType)
        .def_readwrite("backend", &AllReduceParam::backend)
        .def_readwrite("hccl_comm", &AllReduceParam::hcclComm)
        .def_readwrite("comm_mode", &AllReduceParam::commMode)
        .def_readwrite("rank_table_file", &AllReduceParam::rankTableFile)
        .def_readwrite("comm_domain", &AllReduceParam::commDomain)
        .def_readwrite("quant_type", &AllReduceParam::quantType)
        .def_readwrite("out_data_type", &AllReduceParam::outDataType)
        .def("__repr__", [](const AllReduceParam &param) { return "AllReduceParam: " + OpParamToJson(param).dump(); });
    
    py::class_<BroadcastParam>(m, "BroadcastParam")
        .def(py::init<int, int, int, HcclComm, CommMode, std::string, std::string, std::string>(),
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("backend") = "hccl",
             py::arg("rank_table_file") = "",
             py::arg("comm_domain") = "")
        .def_readwrite("rank", &BroadcastParam::rank)
        .def_readwrite("rank_size", &BroadcastParam::rankSize)
        .def_readwrite("rank_root", &BroadcastParam::rankRoot)
        .def_readwrite("hccl_comm", &BroadcastParam::hcclComm)
        .def_readwrite("comm_mode", &BroadcastParam::commMode)
        .def_readwrite("backend", &BroadcastParam::backend)
        .def_readwrite("rank_table_file", &BroadcastParam::rankTableFile)
        .def_readwrite("comm_domain", &BroadcastParam::commDomain)
        .def("__repr__",
             [](const BroadcastParam &param) { return "BroadcastParam: " + OpParamToJson(param).dump(); });

    py::class_<ReduceScatterParam>(m, "ReduceScatterParam")
        .def(py::init<int, int, int, std::string, HcclComm, CommMode, std::string, std::string, std::string>(),
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("reduce_type") = "sum",
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("backend") = "lccl",
             py::arg("rank_table_file") = "",
             py::arg("comm_domain") = "")
        .def_readwrite("rank", &ReduceScatterParam::rank)
        .def_readwrite("rank_size", &ReduceScatterParam::rankSize)
        .def_readwrite("rank_root", &ReduceScatterParam::rankRoot)
        .def_readwrite("reduce_type", &ReduceScatterParam::reduceType)
        .def_readwrite("hccl_comm", &ReduceScatterParam::hcclComm)
        .def_readwrite("comm_mode", &ReduceScatterParam::commMode)
        .def_readwrite("backend", &ReduceScatterParam::backend)
        .def_readwrite("rank_table_file", &ReduceScatterParam::rankTableFile)
        .def_readwrite("comm_domain", &ReduceScatterParam::commDomain)
        .def("__repr__",
             [](const ReduceScatterParam &param) { return "ReduceScatterParam: " + OpParamToJson(param).dump(); });

    py::class_<ReduceScatterVParam>(m, "ReduceScatterVParam")
        .def(py::init([](int rank, int rankSize, int rankRoot, const std::vector<int64_t> &sendCounts,
                         const std::vector<int64_t> &sdispls, std::int64_t recvCount, std::string reduceType,
                         HcclComm hcclComm, CommMode commMode, std::string backend, std::string rankTableFile,
                         std::string commDomain) {
                ReduceScatterVParam param;
                param.rank = rank;
                param.rankSize = rankSize;
                param.rankRoot = rankRoot;
                AddElements(sendCounts, param.sendCounts);
                AddElements(sdispls, param.sdispls);
                param.recvCount = recvCount;
                param.reduceType = reduceType;
                param.hcclComm = hcclComm;
                param.commMode = commMode;
                param.backend = backend;
                param.rankTableFile = rankTableFile;
                param.commDomain = commDomain;
                return param;
                }),
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("send_counts") = py::list(),
             py::arg("sdispls") = py::list(),
             py::arg("recv_count") = 0,
             py::arg("reduce_type") = "sum",
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("backend") = "hccl",
             py::arg("rank_table_file") = "",
             py::arg("comm_domain") = "")
        .def_readwrite("rank", &ReduceScatterVParam::rank)
        .def_readwrite("rank_size", &ReduceScatterVParam::rankSize)
        .def_readwrite("rank_root", &ReduceScatterVParam::rankRoot)
        .def_readwrite("send_counts", &ReduceScatterVParam::sendCounts)
        .def_readwrite("sdispls", &ReduceScatterVParam::sdispls)
        .def_readwrite("recv_count", &ReduceScatterVParam::recvCount)
        .def_readwrite("reduce_type", &ReduceScatterVParam::reduceType)
        .def_readwrite("hccl_comm", &ReduceScatterVParam::hcclComm)
        .def_readwrite("comm_mode", &ReduceScatterVParam::commMode)
        .def_readwrite("backend", &ReduceScatterVParam::backend)
        .def_readwrite("rank_table_file", &ReduceScatterVParam::rankTableFile)
        .def_readwrite("comm_domain", &ReduceScatterVParam::commDomain)
        .def("__repr__",
             [](const ReduceScatterVParam &param) { return "ReduceScatterVParam: " + OpParamToJson(param).dump(); });

    py::class_<FaUpdateParam> faUpdate(m, "FaUpdateParam");

    py::enum_<FaUpdateParam::FaUpdateType>(faUpdate, "FaUpdateType")
        .value("DECODE_UPDATE", FaUpdateParam::FaUpdateType::DECODE_UPDATE);

    faUpdate.def(py::init<FaUpdateParam::FaUpdateType, uint32_t>(),
                 py::arg("fa_update_type") = FaUpdateParam::FaUpdateType::DECODE_UPDATE,
                 py::arg("sp") = 1)
        .def_readwrite("fa_update_type", &FaUpdateParam::faUpdateType)
        .def_readwrite("sp", &FaUpdateParam::sp)
        .def("__repr__", [](const FaUpdateParam &param) { return "FaUpdateParam: " + OpParamToJson(param).dump(); });

    py::class_<LinearParallelParam> linearParallel(m, "LinearParallelParam");

    py::enum_<LinearParallelParam::ParallelType>(linearParallel, "ParallelType")
        .value("UNDEFINED", LinearParallelParam::ParallelType::UNDEFINED)
        .value("LINEAR_ALL_REDUCE", LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE)
        .value("LINEAR_REDUCE_SCATTER", LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER)
        .value("ALL_GATHER_LINEAR", LinearParallelParam::ParallelType::ALL_GATHER_LINEAR)
        .value("PURE_LINEAR", LinearParallelParam::ParallelType::PURE_LINEAR)
        .value("ALL_GATHER_LINEAR_REDUCE_SCATTER", LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER)
        .value("MAX", LinearParallelParam::ParallelType::MAX);

    py::enum_<LinearParallelParam::QuantType>(linearParallel, "QuantType")
        .value("QUANT_TYPE_UNDEFINED", LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED)
        .value("QUANT_TYPE_UNQUANT", LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT)
        .value("QUANT_TYPE_PER_TENSOR", LinearParallelParam::QuantType::QUANT_TYPE_PER_TENSOR)
        .value("QUANT_TYPE_PER_CHANNEL", LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL)
        .value("QUANT_TYPE_PER_GROUP", LinearParallelParam::QuantType::QUANT_TYPE_PER_GROUP)
        .value("QUANT_TYPE_PER_TOKEN", LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN)
        .value("QUANT_TYPE_MAX", LinearParallelParam::QuantType::QUANT_TYPE_MAX);

    py::class_<LinearParallelParam::TwoDimTPInfo>(linearParallel, "TwoDimTPInfo")
        .def(py::init<uint16_t, uint16_t, uint8_t>(),
             py::arg("ag_dim") = 0,
             py::arg("rs_dim") = 0,
             py::arg("inner_dim_is_ag") = 1)
        .def_readwrite("ag_dim", &LinearParallelParam::TwoDimTPInfo::agDim)
        .def_readwrite("rs_dim", &LinearParallelParam::TwoDimTPInfo::rsDim)
        .def_readwrite("inner_dim_is_ag", &LinearParallelParam::TwoDimTPInfo::innerDimIsAg);

    py::class_<LinearParallelParam::MoeInfo>(linearParallel, "MoeInfo")
        .def(py::init<int16_t, int8_t, int8_t>(),
             py::arg("localExpertNums") = 0,
             py::arg("epSize") = 0,
             py::arg("tpSize") = 0)
        .def_readwrite("localExpertNums", &LinearParallelParam::MoeInfo::localExpertNums)
        .def_readwrite("epSize", &LinearParallelParam::MoeInfo::epSize)
        .def_readwrite("tpSize", &LinearParallelParam::MoeInfo::tpSize);

    linearParallel
        .def(py::init<bool, int, int, int, bool, std::string, HcclComm, CommMode, std::string,
                      LinearParallelParam::ParallelType, bool, LinearParallelParam::QuantType, int32_t, aclDataType,
                      std::string, LinearParallelParam::TwoDimTPInfo, LinearParallelParam::MoeInfo>(),
             py::arg("trans_weight") = true,
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("has_residual") = false,
             py::arg("backend") = "hccl",
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("rank_table_file") = "",
             py::arg("type") = LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE,
             py::arg("keep_intermediate") = false,
             py::arg("quant_type") = LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT,
             py::arg("quant_group_size") = 0,
             py::arg("out_data_type") = aclDataType::ACL_DT_UNDEFINED,
             py::arg("comm_domain") = "",
             py::arg("two_dim_TP_info") = LinearParallelParam::TwoDimTPInfo(),
             py::arg("moe_info") = LinearParallelParam::MoeInfo())
        .def_readwrite("trans_weight", &LinearParallelParam::transWeight)
        .def_readwrite("rank", &LinearParallelParam::rank)
        .def_readwrite("rank_size", &LinearParallelParam::rankSize)
        .def_readwrite("rank_root", &LinearParallelParam::rankRoot)
        .def_readwrite("has_residual", &LinearParallelParam::hasResidual)
        .def_readwrite("backend", &LinearParallelParam::backend)
        .def_readwrite("hccl_comm", &LinearParallelParam::hcclComm)
        .def_readwrite("comm_mode", &LinearParallelParam::commMode)
        .def_readwrite("rank_table_file", &LinearParallelParam::rankTableFile)
        .def_readwrite("type", &LinearParallelParam::type)
        .def_readwrite("keep_intermediate", &LinearParallelParam::keepIntermediate)
        .def_readwrite("quant_type", &LinearParallelParam::quantType)
        .def_readwrite("quant_group_size", &LinearParallelParam::quantGroupSize)
        .def_readwrite("out_data_type", &LinearParallelParam::outDataType)
        .def_readwrite("comm_domain", &LinearParallelParam::commDomain)
        .def_readwrite("two_dim_TP_info", &LinearParallelParam::twoDimTPInfo)
        .def_readwrite("moe_info", &LinearParallelParam::moeInfo)
        .def("__repr__",
             [](const LinearParallelParam &param) { return "LinearParallelParam: " + OpParamToJson(param).dump(); });

    py::class_<LinearSparseParam>(m, "LinearSparseParam")
        .def(py::init<bool, bool, uint32_t, uint32_t>(),
             py::arg("transpose_a") = false,
             py::arg("transpose_b") = true,
             py::arg("tiling_k") = 8,
             py::arg("tiling_n") = 8)
        .def_readwrite("transpose_a", &LinearSparseParam::transposeA)
        .def_readwrite("transpose_b", &LinearSparseParam::transposeB)
        .def_readwrite("tiling_k", &LinearSparseParam::tilingK)
        .def_readwrite("tiling_n", &LinearSparseParam::tilingN)
        .def("__repr__",
             [](const LinearSparseParam &param) { return "LinearSparseParam: " + OpParamToJson(param).dump(); });

    py::class_<RelayAttentionParam> relayAttention(m, "RelayAttentionParam");

    py::enum_<RelayAttentionParam::MaskType>(relayAttention, "MaskType")
        .value("MASK_TYPE_UNDEFINED", RelayAttentionParam::MaskType::MASK_TYPE_UNDEFINED)
        .value("MASK_TYPE_NORM", RelayAttentionParam::MaskType::MASK_TYPE_NORM);

    relayAttention
        .def(py::init<int32_t, float, int32_t, RelayAttentionParam::MaskType>(),
             py::arg("head_num") = 0,
             py::arg("qk_scale") = 1,
             py::arg("expected_kv_head_num") = 0,
             py::arg("mask_type") = RelayAttentionParam::MaskType::MASK_TYPE_UNDEFINED)
        .def_readwrite("head_num", &RelayAttentionParam::headNum)
        .def_readwrite("qk_scale", &RelayAttentionParam::qkScale)
        .def_readwrite("kv_head_num", &RelayAttentionParam::kvHeadNum)
        .def_readwrite("mask_type", &RelayAttentionParam::maskType)
        .def("__repr__",
             [](const RelayAttentionParam &param) { return "RelayAttentionParam: " + OpParamToJson(param).dump(); });

    py::class_<TopkToppSamplingParam> topkToppSampling(m, "TopkToppSamplingParam");
 
    py::enum_<TopkToppSamplingParam::TopkToppSamplingType>(topkToppSampling, "TopkToppSamplingType")
        .value("SAMPLING_UNDEFINED", TopkToppSamplingParam::TopkToppSamplingType::SAMPLING_UNDEFINED)
        .value("SINGLE_TOPK_SAMPLING", TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING)
        .value("BATCH_TOPK_MULTINOMIAL_SAMPLING",
               TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING)
        .value("BATCH_TOPK_EXPONENTIAL_SAMPLING",
               TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING)
        .value("BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING",
               TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING)
        .value("BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING",
               TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING)
        .value("SAMPLING_MAX", TopkToppSamplingParam::TopkToppSamplingType::SAMPLING_MAX);
 
    topkToppSampling
        .def(py::init([](TopkToppSamplingParam::TopkToppSamplingType topkToppSamplingType,
                         const std::vector<uint32_t> &randSeeds, uint32_t randSeed, uint32_t topk,
                         int32_t logProbsSize) {
                TopkToppSamplingParam param;
                param.topkToppSamplingType = topkToppSamplingType;
                AddElements(randSeeds, param.randSeeds);
                param.randSeed = randSeed;
                param.topk = topk;
                param.logProbsSize = logProbsSize;
                return param;
                }),
             py::arg("topk_topp_sampling_type") = TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING,
             py::arg("rand_seeds") = py::list(),
             py::arg("rand_seed") = 0,
             py::arg("topk") = 100,
             py::arg("log_probs_size") = 0)
        .def_readwrite("topk_topp_sampling_type", &TopkToppSamplingParam::topkToppSamplingType)
        .def_readwrite("rand_seeds", &TopkToppSamplingParam::randSeeds)
        .def_readwrite("rand_seed", &TopkToppSamplingParam::randSeed)
        .def_readwrite("topk", &TopkToppSamplingParam::topk)
        .def_readwrite("log_probs_size", &TopkToppSamplingParam::logProbsSize)
        .def("__repr__", [](const TopkToppSamplingParam &param) {
            return "TopkToppSamplingParam: " + OpParamToJson(param).dump();
        });

    py::class_<AllToAllParam>(m, "AllToAllParam")
        .def(py::init<int, int, int, std::string, HcclComm, CommMode, std::string, std::string, bool>(),
             py::arg("rank") = 0,
             py::arg("rank_size") = 0,
             py::arg("rank_root") = 0,
             py::arg("backend") = "hccl",
             py::arg("hccl_comm") = nullptr,
             py::arg("comm_mode") = CommMode::COMM_MULTI_PROCESS,
             py::arg("rank_table_file") = "",
             py::arg("comm_domain") = "",
             py::arg("transpose") = false)
        .def_readwrite("rank", &AllToAllParam::rank)
        .def_readwrite("rank_size", &AllToAllParam::rankSize)
        .def_readwrite("rank_root", &AllToAllParam::rankRoot)
        .def_readwrite("backend", &AllToAllParam::backend)
        .def_readwrite("hccl_comm", &AllToAllParam::hcclComm)
        .def_readwrite("comm_mode", &AllToAllParam::commMode)
        .def_readwrite("rank_table_file", &AllToAllParam::rankTableFile)
        .def_readwrite("comm_domain", &AllToAllParam::commDomain)
        .def_readwrite("transpose", &AllToAllParam::transpose)
        .def("__repr__",
             [](const AllToAllParam &param) { return "AllToAllParam: " + atb::OpParamToJson(param).dump(); });
}