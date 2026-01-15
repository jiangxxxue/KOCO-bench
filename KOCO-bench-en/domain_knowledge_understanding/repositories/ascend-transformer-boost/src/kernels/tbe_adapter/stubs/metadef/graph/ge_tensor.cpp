/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_tensor.h"

#include <map>
#include <cstring>
#include <sstream>
#include <securec.h>
#include "common/util/mem_utils.h"
#include "debug/ge_util.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/utils/tensor_utils.h"

#define UNUSED_VALUE(x) (void)(x)
namespace ge {
namespace {
const std::map<DeviceType, std::string> kDeviceToStrMap = {
    {NPU, "NPU"},
    {CPU, "CPU"},
};

const std::map<std::string, DeviceType> kStrToDeviceMap = {{"NPU", NPU}, {"CPU", CPU}};

} // namespace

void GeTensorSerializeUtils::GeShapeAsProto(const GeShape &shape, proto::ShapeDef *proto)
{
    UNUSED_VALUE(shape);
    UNUSED_VALUE(proto);
    MKI_LOG(ERROR) << "fail in GeShapeAsProto";
}
void GeTensorSerializeUtils::GeTensorDescAsProto(const GeTensorDescImpl &desc, proto::TensorDescriptor *proto)
{
    UNUSED_VALUE(desc);
    UNUSED_VALUE(proto);
    MKI_LOG(ERROR) << "fail in GeTensorDescAsProto";
}
void GeTensorSerializeUtils::GeTensorDescAsProto(const GeTensorDesc &desc, proto::TensorDescriptor *proto)
{
    UNUSED_VALUE(desc);
    UNUSED_VALUE(proto);
    MKI_LOG(ERROR) << "fail in GeTensorDescAsProto";
}
void GeTensorSerializeUtils::GeTensorAsProto(const GeTensorImpl &tensor, proto::TensorDef *proto)
{
    UNUSED_VALUE(tensor);
    UNUSED_VALUE(proto);
    MKI_LOG(ERROR) << "fail in GeTensorAsProto";
}
void GeTensorSerializeUtils::GeTensorAsProto(const GeTensor &tensor, proto::TensorDef *proto)
{
    UNUSED_VALUE(tensor);
    UNUSED_VALUE(proto);
    MKI_LOG(ERROR) << "fail in GeTensorAsProto";
}

void GeTensorSerializeUtils::AssembleGeShapeFromProto(const proto::ShapeDef *proto, GeShape &shape)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(shape);
    MKI_LOG(ERROR) << "fail in AssembleGeShapeFromProto";
}
void GeTensorSerializeUtils::AssembleGeTensorDescFromProto(const proto::TensorDescriptor *const proto,
                                                           GeTensorDesc &desc)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(desc);
    MKI_LOG(ERROR) << "fail in AssembleGeTensorDescFromProto";
}
void GeTensorSerializeUtils::AssembleGeTensorFromProto(const proto::TensorDef *proto, GeTensor &tensor)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(tensor);
    MKI_LOG(ERROR) << "fail in AssembleGeTensorFromProto";
}

void GeTensorSerializeUtils::NormalizeGeTensorDescProto(proto::TensorDescriptor *proto)
{
    UNUSED_VALUE(proto);
    MKI_LOG(ERROR) << "fail in NormalizeGeTensorDescProto";
}

void GeTensorSerializeUtils::GetShapeFromDescProto(const proto::TensorDescriptor *const proto, GeShape &shape)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(shape);
    MKI_LOG(ERROR) << "fail in GetShapeFromDescProto";
}

void GeTensorSerializeUtils::GetOriginShapeFromDescProto(const proto::TensorDescriptor *const proto, GeShape &shape)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(shape);
    MKI_LOG(ERROR) << "fail in GetOriginShapeFromDescProto";
}

void GeTensorSerializeUtils::GetDtypeFromDescProto(const proto::TensorDescriptor *const proto, DataType &dtype)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(dtype);
    MKI_LOG(ERROR) << "fail in GetDtypeFromDescProto";
}

void GeTensorSerializeUtils::GetOriginDtypeFromDescProto(const proto::TensorDescriptor *const proto, DataType &dtype)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(dtype);
    MKI_LOG(ERROR) << "fail in GetOriginDtypeFromDescProto";
}

void GeTensorSerializeUtils::GetFormatFromDescProto(const proto::TensorDescriptor *const proto, Format &format)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(format);
    MKI_LOG(ERROR) << "fail in GetFormatFromDescProto";
}

void GeTensorSerializeUtils::GetOriginFormatFromDescProto(const proto::TensorDescriptor *const proto, Format &format)
{
    UNUSED_VALUE(proto);
    UNUSED_VALUE(format);
    MKI_LOG(ERROR) << "fail in GetOriginFormatFromDescProto";
}

class GeShapeImpl {
    using DimsType = SmallVector<int64_t, kDefaultDimsNum>;

public:
    ~GeShapeImpl() = default;
    GeShapeImpl() = default;
    explicit GeShapeImpl(proto::ShapeDef *const proto_msg);
    explicit GeShapeImpl(const std::vector<int64_t> &dims);
    void AppendDim(const int64_t dim_size);
    void SetDimNum(const size_t dim_num);
    void SetIsUnknownDimNum();
    bool IsUnknownDimNum() const;
    int64_t GetDim(const size_t idx) const;
    size_t GetDimNum() const;
    std::vector<int64_t> ShapeImplGetDims() const;
    graphStatus SetDim(const size_t idx, const int64_t value);
    std::string ShapeImplToString() const;
    const DimsType &ShapeImplGetMutableDims() const;
    int64_t GetShapeSize() const;
    bool IsScalar() const;
    bool operator==(const GeShapeImpl &other) const;
    bool IsUnknownShape() const;

private:
    DimsType dims_;
    friend class GeTensorDesc;
};

// Default
GeShapeImpl::GeShapeImpl(const std::vector<int64_t> &dims)
{
    UNUSED_VALUE(dims);
    MKI_LOG(ERROR) << "fail in GeShapeImpl";
}

void GeShapeImpl::SetDimNum(const size_t dim_num)
{
    MKI_LOG(ERROR) << "fail in SetDimNum";
    dims_.resize(dim_num, UNKNOWN_DIM);
}

void GeShapeImpl::AppendDim(const int64_t dim_size)
{
    MKI_LOG(ERROR) << "fail in AppendDim";
    dims_.push_back(dim_size);
}

bool GeShapeImpl::IsUnknownDimNum() const
{
    MKI_LOG(ERROR) << "fail in IsUnknownDimNum";
    return (dims_.size() == 1UL) && (dims_[0UL] == UNKNOWN_DIM_NUM);
}

void GeShapeImpl::SetIsUnknownDimNum()
{
    MKI_LOG(ERROR) << "fail in SetIsUnknownDimNum";
}

size_t GeShapeImpl::GetDimNum() const
{
    MKI_LOG(ERROR) << "fail in GetDimNum";
    return static_cast<size_t>(-1);
}

int64_t GeShapeImpl::GetDim(const size_t idx) const
{
    UNUSED_VALUE(idx);
    MKI_LOG(ERROR) << "fail in GetDim";
    return 0;
}

graphStatus GeShapeImpl::SetDim(const size_t idx, const int64_t value)
{
    UNUSED_VALUE(idx);
    UNUSED_VALUE(value);
    MKI_LOG(ERROR) << "fail in SetDim";
    return GRAPH_FAILED;
}

std::vector<int64_t> GeShapeImpl::ShapeImplGetDims() const
{
    MKI_LOG(ERROR) << "fail in ShapeImplGetDims";
    return std::vector<int64_t>();
}

const SmallVector<int64_t, kDefaultDimsNum> &GeShapeImpl::ShapeImplGetMutableDims() const
{
    MKI_LOG(ERROR) << "fail in ShapeImplGetMutableDims";
    return dims_;
}

std::string GeShapeImpl::ShapeImplToString() const
{
    MKI_LOG(ERROR) << "fail in ShapeImplToString";
    return std::string();
}

int64_t GeShapeImpl::GetShapeSize() const
{
    MKI_LOG(ERROR) << "fail in GetShapeSize";
    return 0;
}

bool GeShapeImpl::IsUnknownShape() const
{
    MKI_LOG(ERROR) << "fail in IsUnknownShape";
    return std::any_of(dims_.begin(), dims_.end(), [](const int64_t &dim) {
        return (dim == UNKNOWN_DIM) || (dim == UNKNOWN_DIM_NUM) || (dim < 0);
    });
}

bool GeShapeImpl::IsScalar() const
{
    MKI_LOG(ERROR) << "fail in IsScalar";
    return dims_.empty();
}

GeShapeImpl::GeShapeImpl(proto::ShapeDef *const proto_msg)
{
    MKI_LOG(ERROR) << "fail in GeShapeImpl";
    UNUSED_VALUE(proto_msg);
}

bool GeShapeImpl::operator==(const GeShapeImpl &other) const
{
    MKI_LOG(ERROR) << "fail in operator";
    return this->ShapeImplGetDims() == other.ShapeImplGetDims();
}

GeShape::GeShape(std::vector<int64_t> s) : impl_(MakeShared<GeShapeImpl>(std::move(s))) {}
GeShape::GeShape() : impl_(MakeShared<GeShapeImpl>()) {}

GeShape::GeShape(GeShape &&other) : impl_(MakeShared<GeShapeImpl>(std::move(*(other.impl_)))) {}
GeShape::GeShape(const GeShape &other) : impl_(MakeShared<GeShapeImpl>(*(other.impl_))) {}

GeShape::GeShape(const ProtoMsgOwner &proto_owner, proto::ShapeDef *const proto_msg)
    : impl_(MakeShared<GeShapeImpl>(proto_msg))
{
    MKI_LOG(ERROR) << "fail in MutableAttrMap";

    UNUSED_VALUE(proto_owner);
}
GeShape::~GeShape() = default;

size_t GeShape::GetDimNum() const
{
    MKI_LOG(ERROR) << "fail in GetDimNum";
    return impl_->GetDimNum();
}

void GeShape::SetDimNum(const size_t dim_num)
{
    MKI_LOG(ERROR) << "fail in SetDimNum";
    impl_->SetDimNum(dim_num);
}

void GeShape::AppendDim(const int64_t dim_size)
{
    MKI_LOG(ERROR) << "fail in AppendDim";
    impl_->AppendDim(dim_size);
}

bool GeShape::IsUnknownDimNum() const
{
    MKI_LOG(ERROR) << "fail in IsUnknownDimNum";
    return impl_->IsUnknownDimNum();
}

void GeShape::SetIsUnknownDimNum()
{
    MKI_LOG(ERROR) << "fail in SetIsUnknownDimNum";
    impl_->SetIsUnknownDimNum();
}

int64_t GeShape::GetDim(const size_t idx) const
{
    MKI_LOG(ERROR) << "fail in GetDim";
    return impl_->GetDim(idx);
}

graphStatus GeShape::SetDim(const size_t idx, const int64_t value)
{
    MKI_LOG(ERROR) << "fail in SetDim";
    return impl_->SetDim(idx, value);
}

std::vector<int64_t> GeShape::GetDims() const
{
    MKI_LOG(ERROR) << "fail in GetDims";
    return impl_->ShapeImplGetDims();
}

const SmallVector<int64_t, kDefaultDimsNum> &GeShape::GetMutableDims() const
{
    MKI_LOG(ERROR) << "fail in GetMutableDims";
    return impl_->ShapeImplGetMutableDims();
}

std::string GeShape::ToString() const
{
    MKI_LOG(ERROR) << "fail in ToString";
    return impl_->ShapeImplToString();
}

int64_t GeShape::GetShapeSize() const
{
    MKI_LOG(ERROR) << "fail in GetShapeSize";
    return impl_->GetShapeSize();
}

bool GeShape::IsUnknownShape() const
{
    MKI_LOG(ERROR) << "fail in IsUnknownShape";
    return impl_->IsUnknownShape();
}

bool GeShape::IsScalar() const
{
    MKI_LOG(ERROR) << "fail in IsScalar";
    return impl_->IsScalar();
}

GeShape &GeShape::operator=(const GeShape &other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

GeShape &GeShape::operator=(GeShape &&other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

bool GeShape::operator==(const GeShape &other) const { return (*impl_) == (*(other.impl_)); }

// GeTensorDesc
GeTensorDescImpl::GeTensorDescImpl(const GeShape &shape, const Format format, const DataType dt) : GeTensorDescImpl()
{
    SetFormat(format);
    SetDataType(dt);
    shape_ = shape;
}

GeTensorDescImpl::GeTensorDescImpl(proto::TensorDescriptor *const proto_msg) : GeTensorDescImpl()
{
    UNUSED_VALUE(proto_msg);
    MKI_LOG(ERROR) << "fail in GeTensorDescImpl";
}

void GeTensorDescImpl::SetDataType(const DataType dtype)
{
    MKI_LOG(DEBUG) << "stub SetDataType";
    dtype_ = dtype;
}

void GeTensorDescImpl::SetOriginDataType(const DataType dtype)
{
    MKI_LOG(ERROR) << "fail in SetOriginDataType";
    origin_dtype_ = dtype;
}

DataType GeTensorDescImpl::GetOriginDataType() const
{
    MKI_LOG(ERROR) << "fail in GetOriginDataType";
    return origin_dtype_;
}

void GeTensorDescImpl::SetFormat(const Format format)
{
    MKI_LOG(DEBUG) << "stub SetFormat";
    format_ = format;
}

void GeTensorDescImpl::SetOriginFormat(const Format format)
{
    MKI_LOG(DEBUG) << "stub SetOriginFormat";
    origin_format_ = format;
}

Format GeTensorDescImpl::GetOriginFormat() const
{
    MKI_LOG(ERROR) << "fail in GetOriginFormat";
    return origin_format_;
}

GeShape &GeTensorDescImpl::ShapeReference() const
{
    MKI_LOG(ERROR) << "fail in ShapeReference";
    return shape_;
}

GeShape &GeTensorDescImpl::OriginShapeReference() const
{
    MKI_LOG(ERROR) << "fail in OriginShapeReference";
    return origin_shape_;
}

bool GeTensorDescImpl::GeTensorDescAttrsAreEqual(const GeTensorDescImpl &other) const
{
    // The definition of attribute equality remains unchanged
    MKI_LOG(ERROR) << "fail in GeTensorDescAttrsAreEqual";
    return ((shape_ == other.shape_) && (dtype_ == other.dtype_) && (format_ == other.format_) &&
            (ext_meta_ == other.ext_meta_));
}

bool GeTensorDescImpl::operator==(const GeTensorDescImpl &other) const
{
    // The definition of attribute equality remains unchanged
    MKI_LOG(ERROR) << "fail in operator";
    return (origin_shape_ == other.origin_shape_) && (origin_format_ == other.origin_format_) &&
           (origin_dtype_ == other.origin_dtype_) && (GeTensorDescAttrsAreEqual(other));
}

ProtoAttrMap &GeTensorDescImpl::MutableAttrMap()
{
    MKI_LOG(ERROR) << "fail in MutableAttrMap";
    return attrs_;
}

ConstProtoAttrMap &GeTensorDescImpl::GetAttrMap() const
{
    MKI_LOG(ERROR) << "fail in GetAttrMap";
    return attrs_;
}

void GeTensorDescImpl::SetShape(GeShape &shape) const
{
    MKI_LOG(ERROR) << "fail in SetShape";
    ShapeReference() = std::move(shape);
}

Format GeTensorDescImpl::GetFormat() const
{
    return format_;
}

void GeTensorDescImpl::SetName(const std::string &name)
{
    MKI_LOG(ERROR) << "fail in SetName";
    ext_meta_.SetName(name);
}

const std::string GeTensorDescImpl::GetName() const
{
    MKI_LOG(ERROR) << "fail in GetName";
    return ext_meta_.GetName();
}

DataType GeTensorDescImpl::GetDataType() const
{
    return dtype_;
}

std::string GeTensorDescImpl::ExtMeta::GetDeviceTypeStr() const
{
    MKI_LOG(ERROR) << "fail in GetDeviceTypeStr";
    return std::string();
}

GeTensorDesc::GeTensorDesc() : AttrHolder(), impl_(ComGraphMakeShared<GeTensorDescImpl>()) {}

// Default
GeTensorDesc::GeTensorDesc(const GeShape &shape, const Format format, const DataType dt)
    : AttrHolder(), impl_(ComGraphMakeShared<GeTensorDescImpl>(shape, format, dt))
{
}

// Default
GeTensorDesc::GeTensorDesc(const GeTensorDesc &desc)
    : AttrHolder(desc), impl_(ComGraphMakeShared<GeTensorDescImpl>(*(desc.impl_)))
{
}

// Default
GeTensorDesc::GeTensorDesc(GeTensorDesc &&desc) : AttrHolder(desc), impl_(desc.impl_) {}

GeTensorDesc::~GeTensorDesc() = default;

GeTensorDesc::GeTensorDesc(proto::TensorDescriptor *const proto_msg)
    : AttrHolder(), impl_(ComGraphMakeShared<GeTensorDescImpl>(proto_msg))
{
}

bool GeTensorDesc::GeTensorDescAttrsAreEqual(const GeTensorDesc &r_ge_tensor_desc) const
{
    MKI_LOG(ERROR)<< "fail in GeTensorDescAttrsAreEqual";
    return impl_->GeTensorDescAttrsAreEqual(*(r_ge_tensor_desc.impl_));
}

bool GeTensorDesc::operator==(const GeTensorDesc &r_ge_tensor_desc) const
{
    MKI_LOG(ERROR)<< "fail in operator";
    return (*impl_) == (*(r_ge_tensor_desc.impl_));
}

GeShape &GeTensorDesc::ShapeReference() const
{
    MKI_LOG(ERROR)<< "fail in ShapeReference";
    return impl_->ShapeReference();
}

ProtoAttrMap &GeTensorDesc::MutableAttrMap()
{
    MKI_LOG(ERROR)<< "fail in MutableAttrMap";
    return impl_->MutableAttrMap();
}

ConstProtoAttrMap &GeTensorDesc::GetAttrMap() const
{
    MKI_LOG(ERROR)<< "fail in GetAttrMap";
    return impl_->GetAttrMap();
}

void GeTensorDesc::Update(const GeShape &shape, const Format format, const DataType dt)
{
    MKI_LOG(ERROR)<< "fail in Update";
    UNUSED_VALUE(shape);
    UNUSED_VALUE(format);
    UNUSED_VALUE(dt);
}
const GeShape &GeTensorDesc::GetShape() const
{
    MKI_LOG(ERROR)<< "fail in GetShape";
    return ShapeReference();
}

GeShape &GeTensorDesc::MutableShape()
{
    MKI_LOG(ERROR)<< "fail in MutableShape";
    return ShapeReference();
}

void GeTensorDesc::SetShape(const GeShape &shape)
{
    MKI_LOG(ERROR)<< "fail in SetShape";
    UNUSED_VALUE(shape);
}

void GeTensorDesc::SetShape(GeShape &&shape)
{
    MKI_LOG(ERROR)<< "fail in SetShape";
    UNUSED_VALUE(shape);
}

// set shape with -2, it stand for unknown shape
void GeTensorDesc::SetUnknownDimNumShape()
{
    MKI_LOG(ERROR)<< "fail in SetUnknownDimNumShape";
}

// for unknown shape
graphStatus GeTensorDesc::SetValueRange(const std::vector<std::pair<int64_t, int64_t>> &range)
{
    UNUSED_VALUE(range);
    MKI_LOG(ERROR) << "fail in SetValueRange";
    return GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetValueRange(std::vector<std::pair<int64_t, int64_t>> &range) const
{
    UNUSED_VALUE(range);
    std::vector<std::vector<int64_t>> value_range;
    MKI_LOG(ERROR) << "fail in GetValueRange";
    return GRAPH_FAILED;
}

graphStatus GeTensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range)
{
    UNUSED_VALUE(range);
    MKI_LOG(ERROR) << "fail in SetShapeRange";
    return GRAPH_FAILED;
}

graphStatus GeTensorDesc::SetOriginShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range)
{
    UNUSED_VALUE(range);
    std::vector<std::vector<int64_t>> origin_shape_range;
    MKI_LOG(ERROR) << "fail in SetOriginShapeRange";
    return GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const
{
    UNUSED_VALUE(range);
    std::vector<std::vector<int64_t>> shape_range;
    MKI_LOG(ERROR) << "fail in GetShapeRange";
    return GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetOriginShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const
{
    UNUSED_VALUE(range);
    MKI_LOG(ERROR) << "fail in GetOriginShape";
    std::vector<std::vector<int64_t>> origin_shape_range;
    return GRAPH_FAILED;
}

const GeShape &GeTensorDesc::GetOriginShape() const
{
    MKI_LOG(ERROR) << "fail in GetOriginShape";
    return impl_->OriginShapeReference();
}

GeShape &GeTensorDesc::MutableOriginShape() const { return impl_->OriginShapeReference(); }

void GeTensorDesc::SetOriginShape(const GeShape &origin_shape)
{
    MKI_LOG(ERROR) << "fail in SetOriginShape";
    impl_->OriginShapeReference() = origin_shape;
    impl_->SetOriginShapeInited(true);
}

bool GeTensorDesc::IsOriginShapeInitialized() const
{
    MKI_LOG(ERROR) << "fail in IsOriginShapeInitialized";
    return impl_->IsOriginShapeInited();
}

Format GeTensorDesc::GetFormat() const
{
    MKI_LOG(DEBUG) << "stub GetFormat";
    return impl_->GetFormat();
}

void GeTensorDesc::SetFormat(const Format format)
{
    MKI_LOG(DEBUG) << "stub SetFormat";
    return impl_->SetFormat(format);
}

void GeTensorDesc::SetName(const std::string &name)
{
    MKI_LOG(ERROR) << "fail in SetName";
    return impl_->SetName(name);
}

const std::string GeTensorDesc::GetName() const
{
    MKI_LOG(ERROR) << "fail in GetName";
    return impl_->GetName();
}

Format GeTensorDesc::GetOriginFormat() const
{
    MKI_LOG(ERROR) << "fail in GetOriginFormat";
    return impl_->GetOriginFormat();
}

void GeTensorDesc::SetOriginFormat(const Format origin_format)
{
    MKI_LOG(DEBUG) << "stub SetOriginFormat";
    impl_->SetOriginFormat(origin_format);
}

void GeTensorDesc::SetDataType(const DataType data_type)
{
    MKI_LOG(DEBUG) << "stub SetDataType";
    return impl_->SetDataType(data_type);
}

DataType GeTensorDesc::GetDataType() const
{
    MKI_LOG(DEBUG) << "stub GetDataType";
    return impl_->GetDataType();
}

void GeTensorDesc::SetOriginDataType(const DataType origin_data_type)
{
    MKI_LOG(ERROR) << "fail in SetOriginDataType";
    impl_->SetOriginDataType(origin_data_type);
}

DataType GeTensorDesc::GetOriginDataType() const
{
    MKI_LOG(ERROR) << "fail in GetOriginDataType";
    return impl_->GetOriginDataType();
}

std::vector<uint32_t> GeTensorDesc::GetRefPortIndex() const
{
    MKI_LOG(ERROR) << "fail in GetRefPortIndex";
    std::vector<uint32_t> ref_port_index;
    return ref_port_index;
}

void GeTensorDesc::SetRefPortByIndex(const std::vector<uint32_t> &index)
{
    MKI_LOG(ERROR) << "fail in SetRefPortByIndex";
    UNUSED_VALUE(index);
}

Placement GeTensorDesc::GetPlacement() const
{
    MKI_LOG(ERROR) << "fail in GetPlacement";
    int64_t placement = 0;
    return static_cast<Placement>(placement);
}

void GeTensorDesc::SetPlacement(const Placement placement)
{
    MKI_LOG(ERROR) << "fail in SetPlacement";
    UNUSED_VALUE(placement);
}

graphStatus GeTensorDesc::IsValid() const
{
    if ((this->GetDataType() != DT_UNDEFINED) || (this->GetFormat() != FORMAT_RESERVED)) {
        return GRAPH_SUCCESS;
    }
    MKI_LOG(ERROR) << "fail in IsValid";
    return GRAPH_PARAM_INVALID;
}

GeTensorDesc GeTensorDesc::Clone() const
{
    MKI_LOG(ERROR) << "fail in Clone";
    return *this;
}

GeTensorDesc &GeTensorDesc::operator=(const GeTensorDesc &desc)
{
    UNUSED_VALUE(desc);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

GeTensorDesc &GeTensorDesc::operator=(GeTensorDesc &&desc)
{
    UNUSED_VALUE(desc);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

const std::string GeTensorDesc::GetExpandDimsRule() const { return std::string(); }
void GeTensorDesc::SetExpandDimsRule(const std::string &expand_dims_rule)
{
    UNUSED_VALUE(expand_dims_rule);
    MKI_LOG(ERROR) << "fail in SetExpandDimsRule";
}

TensorData::TensorData()
{
    MKI_LOG(ERROR) << "fail in TensorData";
}

TensorData::TensorData(const TensorData &other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in TensorData";
}

TensorData::~TensorData() = default;

TensorData &TensorData::operator=(const TensorData &other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

graphStatus TensorData::SetData(std::vector<uint8_t> &&data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}
graphStatus TensorData::SetData(const std::vector<uint8_t> &data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}
graphStatus TensorData::SetData(const Buffer &data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}
graphStatus TensorData::SetData(const TensorData &data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus TensorData::SetData(const uint8_t *const data, const size_t size)
{
    UNUSED_VALUE(data);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus TensorData::SetData(uint8_t *const data, const size_t size, const AlignedPtr::Deleter &delete_fuc)
{
    UNUSED_VALUE(data);
    UNUSED_VALUE(size);
    UNUSED_VALUE(delete_fuc);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

void TensorData::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size)
{
    UNUSED_VALUE(aligned_ptr);
    MKI_LOG(ERROR) << "fail in SetData";
    UNUSED_VALUE(size);
}

const uint8_t *TensorData::MallocAlignedPtr(const size_t size)
{
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in MallocAlignedPtr";
    return nullptr;
}

size_t TensorData::GetSize() const
{
    MKI_LOG(ERROR) << "fail in GetSize";
    return 0;
}

const uint8_t *TensorData::GetData() const
{
    MKI_LOG(ERROR) << "fail in GetData";
    return nullptr;
}

uint8_t *TensorData::GetData()
{
    MKI_LOG(ERROR) << "fail in GetData";
    return nullptr;
}

const std::uint8_t *TensorData::data() const
{
    MKI_LOG(ERROR) << "fail in data";
    return GetData();
}
std::uint8_t *TensorData::data()
{
    MKI_LOG(ERROR) << "fail in data";
    return GetData();
}
std::size_t TensorData::size() const
{
    MKI_LOG(ERROR) << "fail in size";
    return GetSize();
}
void TensorData::clear()
{
    MKI_LOG(ERROR) << "fail in clear";
}

uint8_t TensorData::operator[](const size_t index) const
{
    UNUSED_VALUE(index);
    MKI_LOG(ERROR) << "fail in operator";
    return 0;
}

const std::shared_ptr<AlignedPtr> &TensorData::GetAlignedPtr()
{
    static std::shared_ptr<AlignedPtr> ptr = nullptr;
    MKI_LOG(ERROR) << "fail in GetAlignedPtr";
    return ptr;
}

GeTensor::GeTensor() {}

GeTensor::GeTensor(GeTensor &&other) noexcept
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(GeTensorImplPtr impl)
{
    UNUSED_VALUE(impl);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc)
{
    UNUSED_VALUE(tensor_desc);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const std::vector<uint8_t> &data)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const uint8_t *const data, const size_t size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(size);
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(GeTensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const Buffer &data)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(aligned_ptr);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const size_t size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::GeTensor(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg)
{
    UNUSED_VALUE(proto_owner);
    UNUSED_VALUE(proto_msg);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor::~GeTensor() = default;

void GeTensor::BuildAlignerPtrWithProtoData()
{
    MKI_LOG(ERROR) << "fail in BuildAlignerPtrWithProtoData";
}

const GeTensorDesc &GeTensor::GetTensorDesc() const
{
    MKI_LOG(ERROR) << "fail in GetTensorDesc";
    return DescReference();
}

GeTensorDesc &GeTensor::MutableTensorDesc()
{
    MKI_LOG(ERROR) << "fail in MutableTensorDesc";
    return DescReference();
}

GeTensorDesc &GeTensor::DescReference() const
{
    MKI_LOG(ERROR) << "fail in DescReference";
    return DescReference();
}

void GeTensor::SetTensorDesc(const GeTensorDesc &tensor_desc)
{
    MKI_LOG(ERROR) << "fail in SetTensorDesc";
    UNUSED_VALUE(tensor_desc);
}

graphStatus GeTensor::SetData(std::vector<uint8_t> &&data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const std::vector<uint8_t> &data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const uint8_t *const data, const size_t size)
{
    UNUSED_VALUE(data);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const Buffer &data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(const TensorData &data)
{
    UNUSED_VALUE(data);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus GeTensor::SetData(uint8_t *const data, const size_t size, const AlignedPtr::Deleter &delete_fuc)
{
    UNUSED_VALUE(data);
    UNUSED_VALUE(size);
    UNUSED_VALUE(delete_fuc);
    MKI_LOG(ERROR) << "fail in SetData";
    return GRAPH_SUCCESS;
}

graphStatus GeTensor::ResetData(uint8_t *const data, const size_t size, const AlignedPtr::Deleter &delete_fuc)
{
    UNUSED_VALUE(data);
    UNUSED_VALUE(size);
    UNUSED_VALUE(delete_fuc);
    return GRAPH_SUCCESS;
}

void GeTensor::ClearData()
{
    MKI_LOG(ERROR) << "fail in ClearData";
}

GeTensor GeTensor::Clone() const
{
    MKI_LOG(ERROR) << "fail in Clone";
    return GeTensor();
}

GeTensor::GeTensor(const GeTensor &other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in GeTensor";
}

GeTensor &GeTensor::operator=(const GeTensor &other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

GeTensor &GeTensor::operator=(GeTensor &&other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in operator";
    return *this;
}

std::shared_ptr<AlignedPtr> GeTensor::GetAlignedPtr()
{
    MKI_LOG(ERROR) << "fail in GetAlignedPtr";
    return nullptr;
}

const TensorData &GeTensor::GetData() const
{
    static TensorData tensorData;
    MKI_LOG(ERROR) << "fail in GetData";
    return tensorData;
}
TensorData &GeTensor::MutableData()
{
    static TensorData tensorData;
    MKI_LOG(ERROR) << "fail in MutableData";
    return tensorData;
}
// zero copy SetData
void GeTensor::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size)
{
    UNUSED_VALUE(aligned_ptr);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in SetData";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetSize(const GeTensorDesc &tensor_desc,
                                                                                int64_t &size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in GetSize";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetSize(GeTensorDesc &tensor_desc, const int64_t size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in SetSize";
}

int64_t TensorUtils::GetWeightSize(const GeTensorDesc &tensor_desc)
{
    UNUSED_VALUE(tensor_desc);
    MKI_LOG(ERROR) << "fail in GetWeightSize";
    return 0;
}

int64_t TensorUtils::GetWeightSize(const GeTensor &tensor)
{
    UNUSED_VALUE(tensor);
    MKI_LOG(ERROR) << "fail in GetWeightSize";
    return 0;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t TensorUtils::GetWeightSize(const ConstGeTensorPtr &tensor_ptr)
{
    UNUSED_VALUE(tensor_ptr);
    MKI_LOG(ERROR) << "fail in GetWeightSize";
    return 0;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint8_t *TensorUtils::GetWeightAddr(const ConstGeTensorPtr &tensor_ptr,
                                                                                   const uint8_t *const base)
{
    UNUSED_VALUE(tensor_ptr);
    UNUSED_VALUE(base);
    MKI_LOG(ERROR) << "fail in GetWeightAddr";
    return nullptr;
}

uint8_t *TensorUtils::GetWeightAddr(const GeTensor &tensor, const uint8_t *const base)
{
    UNUSED_VALUE(tensor);
    UNUSED_VALUE(base);
    MKI_LOG(ERROR) << "fail in GetWeightAddr";
    return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetWeightSize(GeTensorDesc &tensor_desc,
                                                                               const int64_t size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in SetWeightSize";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetReuseInput(const GeTensorDesc &tensor_desc,
                                                                                      bool &flag)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(flag);
    MKI_LOG(ERROR) << "fail in GetReuseInput";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInput(GeTensorDesc &tensor_desc,
                                                                               const bool flag)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(flag);
    MKI_LOG(ERROR) << "fail in SetReuseInput";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetOutputTensor(const GeTensorDesc &tensor_desc,
                                                                                        bool &flag)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(flag);
    MKI_LOG(ERROR) << "fail in GetOutputTensor";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetOutputTensor(GeTensorDesc &tensor_desc,
                                                                                 const bool flag)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(flag);
    MKI_LOG(ERROR) << "fail in SetOutputTensor";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDeviceType(const GeTensorDesc &tensor_desc,
                                                                                      DeviceType &type)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(type);
    MKI_LOG(ERROR) << "fail in GetDeviceType";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDeviceType(GeTensorDesc &tensor_desc,
                                                                               const DeviceType type)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(type);
    MKI_LOG(ERROR) << "fail in SetDeviceType";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetInputTensor(const GeTensorDesc &tensor_desc,
                                                                                       bool &flag)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(flag);
    MKI_LOG(ERROR) << "fail in GetInputTensor";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetInputTensor(GeTensorDesc &tensor_desc,
                                                                                const bool flag)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(flag);
    MKI_LOG(ERROR) << "fail in SetInputTensor";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRealDimCnt(const GeTensorDesc &tensor_desc,
                                                                                      uint32_t &cnt)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(cnt);
    MKI_LOG(ERROR) << "fail in GetRealDimCnt";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRealDimCnt(GeTensorDesc &tensor_desc,
                                                                               const uint32_t cnt)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(cnt);
    MKI_LOG(ERROR) << "fail in SetRealDimCnt";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetReuseInputIndex(const GeTensorDesc &tensor_desc, uint32_t &idx)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(idx);
    MKI_LOG(ERROR) << "fail in GetReuseInputIndex";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInputIndex(GeTensorDesc &tensor_desc,
                                                                                    const uint32_t idx)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(idx);
    MKI_LOG(ERROR) << "fail in SetReuseInputIndex";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDataOffset(const GeTensorDesc &tensor_desc,
                                                                                      int64_t &offset)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(offset);
    MKI_LOG(ERROR) << "fail in GetDataOffset";
    return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDataOffset(GeTensorDesc &tensor_desc,
                                                                               const int64_t offset)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(offset);
    MKI_LOG(ERROR) << "fail in SetDataOffset";
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRC(const GeTensorDesc &tensor_desc,
                                                                              uint32_t &rc)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(rc);
    MKI_LOG(ERROR) << "fail in GetRC";
    return 0;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRC(GeTensorDesc &tensor_desc, const uint32_t rc)
{
    UNUSED_VALUE(tensor_desc);
    MKI_LOG(ERROR) << "fail in SetRC";
    UNUSED_VALUE(rc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool TensorUtils::IsOriginShapeInited(const GeTensorDesc &tensor_desc)
{
    UNUSED_VALUE(tensor_desc);
    MKI_LOG(ERROR) << "fail in IsOriginShapeInited";
    return true;
}

GeTensor TensorUtils::CreateShareTensor(const GeTensor &other)
{
    GeTensor tensor;
    ShareTensor(other, tensor);
    MKI_LOG(ERROR) << "fail in CreateShareTensor";
    return tensor;
}

GeTensor TensorUtils::CreateShareTensor(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr,
                                        const size_t size)
{
    UNUSED_VALUE(tensor_desc);
    UNUSED_VALUE(aligned_ptr);
    UNUSED_VALUE(size);
    MKI_LOG(ERROR) << "fail in CreateShareTensor";
    return GeTensor();
}

void TensorUtils::ShareTensor(const GeTensor &from, GeTensor &to)
{
    UNUSED_VALUE(from);
    UNUSED_VALUE(to);
    MKI_LOG(ERROR) << "fail in ShareTensor";
}

void TensorUtils::ShareTensorData(const TensorData &from, TensorData &to)
{
    UNUSED_VALUE(from);
    UNUSED_VALUE(to);
    MKI_LOG(ERROR) << "fail in ShareTensorData";
}

TensorData TensorUtils::CreateShareTensorData(const TensorData &other)
{
    UNUSED_VALUE(other);
    MKI_LOG(ERROR) << "fail in CreateShareTensorData";
    return TensorData();
}

void TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, TensorData &to)
{
    UNUSED_VALUE(ptr);
    UNUSED_VALUE(size);
    UNUSED_VALUE(to);
    MKI_LOG(ERROR) << "fail in ShareAlignedPtr";
}

void TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, GeTensor &to)
{
    UNUSED_VALUE(ptr);
    UNUSED_VALUE(size);
    UNUSED_VALUE(to);
    MKI_LOG(ERROR) << "fail in ShareAlignedPtr";
}

// UT
void TensorUtils::CopyTensor(const GeTensor &from, GeTensor &to)
{
    UNUSED_VALUE(from);
    UNUSED_VALUE(to);
    MKI_LOG(ERROR) << "fail in CopyTensor";
}
} // namespace ge
