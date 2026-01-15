# TransdataOp使用说明

## TransdataOp

### 功能

`TransdataOp` 函数用于创建并执行一个 `transdata` 操作，将输入的 ND 格式的 `tensor` 转换为 `FRACTAL_NZ` （后文简称为NZ）格式的 `tensor`。

### 函数介绍

```CPP
atb::Status TransdataOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
                        const aclDataType tensorType, atb::Tensor &outTensor, std::vector<int64_t> shape)
```

#### 输入参数

| 参数名称| 含义 |
| --- | --- |
| `contextPtr` | 指向 `atb::Context `的指针，用于提供上下文信息 |
|`stream`  |  用于执行操作的流 |
|  `inTensor`| 待转换的 ND 格式的tensor |
|`tensorType`| 输入 tensor 的数据类型|
|`outTensor`|输入 tensor 的数据类型|
|`outTensor`|输出的 `FRACTAL_NZ` 格式的 tensor，函数执行后将被填充|
|`shape`|输出 tensor的维度|

#### 返回值

`atb::Status`: 函数执行的状态码，表示操作是否成功。如果成功，返回 `atb::ErrorType::NO_ERROR`，否则返回相应的错误码。

## GetShape

### 功能

`GetShape` 函数用于根据输入的 `inShape` 判断其是 NZ格式的`shape` 还是ND 格式的 `shape`，然后分别计算出两种格式下的 `shape`。

### 函数介绍

```
atb::Status GetShape(const aclDataType tensorType, const std::vector<int64_t> &inShape, std::vector<int64_t> &ndShape,
                     std::vector<int64_t> &nzShape)
```

#### 输入参数

| 参数名称| 含义 |
| --- | --- |
|  `tensorType`|输入 tensor 的数据类型，不同数据类型的tensor的格式转换方式不同  |
|  `inShape`|  输入 tensor 的维度|
|  `ndShape`|  输出的 ND 格式的 shape|
|`nzShape`  | 输出的 NZ 格式的 shape |

#### 返回值

`atb::Status`: 函数执行的状态码，表示操作是否成功。如果成功，返回 `atb::ErrorType::NO_ERROR`，否则返回相应的错误码。

## Transdata使用说明
注意：
- 使用`TransdataOp`转换ND格式的tensor时,需要保证传入的tensor是ND格式，因此创建初始的tensor时，应当使用ND格式下的shape，而使用`TransdataOp`函数时传入最终想得到的NZ格式下tensor的shape；
- 在`CreateTensorFromVector`中只传入了一种格式下的shape，因此需要使用GetShape函数得到ND和NZ两种数据格式下的shape

使用`TransdataOp`进行ND格式tensor转NZ格式tensor的`CreateTensorFromVector`的内容如下：

```C++
/**
 * @brief 简单封装，拷贝vector data中数据以创建tensor
 * @details 用于创建outTensorType类型的tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param data 输入vector数据
 * @param outTensorType 期望输出tensor数据类型
 * @param format 输出tensor的格式，即NZ，ND等
 * @param shape 输出tensor的shape
 * @param outTensor 返回创建的tensor
 */
template <typename T>
atb::Status CreateTensorFromVector(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
                                   const aclDataType outTensorType, const aclFormat format, std::vector<int64_t> shape,
                                   atb::Tensor &outTensor, const aclDataType inTensorType = ACL_DT_UNDEFINED)
{
    atb::Tensor tensor;
    aclDataType intermediateType;
    switch (outTensorType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_DOUBLE:
            intermediateType = aclDataType::ACL_FLOAT;
            break;
        default:
            intermediateType = outTensorType;
    }
    if (inTensorType == outTensorType && inTensorType != ACL_DT_UNDEFINED) {
        intermediateType = outTensorType;
    }
    aclFormat tensorFormat = format;
    std::vector<int64_t> ndShape, nzShape;
    if (intermediateType != outTensorType && format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        tensorFormat = aclFormat::ACL_FORMAT_ND;
        CHECK_STATUS(GetShape(outTensorType, shape, ndShape, nzShape));
    } else {
        ndShape.assign(shape.begin(), shape.end());
    }
    CHECK_STATUS(CreateTensor(intermediateType, tensorFormat, shape, tensor));
    CHECK_STATUS(aclrtMemcpy(tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(),
                             ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_STATUS(CreateTensor(outTensorType, aclFormat::ACL_FORMAT_ND, ndShape, outTensor));
    if (intermediateType == outTensorType) {
        // 原始创建的tensor类型，不需要转换
        outTensor = tensor;
        return atb::ErrorType::NO_ERROR;
    }
    CHECK_STATUS(CastOp(contextPtr, stream, tensor, outTensorType, outTensor));
    if (format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        CHECK_STATUS(TransdataOp(contextPtr, stream, outTensor, outTensorType, outTensor, nzShape));
    }

    return atb::ErrorType::NO_ERROR;
}
```