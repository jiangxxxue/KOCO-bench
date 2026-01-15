# Fill

## 参数说明

- **OpName**：FillOperation
- **PARAM**
  **Param Type**：OpParam::Fill

| 名称          | 类型                | 描述                                |
| ------------- | ------------------- | ----------------------------------- |
| value         | SVector `<float>` | 需要填充的元素                      |
| outDim        | SVector<int32_t>    | 指定输出tensor的shape，非负数       |
| outTensorType | enum         | 指定输出tensor的type，默认为float16 |

- **In/Out Tensor**

| 名称 | 类型 | dims                             | dtype             | format |
| ---- | ---- | -------------------------------- | ----------------- | ------ |
| y    | Out  | [$d_0$, $d_1$, ..., $d_n$] | bfloat16、float16 | ND     |

## 功能描述

- 算子功能：生成一个指定shape的Tensor,并将其填充为value。
  shape通过outDim参数描述。

## 示例

```
输入：
    value = 5.0

    outDim = [2, 3]

输出:
    y = [[5.0, 5.0, 5.0],
         [5.0, 5.0, 5.0]]   
```

#### 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
