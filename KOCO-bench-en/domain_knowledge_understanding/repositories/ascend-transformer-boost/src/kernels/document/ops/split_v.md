# Split_v

## 参数说明

- **OpName**：SplitOperation
- **PARAM**
  **Param Type**：OpParam::Split

| 名称      | 类型             | 描述            |
| --------- | ---------------- | --------------- |
| splitVDim | SVector<int32_t> | split的维度     |
| splitSize | SVector<int32_t> | split的切分大小 |
| splitNum  | int32            | split的切分次数 |

- **In/Out Tensor**

| 名称      | 类型 | dims                                                             | dtype                    | format | 描述        |
| --------- | ---- | ---------------------------------------------------------------- | ------------------------ | ------ | ----------- |
| x         | In   | [$d_0$, $d_1$, ...,$d_{splitVDim[0]}$, ..., $d_n$]       | float16、bfloat16、int64 | ND     |             |
| splitSize | In   | [$splitSize_0$, $splitSize_1$, ...,$splitSize_{splitNum}$] | int32_t                  | ND     |             |
| splitVDim | In   | [$splitVDim$]                                                  | int32_t                  | ND     |             |
| z         | Out  | [$d_0$, $d_1$, ...,$d_{splitSize[0]}$]                     | 与x一致                  | ND     | 与x相同维度 |
| z1        | Out  | [$d_0$, $d_1$, ...,$d_{splitSize[1]}$]                     | 与x一致                  | ND     | 与x相同维度 |
| z2        | Out  | [$d_0$, $d_1$, ...,$d_{splitSize[2]}$]                     | 与x一致                  | ND     | 与x相同维度 |

注：int64暂时只支持2分割

## 功能描述

- 算子功能：对splitDim维度进行3不等长切分或2不等长切分。
  MAX_SUPPORT_SPLIT_NUM最大split个数为3。

## 示例

```
输入：
    splitSize = [5, 3, 1]
    splitVDim = 0
    splitNum = 3

    x = [3, 3, 3, 3, 3, 3, 3, 3，3]
 
输出：
    z = [3, 3, 3, 3, 3]

    z1 = [3, 3, 3]
  
    z2 = [3]
 
```

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
