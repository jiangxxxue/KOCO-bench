# Index

## 参数说明

- **OpName**：IndexOperation
- **PARAM**
  **Param Type**：OpParam::Index

| 名称      | 类型    | 描述                                                                             |
| --------- | ------- | -------------------------------------------------------------------------------- |
| indexType | enum    | Index类型（对应各个算子）INDEX_UNDEFINED = 0,INDEX_ADD 对应 inplaceIndexAdd 算子 |
| axis      | int64_t | 可选参数，将在哪个轴上进行操作（默认为0）                                        |

- **In/Out Tensor**

| 名称    | 类型    | dims                             | dtype            | format | 描述                                |
| ------- | ------- | -------------------------------- | ---------------- | ------ | ----------------------------------- |
| var     | In &Out | [$d_0$, $d_1$, ..., $d_n$] | bloat16、float16 | ND     |                                     |
| indices | In      | [$d_0$]                        | int32            | ND     |                                     |
| updates | In      | [$d_0$, $d_1$, ..., $d_n$] | bloat16、float16 | ND     | axis维度上的大小须与indices大小相同 |
| alpha   | In      | [1]                              | bloat16、float16 | ND     |                                     |

## 功能描述

- 算子功能：对于输入var,按axis维度根据索引表indices,原地累加alpha次updates值。

## 示例

```
输入：
    axis = 1

    var = [[1, 2, 3], 
           [4, 5, 6], 
           [7, 8 ,9]]

    indices = [-1, 0]
  
    updates = [[1, 2],
               [1, 2],
               [1, 2]]
    alpha = [1]

输出：
    var = [[3, 2, 4], 
           [6, 5, 7], 
           [9, 8 ,10]]  
```

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
