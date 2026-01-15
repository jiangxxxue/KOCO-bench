# Expand

## 参数说明

- **OpName**：ExpandOperation
- **PARAM**
  **Param Type**：OpParam::Expand

| 名称  | 类型  | 描述           |
| ----- | ----- | -------------- |
| shape | int64 | 不小于输入维度 |

- **In/Out Tensor**

| 名称 | 类型 | dims                             | dtype             | format |
| ---- | ---- | -------------------------------- | ----------------- | ------ |
| x    | In   | [$d_0$, $d_1$, ..., $d_n$] | bfloat16、float16 | ND     |
| y    | Out  | shape                            | 与x相同           | ND     |

## 功能描述

- 算子功能：将输入算子x的shape扩展到指定shape。
  x中对应维度值不能大于指定的扩展后的shape维度值(除去最大维度为1);
  x中能扩展的维度shape数值必须为1;

## 示例

```
输入：
    x.Dim = [2, 1, 1]

    shape = [2, 3, 3]
  
输出:
    y.Dim = [2, 3, 3]
```

#### 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
