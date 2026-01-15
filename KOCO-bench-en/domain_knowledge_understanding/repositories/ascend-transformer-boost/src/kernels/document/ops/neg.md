# Neg

## 参数说明

- **OpName**：ElewiseOperation
- **PARAM**
  **Param Type**：OpParam::Elewise

| 名称        | 类型 | 描述        |
| ----------- | ---- | ----------- |
| elewiseType | enum | ELEWISE_NEG |

- **In/Out Tensor**

| 名称 | 类型 | dims                             | dtype             | format |
| ---- | ---- | -------------------------------- | ----------------- | ------ |
| x    | In   | [$d_0$, $d_1$, ..., $d_n$] | float16、bfloat16 | ND     |
| y    | Out  | [$d_0$, $d_1$, ..., $d_n$] | 与x一致           | ND     |

## 功能描述

- 算子功能：Tensor内对每个element取反，如11288转换为-11288。
- 计算公式：$y=-x$

## 示例

```
输入：
    x = [[2.2, 3.4], [4.8, 5.9]]
输出：
    y = [[-2.2, -3.4], [-4.8, -5.9]]  
```

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
