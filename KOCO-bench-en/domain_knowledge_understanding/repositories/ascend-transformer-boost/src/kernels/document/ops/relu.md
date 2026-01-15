# Relu

## 参数说明

- **OpName**：ActivationOperation
- **PARAM**
  **Param Type**：OpParam::Activation

| 名称           | 类型 | 描述            |
| -------------- | ---- | --------------- |
| activationType | enum | ACTIVATION_RELU |

- **In/Out Tensor**

| 名称 | 类型 | dims                             | dtype   | format |
| ---- | ---- | -------------------------------- | ------- | ------ |
| x    | In   | [$d_0$, $d_1$, ..., $d_n$] | float32 | ND     |
| y    | Out  | [$d_0$, $d_1$, ..., $d_n$] | 与x一致 | ND     |

## 功能描述

- 算子功能：去输入中的负数并保留正值。
- 计算公式：$y=x<0?0:x$

## 示例

```
输入：
	x = [2.0,-1.0,2.0,-3.0]
输出：
	y = [2.0,0.0,2.0,0.0]
```

## 支持芯片型号

| 芯片名称   | 约束 |
| ---------- | ---- |
| ascend910b | 无   |
| ascend310p | 无   |
