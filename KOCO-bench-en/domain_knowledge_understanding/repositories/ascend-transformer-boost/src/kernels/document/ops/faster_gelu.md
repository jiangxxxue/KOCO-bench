# FasterGelu

## 参数说明

- **OpName**: ActivationOperation
- **Param Type**: OpParam::Activation

| 名称           | 类型 | 描述                           |
| -------------- | ---- | ------------------------------ |
| activationType | enum | ACTIVATION_FASTER_GELU_FORWARD |

- **In/Out Tensor**

| 名称 | 类型 | dims              | dtype                      | format |
| ---- | ---- | ----------------- | -------------------------- | ------ |
| x    | In   | [$d_0,...d_n$ ] | float32、float16、bfloat16 | ND/NZ  |
| y    | Out  | [$d_0,...d_n$ ] | float32、float16、bfloat16 | ND/NZ  |

## 功能描述

- 算子功能，element wise类型算子，快速运算Gelu函数，比fast_gelu算子运算更快
- 算子公式

$$
FasterGelu(x)=\frac{x}{1+e^{-1.702x}}
$$

## 示例

```
输入：
x = [-1.9173578  3.5180674  3.2089603 ...  2.4093502  1.6536514  3.4666355]
输出：
y = [-0.0707,  3.5093,  3.1954,  ...,  2.3701,  1.5601,  3.4572]
```

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
