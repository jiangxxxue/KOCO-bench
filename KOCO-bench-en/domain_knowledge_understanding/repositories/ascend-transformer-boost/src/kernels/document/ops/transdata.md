# Transdata

## 参数说明

- **OpName**：TransdataOperation
- **PARAM**
  **Param Type**：OpParam::Transdata

| 名称          | 类型      | 描述                                                                                                   |
| ------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| transdataType | enum      | 数据转换类型枚举,FRACTAL_NZ_TO_ND：NZ转ND                                                              |
|               |           | ND_TO_FRACTAL_NZ：ND转NZ                                                                               |
| outCrops      | int64数组 | NZ转ND时用于去除ND转NZ时的对齐padding,设置为原始ND维度的最后2维;默认值是{0, 0},表示不进行padding去除。 |

- **In/Out Tensor**
- TransdataNdToNz

|名称 | 类型  | dims  | dtype  |format|
| ------------ | ------------ | ------------ | ------------ |
| x|In |  [$b$, $m$, $n$]|float16、int8|ND|
| y|Out |[$b$, $n_1$, $m1m0$, $n0$]|与x一致|NZ|

- TransdataNzToNd

| 名称 | 类型 | dims                                  | dtype                   | format |
| ---- | ---- | ------------------------------------- | ----------------------- | ------ |
| x    | In   | [$b$, $n_1$, $m_1m_0$, $n_0$] | bfloat16、float16、int8 | NZ     |
| y    | Out  | [$b$, $m$, $n$]                 | 与x一致                 | ND     |

## 功能描述

- 算子功能：$ND$转$NZ$。
- 算子公式：
  过程为[b, m, n] -> padding [b, m1m0, n1n0] -> reshape [b, m1m0, n1, n0] -> transpose [b, n1, m1m0, n0]
  本算子使用的$NZ$的$dims$约定表示方式：[b, n1, m1m0, n0],对应的$ND$的$dims$是[b, m, n], 其中:
  $b$表示$batch$, 如果$batch$为1, 该维度为1, 不可省略;
  如果$batch$有多个, 该维度为所有$batch$维度合轴的结果;
  $m_0/n_0$表示对齐位, float16为16;
  $m_1m_0$表示原始$ND$的$m$维度经过对齐位向上对齐;
  $n_1$表示原始$ND$的$n$维度经过对齐位向上对齐后, 除以$n_0$的余数。
- 算子功能：$NZ$转$ND$。
- 算子描述：NZ转ND，过程为ND转NZ的反向，其中最后一步padding的反向依赖参数中的outCrops。

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
