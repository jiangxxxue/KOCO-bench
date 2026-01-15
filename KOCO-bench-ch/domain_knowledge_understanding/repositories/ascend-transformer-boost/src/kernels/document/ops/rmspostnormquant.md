# RmsPostNormQuant
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | RMS_NORM |
|epsilon|float|LayerNorm/RmsNorm类算子属性|
|inGamma|bool|True|
|inBeta|bool|False|
|outRes|bool|True|
|inRes|bool|True|
|inNormBias|bool|False|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  X     |In  | [$d_0$, $d_1$, ..., $d_n$]|float16、bfloat16|ND||
| res_in |  In| [$d_0$, $d_1$, ..., $d_n$]| 与X一致|ND|与输入X的shape一致|
| gamma |In  |[1, $d_n$] 或 [$d_n$]|与X一致|ND||
| scale |In  | [1]|与X一致|ND||
| offset |In  | [1]|int8|ND||
| Y |Out  | [$d_0$, $d_1$, ..., $d_n$]|int8|ND|与输入X的shape一致|
| res_out |Out  | [$d_0$, $d_1$, ..., $d_n$]|与X一致|ND|与输入X的shape一致|

## 功能描述
- 算子功能：X和res_in相加后的结果经过RmsNorm操作得到输出res_out，再经过Quant操作得到输出Y。
- 计算公式：

  $res_{out}=RmsNorm(X+res_{in})$

  $Y = Quant(res_{out})$

  $RmsNorm(x) = \frac{x}{RMS(x)} \times gamma$
  
  $RMS(x) = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}x_i^2+epsilon}$
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|输入和输出tensor的最后一维（$d_n$）必须32byte对齐|
|  ascend310p|输入和输出tensor的最后一维（$d_n$）必须32byte对齐|
||310p中不支持bfloat16类型数据|