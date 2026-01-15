# RmsNormQuant
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | RMS_NORM |
|epsilon|float|LayerNorm/RmsNorm类算子属性|
|inGamma|bool|True|
|inBeta|bool|True|
|outRes|bool|False|
|inRes|bool|False|
|inNormBias|bool|False|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  X     |In  | [$d_0$, $d_1$, ..., $d_n$]|float16、bfloat16|ND||
| gamma |  In| [1, $d_n$] 或 [$d_n$]| 与X一致|ND||
| beta |In  | [1, $d_n$] 或 [$d_n$]|与X一致|ND|与gamma的shape一致|
| scale |In  | [1]|与X一致|ND||
| offset |In  | [1]|int8|ND||
| Y |Out  | [$d$, $d_1$, ..., $d_n$]|int8|ND|与输入X的shape一致|

## 功能描述
- 算子功能：将输入X进行rmsnorm操作，把得到的结果再量化为int8类型的输出Y。
- 计算公式（beta是可选参数）：

  $Y = Quant(\frac{X}{RMS(X)}*gamma+beta$)

  $RMS(x) = \sqrt{ \frac{1} {n} \sum\limits_{i = 1}^n {x_i^2}+epsilon}$
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b| 无 |
|  ascend310p|310p中不支持bfloat16类型数据|