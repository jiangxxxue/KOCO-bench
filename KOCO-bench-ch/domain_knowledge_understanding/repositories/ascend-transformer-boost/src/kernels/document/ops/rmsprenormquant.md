# RmsPreNormQuant
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
|inNormBias|bool|True|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  X     |In  | [$d_0$, $d_1$, ..., $d_n$]|float16|ND||
| res_in |  In| [$d_0$, $d_1$, ..., $d_n$]| 与X一致|ND|与输入X的shape一致|
| gamma |In  | [1, $d_n$] 或 [$d_n$]|与X一致|ND||
| bias |In  | [1, $d_n$] 或 [$d_n$]|与X一致|ND|与gamma的shape一致|
| scale |In  | [1]|与X一致|ND||
| offset |In  | [1]|int8|ND||
| Y |Out  | [$d$, $d_1$, ..., $d_n$]|int8|ND|与输入X的shape一致|
| res_out |Out  | [$d$, $d_1$, ..., $d_n$]|float16|ND|与输入X的shape一致|

## 功能描述
- 算子功能：将输入X和res_in相加得到输出res_out，再对输出res_out进行RmsNorm操作和Quant操作得到输出Y。
- 计算公式：

  $ res_{out} = X+res_{in}$

  $Y = Quant(\frac{res_{out}}{RMS(res_{out})}*gamma+bias$)

  $RMS(x) = \sqrt{ \frac{1} {n} \sum\limits_{i = 1}^n {x_i^2}+epsilon}$
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|输入tensor最后一维（n）必须32byte对齐|
|  ascend310p|输入tensor最后一维（n）必须32byte对齐|
||310p中不支持bfloat16类型数据|
