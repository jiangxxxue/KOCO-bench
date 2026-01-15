# LayerNorm
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | LAYER_NORM |
|beginNormAxis|int32|Layernorm算子属性归一化的维度 默认值为1, 从第几维开始norm, 同时决定输入gamma和beta维度|
|beginParamsAxis|int32|Layernorm算子属性归一化的维度 默认值为1, 从第几维开始norm, 同时决定输入gamma和beta维度|
|epsilon|float|Layernorm类算子属性|
|inGamma|bool|True|
|inBeta|bool|True|
|inRes|bool|False|
|outMean|bool|True|
|outVarience|bool|True|
|outResQuant|bool|False|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  x     |In  | [$d_0$, $d_1$, ..., $d_n$]|float、float16、bfloat16|ND||
| gamma  |  In| [beginParamsAxis:]| 与x一致|ND|根据begin_norm_axis属性定维度|
| beta   | In| [beginParamsAxis:] | 与x一致|ND|根据begin_norm_axis属性定维度|
| result |Out  | [$d_0$, $d_1$, ..., $d_n$]|与x一致|ND||
|mean    |Out|[$d_0$, $d_1$, ..., $d_n$]|与x一致|ND|暂时不使用|
|varience|Out|[$d_0$, $d_1$, ..., $d_n$]|与x一致|ND|暂时不使用|

注：gamma和beta的shape根据norm的维度调整。
gamma, beta 根据norm的输入shape和属性的输入shape为[begin_params_axis:]
mean，varience的维度 根据 begin_norm_axis 进行降维
## 功能描述
- 算子功能：对单batch数据进行归一化操作。
- 计算公式：$y=\frac{x-E[x]}{√Var[x]+ε}*γ +β$

E[x]为均值, Var[x]为方差, ε为epsilon, γ为权重, β为偏置
## 示例
```
输入：
	begin_norm_axis = 2

	x.dims = [a, b, c, d]
	gamma = [c, d]
	beta.dims = [c, d]

输出： 
	result.dims = [a, b, c, d]
```
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|无 |
|  ascend310p|310p中不支持bfloat16类型数据|