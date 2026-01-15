# CohereLayerNorm
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm
 
| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | LAYER_NORM |
|epsilon|float|Layernorm类算子属性|
|inGamma|bool|True|
|inBeta|bool|False|
|inRes|bool|False|
|outMean|bool|False|
|outVarience|bool|False|
|outResQuant|bool|False|
- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  x     |In  | [$d_0$, $d_1$, ..., $d_n$]|float16、bfloat16|-||
| gamma  |  In| [$d_{n-1}$, $d_n$]| 与x一致|-|2维度|
| result |Out  | [$d_0$, $d_1$, ..., $d_n$]|与x一致|-||
 
注：gamma的shape是2维度，其形状为 [$d_{n-1}$, $d_n$]，其中 $d_{n-1}$ 表示x倒数第二维的大小，$d_n$ 表示x最后一维的大小
 
 
## 功能描述
- 算子功能：针对Command R Plus模型，对多batch数据用于表示headsize的最后一维进行归一化操作。
- 计算公式：$y=\frac{x-E[x]}{√Var[x]+ε}*γ$
 
E[x]为均值, Var[x]为方差, ε为epsilon, γ为权重
## 示例
```
输入：
	x.dims = [a, b, c, d]
	gamma = [c, d]
 
输出： 
	result.dims = [a, b, c, d]
```
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|输入tensor最后一维（n）必须32byte对齐|
|  ascend310p|输入tensor最后一维（n）必须32byte对齐|
|  |310p中不支持bfloat16类型数据|