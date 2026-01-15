# AdalayerNorm
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | RMS_NORM |
|epsilon|float|Layernorm类算子属性|
|inGamma|bool|True|
|inBeta|bool|True|
|inRes|bool|False|
|outMean|bool|False|
|outVarience|bool|False|
|outResQuant|bool|False|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  x     |In  | [$d0$, $d1$, $d2$]|float16、bfloat16|ND||
| gamma |In  | [$d_0$, $d_1$, $d2$]/[$d_0$, $d2$]|与x一致|ND|$d_0$, $d_1$为对应维度$d0$, $d1$的因数|
| beta  |  In| [$d_0$, $d_1$, $d2$]/[$d_0$, $d2$]|与x一致|ND|
| z |Out | [$d0$, $d1$, $d2$]|与x一致|ND|


## 功能描述
- 算子功能：针对layerNorm算子，支持gamma和beta在多batch和seqlen上实现不同的归一化操作。
- 计算公式：$z_i=\frac{x_i-\mu_B}{\sqrt{\delta_B^2+\epsilon}}(gamma+ 1)+beta$

其中
$\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i \quad \delta_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$ 代表输入在$d2$轴的均值和方差

## 示例
```
输入：
	x.dims = [B, K, D]
	gamma.dims = [b, k, D]/[b, D]
    beta.dims = [b, k, D]/[b, D]
 
输出： 
	z.dims = [B, K, D]
```
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|输入tensor最后一维（n）必须32byte对齐 |
|  ascend310p|输入tensor最后一维（n）必须32byte对齐 |
||310p中不支持bfloat16类型数据|