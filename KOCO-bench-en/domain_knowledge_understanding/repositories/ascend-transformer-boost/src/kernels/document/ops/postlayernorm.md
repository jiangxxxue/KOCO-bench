# PostLayerNorm
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | LAYER_NORM |
|epsilon|float|Layernorm类算子属性|
|zoomScaleValue|float|Postlayernorm算子属性，缩放因子|
|inGamma|bool|True|
|inBeta|bool|True|
|inRes|bool|True|
|outMean|bool|False|
|outVarience|bool|False|
|outResQuant|bool|False|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  x     |In  | [$d_0$, $d_1$, ..., $d_n$]|float16|ND||
| res_in |In  | [$d_0$, $d_1$, ..., $d_n$]|float16|ND||
| gamma  |  In| [1, $d_n$]| 与x一致|ND|2维度|
| beta   | In| [1, $d_n$] | 与x一致|ND|2维度|
| result |Out  | [$d_0$, $d_1$, ..., $d_n$]|int8|ND|与输入维度相同|

## 功能描述
- 算子功能：同layernorm相似，在输入layernorm之前做了一次residual add。
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|输入tensor最后一维（n）必须32byte对齐|
|  ascend310p|输入tensor最后一维（n）必须32byte对齐|