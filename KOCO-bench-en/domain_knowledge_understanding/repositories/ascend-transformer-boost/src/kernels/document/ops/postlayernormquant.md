# PostLayerNormQuant
## 参数说明
- **OpName**：NormOperation
- **PARAM**
**Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | LAYER_NORM |
|epsilon|float|Layernorm类算子属性|
|inGamma|bool|True|
|inBeta|bool|True|
|inRes|bool|True|
|outMean|bool|False|
|outVarience|bool|False|
|outResQuant|bool|True|
- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  x     |In  | [$d_0$, $d_1$, ..., $d_n$]|float16|ND||
| gamma  |  In| [1, $d_n$]| 与x一致|ND|2维度|
| beta   | In| [1, $d_n$] | 与x一致|ND|2维度|
| res_in |In  | [$d_0$, $d_1$, ..., $d_n$]|与x一致|ND||
| scale |In  | [1]|与x一致|ND||
| offset |In  | [1]|int8|ND||
| res_quant |Out  | [$d_0$, $d_1$, ..., $d_n$]|与offset一致|ND|与输入维度相同|
| result |Out  | [$d_0$, $d_1$, ..., $d_n$]|与offset一致|ND|与输入维度相同|

## 功能描述
- 算子功能：将postlayernorm的结果量化为int8类型。
## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|输入tensor最后一维（n）必须32byte对齐 |
|  ascend310p|输入tensor最后一维（n）必须32byte对齐|