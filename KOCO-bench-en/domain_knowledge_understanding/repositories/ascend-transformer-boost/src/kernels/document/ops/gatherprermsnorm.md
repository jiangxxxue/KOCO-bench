# GatherPreRmsNorm

## 参数说明

- **OpName**：NormOperation
- **PARAM**
  **Param Type**：OpParam::Norm

| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| normType | enum  | GATHER_PRE_RMS_NORM |
|epsilon|float|LayerNorm/RmsNorm类算子属性|

- **In/Out Tensor**

|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  X     |In  | [$d_m$, $d_n$]|float16、bfloat16|ND|2维|
| res_in |  In| [$d_{res}$, $d_n$]| 与X一致|ND|2维，$d_m$、$d_{res}$可能不相等|
| indices |In  | [$d_m$]|int32|ND|1维，元素取值范围为[0, $d_{res}$)|
| gamma |In  | [1, $d_n$]或[$d_n$]|与X一致|ND|1维或2维|
| Y |Out  | [$d_m$, $d_n$]|float32|ND|dims与X一致|
| res_out |Out  | [$d_m$, $d_n$]|与X一致|ND|dims与X一致|

## 功能描述

- 算子功能：根据输入indices中各个索引值从输入res_in取出对应的行数据，然后再与输入X进行相加得到输出res_out，最后通过RmsNorm操作获得输出Y。
- 计算过程：
  - Gather操作：根据indices中的值索引$res_{in}$中对应的行，得到$res_{inAfter}$
  - Add操作：将输入X和$res_{inAfter}$相加，得到输出$res_{out}$
  - RmsNorm操作：
    $Y = RmsNorm(res_{out})$
    
    $RmsNorm(x) = \frac{x}{RMS(x)} \times gamma$
    
    $RMS(x) = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}x_i^2+epsilon}$

## 支持芯片型号

|芯片名称|约束 | 
| ------------ | ------------ | 
|  800 A2/A3|输入tensor的最后一维（$d_n$）必须32byte对齐且不大于7680|
