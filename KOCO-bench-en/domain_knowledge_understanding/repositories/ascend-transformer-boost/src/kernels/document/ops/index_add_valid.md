# IndexAddValid
## 参数说明
- **OpName**：IndexOperation
- **PARAM**
**Param Type**：OpParam::Index
 
| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| IndexType | enum | INDEX_ADD_VALID |
 
- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  | format | 描述 |
| ------------ | ------------ | ------------ | ------------ |------------ |
| var | In/Out | [$d_1$, $d_2$] | float16 | ND | 被加数；输入为零，原地被加后作为输出； |
| indices | In | [$d_0$ ] | int32_t | ND | 索引 |
| updates | In | [$d_0$, $d_2$] | float16 | ND | 加数；根据indices的值加到var对应位置； |
| validIndicesNum | In | [1 ] | int32_t | ND | indices的有效长度； |
 
## 功能描述
- 算子功能：$根据indices，将updates的i维上的值加到var的indices[i]维。indices中只有前validIndicesNum个值有效。$
- 计算公式：$var[indices[i]] += updates[i]	（0 <= i < validIndicesNum）$
 
## 示例
```
    输入:
        indices = {[0,2,2,-1,-1]}
        validIndicesNum = {[3]}
        updates = {[0,1,2], [1,2,3], [2,3,4], [5,6,7], [6,7,8]}
        var = {[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]}
    输出:
        var = {[0,1,2], [0,0,0], [3,5,7], [0,0,0], [0,0,0]}
```
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
| ascend910b | d2小于8192 |