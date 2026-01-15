# Gating
## 参数说明
- **OpName**：GatingOperation
- **PARAM**
**Param Type**：OpParam::Gating
 
| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| headNum | int32_t  | 每个token选中的专家（专家即权重，每个token可选多个权重，每个权重擅长的领域不同，选出最擅长的前k个权重）数 |
| headSize | int32_t  | 总专家数，例如从64个专家中选8个专家，则headSize为64，headNum为8，取值范围为\[0， 127] |
| cumSumInt64 | bool  | 输出的cumSum类型是否为int64，默认为false，为false时输出类型为int32 |
| deviceExpert | vector<int32_t>  | 当该参数为空时为TP模式，所有专家有效，当不为空时则是EP模式，仅仅该参数中的专家有效，各个输出会将有效的专家的数据前置 |
 
- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  |format|
| ------------ | ------------ | ------------ | ------------ |------------ |
|  topk|In  |  [b\*s\*headNum] |int32|ND|
| idxArr|In  |  [b\*s\*headNum]|int32|ND|
|  tokenIndex | Out | [b\*s\*headNum] |int32|ND|
|  CumSum | Out | [headSize] |int32|ND|
|  originalIndex | Out | [b\*s\*headNum] |int32|ND|
|  validIndex | Out | [1或0] |int32|ND|

1. topk要求每headNum个数内不能重复(比如headNum=2， 则topk要求第0个、第1个数不重复，第2个、第3个数不重复)
2. idxArr要求值等于自身下标(从0开始的等差数列，如 [0, 1, 2, 3])
3. 当开启EP功能时validIndex的shape为[1]，不开启为空张量
 
## 功能描述
- 算子功能：topk是每个token选择topkExpertNum个专家的矩阵拉平后得到的为一维向量（长度n=token数*topkExpertNum），idxArr是从0开始的0、1、2...n-1，也是一个一维向量，这两个输入的长度都为n。将topk从小到大排序，同时idxArr联动（比如，排序后得到idxArrSorted），此时idxArrSorted即为输出originalIndex。然后将originalIndex中每个元素都除以topkExpertNum，即得到输出tokenIndex。将输入topk进行bincount计算（即统计每个专家出现的次数），然后将每个元素累加求和，得到输出CumSum。当开启EP功能时，会将参数中指定的专家的相关输出放到最前，用validIndex指示最后有效位，之后的数据不做保证。
- 计算公式：$originalIndex=argsort(topk)$
    $tokenIndex=originalIndex/topkExpertNum$
    $CumSum=accumulate(bincount(topk))$
 
## 示例
```
输入：
		param = {"topkExpertNum":2, "cumSumNum":4}
		
		topk = [0, 1, 0, 2, 1, 2, 0, 3]
    
        idxArr = [0, 1, 2, 3, 4, 5, 6, 7]
输出：
        tokenIndex = [0, 1, 3, 0, 2, 1, 2, 3]   
    
        CumSum = [3, 5, 7, 8]   
    
        originalIndex = [0, 2, 6, 1, 4, 3, 5, 7]   
 
```
 
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|无|
|  ascend310p| 不支持Ep功能，Ep参数无效 |