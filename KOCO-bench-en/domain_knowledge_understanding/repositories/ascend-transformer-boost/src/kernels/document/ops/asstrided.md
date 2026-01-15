# Asstrided
## 参数说明
- **OpName**：AsStridedOperation
- **PARAM**
**Param Type**：OpParam::AsStrided
 
| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| size| SVector<int64_t>  |OutTensor的shape|
|stride|SVector<int64_t>|用于从InTensor推导OutTensor的各维度的步长|
| offset| SVector<int64_t>  |OutTensor的内存相对于InTensor内存的偏移，作为常数使用|
 
- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  |format|
| ------------ | ------------ | ------------ | ------------ |------------ |
|  x|In  |  [$d_0$, $d_1$, ..., $d_n$]|float16、int64|ND|
| y|Out  |  和size大小一致|与x一致|ND|
 
## 功能描述
- 算子功能：size、stride、offset与InTensor，生成一个数据重新排布过的OutTensor。
- 计算公式：
y.Shape=param.size;
y.data.at(0)=x.data.at(param.offset.at(0))
 
## 示例
```
其他部分数据推导以一个3x3的shape为例：
    输入:
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
 
    当size=3,3;stride=0,1;offset=0时
    输出:
        y = [[1, 2, 3],
             [1, 2, 3],
             [1, 2, 3]]
 
    当size=3,3;stride=1,0;offset=0时
    输出:
        y = [[1, 1, 1],
             [2, 2, 2],
             [3, 3, 3]]
 
    当size=2,2;stride=3,2;offset=0时
    输出:
        y = [[1, 3],
             [4, 6]]
 
    当size=2,2;stride=3,2;offset=1时
    输出:
        y = [[2, 4],
             [5, 7]]
 
```
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b|无 |
|  ascend310p| 无|