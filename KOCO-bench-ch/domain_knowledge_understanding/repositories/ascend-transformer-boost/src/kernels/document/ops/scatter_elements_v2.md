# ScatterElementsV2

## 参数说明

- **OpName**：ScatterElementsV2Operation
- **PARAM**
**Param Type**：OpParam::ScatterElementsV2

| 名称       | 类型    | 描述                                                                |
|-----------|--------|---------------------------------------------------------------------| 
| axis      | int32  | axis只支持 - 1                                      |
| reduction | string | 支持none或add，当reduction为none时为原地更新；当reduction为add是为原地累加  |
 

- **In/Out Tensor**

| 名称            | 类型     | dims                       | dtype                                     | format |
|---------------|--------| -------------------------- |-------------------------------------------| ------ |
| input_tensor  | In     | [$d_0$, $d_1$, ..., $d_n$] | float16/float32/uint8/int8/int32/bfloat16 | ND     |
| indices       | In     | [$d_0$, $d_1$, ..., $d_n$] | int64/int32                               | ND     |
| update        | In     | [$d_0$, $d_1$, ..., $d_n$] | 同`input_tensor`                           | ND     |
| output_tensor | output | [$d_0$, $d_1$, ..., $d_n$] | 同`input_tensor`                           | ND     |

## 功能描述

- 算子功能：在索引张量指定的索引处，将张量update中的所有值写入input_tensor中。
- 算子公式：
```

2维tensor，公式如下:

input_tensor[i][indices[i][j]] = update[i][j]  # if axis == -1 reduction == none
input_tensor[i][indices[i][j]] = input_tensor[indices[i][j]][j] + update[i][j]  # if axis == 1 reduction == add

3维tensor，公式如下:

input_tensor[i][j][indices[i][j][k]] = update[i][j][k]  # if axis == -1 reduction == none
input_tensor[i][j][indices[i][j][k]] = input_tensor[indices[i][j][k]][j][k] + update[i][j][k]  # if axis == 2 reduction == add
```


## 示例

```
输入：
    input_tensor = [[0,    0,    0,    0,    0],
                    [0,    0,    0,    0,    0],
                    [0,    0,    0,    0,    0],
                    [0,    0,    0,    0,    0],
                    [0,    0,    0,    0,    0]]

    indices = [[0,    1],
               [0,    1]]
               
    update =  [[1,    2],
               [3,    4]]

根据公式可得出过程推导：
input_tensor[i][indices[i][j]] = update[i][j]
input_tensor[0][indices[0][0]] = update[0][0] -> input_tensor[0][0] = update[0][0] -> input_tensor[0][0] = 1
input_tensor[0][indices[0][1]] = update[0][1] -> input_tensor[0][1] = update[0][1] -> input_tensor[0][1] = 2
input_tensor[1][indices[1][0]] = update[1][0] -> input_tensor[1][0] = update[1][0] -> input_tensor[1][0] = 3
input_tensor[1][indices[1][1]] = update[1][1] -> input_tensor[1][1] = update[1][1] -> input_tensor[1][1] = 4


输出:
        [[1,    2,    0,    0,    0],
         [3,    4,    0,    0,    0],
         [0,    0,    0,    0,    0],
         [0,    0,    0,    0,    0],
         [0,    0,    0,    0,    0]]
```

【使用限制】

1：axis 只能等于 -1 （与torch存在差异） 

2：input_tensor/indices_tensor/update_tensor 最高只支持dim = 8（框架侧限制，算子代码中无需限制）

3：input_tensor 和 update_tensor 的dtype必须相同 

4：indices_tensor和update_tensor的dim需要完全相同 

5：indices_tensor和update_tensor的shape需要完全相同 

5：indices_tensor的维度需要和input_tensor的维度相同 

6：indices_tensor中的每一个元素均需要小于 input_tensor[-1].dim() （tiling侧保证，算子代码中无法校验）

7:input_tensor 和 indices_tensor 在非尾轴和非0轴上的shape必须一致 

8:indices_tensor 0轴和尾轴不大于inpute_tensor的0轴和尾轴 

9:indices_tensor\inpute_tensor\update_tensor每个维度都不能为0 

## 支持芯片型号

| 芯片名称   | 约束               |
| ---------- | ------------------ |
| ascend910b | 无                 |
