# Cumsum

## 参数说明

- **OpName**：CumsumOperation
- **PARAM**
  **Param Type**：OpParam::Cumsum

| 名称      | 类型             | 描述                                                                 |
| --------- | ---------------- | -------------------------------------------------------------------- |
| axis      | SVector<int64_t> | cumsum算子在指定axis轴上进行累加和，传入axis数组只能包含一个轴索引   |
| exclusive | bool             | 决定在某一个轴上的累加结果从第一个元素开始，默认为0,包含第一个元素。 |
| reverse   | bool             | 决定正向累加或逆向累加，默认为0，正向累加。                          |

- **In/Out Tensor**

| 名称 | 类型 | dims                             | dtype             | format |
| ---- | ---- | -------------------------------- | ----------------- | ------ |
| x    | In   | [$d_0$, $d_1$, ..., $d_n$] | bfloat16、float16 | ND     |
| y    | Out  | [$d_0$, $d_1$, ..., $d_n$] | 与x一致           | ND     |

## 功能描述

- 算子功能：后处理计算功能。输入张量在指定axis轴上进行累积和计算。
  例如：某一个轴的数据为{a, b, c}, 若exclusive为false, 则结果为{a, a+b, a+b+c}, 若exclusive为true, 则结果为{0, a, a+b}。
  reverse为0, 则为正向累加过程, 反之则为逆向累加。
  例如，某一轴为{a, b, c}, reverse为false, 结果为{a, a+b, a+b+c}, reverse为true, 结果为{a+b+c, a+b, a}。

## 示例

```
输入：
    Axis = [0]
    reverse = false
    exclusive = false
  
	x = [[2,3],
         [4,5]]
 

输出：
    y = [[2,3],
         [6,8]] 
输入：
    Axis = [1]
    reverse = false
    exclusive = false
  
	x = [[2,3],
         [4,5]]
 
输出：
    y = [[2,5],
         [4,9]]   
 
```

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
