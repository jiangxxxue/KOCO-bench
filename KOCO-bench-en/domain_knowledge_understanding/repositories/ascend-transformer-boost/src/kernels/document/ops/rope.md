# Rope
## 参数说明
- **OpName**：RopeOperation
- **PARAM**
**Param Type**：OpParam::Rope

- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  q|In  |[ntokens, hidden_size_q]|float16、bfloat16、float|ND|输入q向量|
| k  |  In| [ntokens, hidden_size_k]| 与q一致|ND|输入k向量|
|  cos | In|[ntokens, head_dim]| 与q一致|ND||
|  sin | In|[ntokens, head_dim]| 与q一致|ND||
|  seqlen | In|[batch]| 与q一致|ND|存了每次token的数量|
|  repe_q | Out| [ntokens, hidden_size_q]| 与q一致|ND|输出q向量|
|  rope_k | Out| [ntokens, hidden_size_q]| 与q一致|ND|输出k向量|
 
## 功能描述
- 算子功能：旋转位置编码。
ntokens是原始向量的[batch, seqlen]维度进行unpad合并之后的产生的维度，每个batch的实际seqlen通过“seqlen”向量传入。
注：1.hidden_size_q一般情况下和hidden_size_k 是一样的， 为 head_dim * head_num。
也会出现 hidden_size_q = head_dim * head_num，hidden_size_k = head_dim的情况。此处head_num必须是16的倍数。
 
2. rotaryCoeff == 2 的情况下
输入cos向量,在单token下，0~head_dim/2 和 head_dim/2+1~head_dim的值相同
输入sin向量,在单token下，0~head_dim/2 和 head_dim/2+1~head_dim的值相同
rotaryCoeff == 4 的情况下
输入cos向量,在单token下，0~head_dim/4 和 head_dim/4+1~head_dim/2的值相同
输入cos向量,在单token下，head_dim/2~ 2 * head_dim 3 和 2*head_dim/3+1~head_dim的值相同
输入sin向量,在单token下，0~head_dim/4 和 head_dim/4+1~head_dim/2的值相同
输入sin向量,在单token下，head_dim/2~ 2 * head_dim 3 和 2*head_dim/3+1~head_dim的值相同
rotaryCoeff == head_dim的情况下
每两个cos,sin的值相等。
此处的sin和cos的值跟随rotaryCoeff变化。
**重要须知**
如果rotaryCoeff==head_dim 要使用workspace去存错位差的数据。workspace分配的大小为：
ntokens * hidden_size_q * sizeof(half)
 
 
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b| 无|
|  ascend310p| 无|