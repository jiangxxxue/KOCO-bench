# KVCacheNz
## 参数说明
- **OpName**：KVCacheOperation
- **PARAM**
**Param Type**：OpParam::KVCache
 
| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| KVCache | enum  | KVCACHE_NZ |
 
- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  new_kv|In  |[1,hiddenSize/16,ntokens,16]|float16|NZ||
| layer_id  |  In| [1]| int32|ND|当前层的ID|
|  cache_in | In|[layer,batch,hiddensize/16, maxseqLen, 16]| float16|NZ|缓存的K或V|
|  token_offset | In| [batch]| int32|ND|可选|
|  seqlen | In|[batch]| int32|ND|可选|
|  cache_out | Out| [layer, batch, max_seqlen, hidden_size]| float16|NZ|新的K或V|
 
## 功能描述
- 算子功能：当前轮输出seqlen长度的token结果与上一轮的cache存储结果进行拼接, 数据格式为Nz。
ntokens是原始向量的[batch, seqlen]维度进行unpad合并之后的产生的维度, 每个batch的实际seqlen通过“seqlen”向量传入。
max_seqlen是预申请的一个较大的长度, 实际的每个batch的seqlen通过seqlen向量传入, tokenoffset表示放入cache后的实际长度。
输入向量cache_in与输出向量cache_out的data指向同一块地址。
 
 
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend310p|无 |