# KVCacheNdDynamicBatch
## 参数说明
- **OpName**：KVCacheOperation
- **PARAM**
**Param Type**：OpParam::KVCache
 
| 名称  | 类型  | 描述 |
| ------------ | ------------ | ------------ |
| KVCache | enum  | KVCACHE_DYNAMIC_BATCH |
|batch_run_status|SVector<int32_t>|for kvcacheDynamicBatch, 指示当前batch是否生效cache拼接。取值范围[0, 1] |
- **In/Out Tensor**
 
|名称 | 类型  | dims  | dtype  |format|描述|
| ------------ | ------------ | ------------ | ------------ |------------ |------------ |
|  new_kv|In  |[ntokens, hidden_size]|float16|ND||
| layer_id  |  In| [1]| int32|ND|当前层的ID|
|  cache_in | In|[batch, max_seqlen, hidden_size]| float16|ND|缓存的K或V|
|  token_offset | In| [batch]| int32|ND|可选|
|  seqlen | In|[batch]| int32|ND|可选|
|  cache_out | Out| [batch, max_seqlen, hidden_size]| float16|ND|新的K或V|
|  batch_run_status | OpParam| [batch]| SVector<int32_t>|ND|指示当前batch是否生效cache拼接。取值范围[0, 1]|
 
## 功能描述
- 算子功能：当前轮输出seqlen长度的token结果与上一轮的cache存储结果进行拼接, 支持动态batch, 即通过batch_run_status参数来控制当前kv是否进行cache拼接。
ntokens是原始向量的[batch, seqlen]维度进行unpad合并之后的产生的维度, 每个batch的实际seqlen通过“seqlen”向量传入。
max_seqlen是预申请的一个较大的长度, 实际的每个batch的seqlen通过seqlen向量传入, tokenoffset表示放入cache后的实际长度。
输入向量cache_in与输出向量cache_out的data指向同一块地址。
 
## 支持芯片型号
 
|芯片名称|约束 | 
| ------------ | ------------ | 
|  ascend910b| 无|