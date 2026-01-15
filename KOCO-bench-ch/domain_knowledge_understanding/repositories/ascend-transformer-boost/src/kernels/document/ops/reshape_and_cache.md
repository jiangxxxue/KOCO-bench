# ReshapeAndCache

## 参数说明

- **OpName**：ReshapeAndCacheOperation
- **PARAM**
  **Param Type**：OpParam::ReshapeAndCache

| 名称            | 类型 | 描述                                                                  |
| --------------- | ---- | --------------------------------------------------------------------- |
| ReshapeAndCache | enum | RESHAPE_AND_CACHE_ND / RESHAPE_AND_CACHE_NZ  / RESHAPE_AND_CACHE_WINS |

- **In/Out Tensor**

| 名称         | 类型 | dims                                        | dims1                                          | dtype                   | format | 描述       |
| ------------ | ---- | ------------------------------------------- | ---------------------------------------------- | ----------------------- | ------ | ---------- |
| k            | In   | [batch*seqlen, kv_heads, head_size]         | [batch*seqlen, kv_heads, head_size]            | float16、int8、bfloat16 | ND     |            |
| v            | In   | [batch*seqlen, kv_heads, head_size]         | [batch*seqlen, kv_heads, head_size]            | 与k一致                 | ND     | 当前层的ID |
| key_cache    | In   | [num_blocks, block_size, head_size]         | [num_blocks*kv_head, block_size, 1,head_size]  | 与k一致                 | ND     | 缓存的K或V |
| value_cache  | In   | [num_blocks, block_size, head_size]         | [num_blocks*kv_head, block_size, 1,head_size]  | 与k一致                 | ND     |            |
| slot_mapping | In   | [batch_seqlen]                              | [batch*kv_head]                                | int32                   | ND     |            |
| wins         | In   |                                             | [batch*kv_head]                                | int32                   | ND     | 可选       |
| seq_len      | In   |                                             | [batch]                                        | int32                   | ND     | 可选       |
| key_output   | Out  | [num_blocks,block_size,kv_heads, head_size] | [num_blocks*kv_head, block_size, 1, head_size] | 与k一致                 | ND     | 新的K或V   |
| value_output | Out  | [num_blocks,block_size,kv_heads,head_size]  | [num_blocks*kv_head, block_size, 1, head_size] | 与k一致                 | ND     | 新的K或V   |

## 功能描述

- 算子功能：ReShape and Cache 算子为PA提供其所需要的KV内存排布。
  ntokens是原始向量的[batch, seqlen]维度进行unpad合并之后的产生的维度
  注：用户需确保 key_output & key_cache；value_output & value_cache 分别使用同一地址。同时slot_mapping存在值域约束：值域在(-num_blocks * block_size, num_blocks * block_size)范围内且不存在重复数值

## 支持芯片型号

| 芯片名称   | 约束                   |
| ---------- | ---------------------- |
| ascend910b | 无                     |
| ascend310p | 不支持bfloat16数据类型 |
