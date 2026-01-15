/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_ACL_H
#define ATB_ACL_H

#include <acl/acl_rt.h>
#include <acl/acl.h>
#include <opdev/common_types.h>
#include "atb/operation.h"
#include "atb/infer_op_params.h"

//!
//! \file atb_acl.h
//!
//! \brief 定义atb封装的acl使用接口
//!

#ifdef __cplusplus
extern "C" {
#endif

//!
//! \brief 关于FusedAddTopkDiv算子使用aclnn风格调用的2段式接口的第1段，
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param x FusedAddTopkDiv算子的输入tensor
//! \param addNum FusedAddTopkDiv算子的输入tensor
//! \param mappingNum FusedAddTopkDiv算子的输入tensor（enableExpertMapping为false时，需要置为nullptr）
//! \param mappingTable FusedAddTopkDiv算子的输入tensor（enableExpertMapping为false时，需要置为nullptr）

//! \param groupNum FusedAddTopkDiv算子分组数量
//! \param groupTopk FusedAddTopkDiv算子选择k个组
//! \param n FusedAddTopkDiv算子分组数量
//! \param k FusedAddTopkDiv算子topk选择前k个值
//! \param activationType FusedAddTopkDiv算子激活类型
//! \param isNorm FusedAddTopkDiv算子是否归一化
//! \param scale FusedAddTopkDiv算子归一化后的乘系数
//! \param enableExpertMapping FusedAddTopkDiv算子中是否开启物理专家向逻辑专家的映射

//! \param y FusedAddTopkDiv算子输出tensor
//! \param indices FusedAddTopkDiv算子输出tensor
//! \param workspaceSize FusedAddTopkDiv算子的workspace大小
//! \param op FusedAddTopkDiv算子的handler
//! \param context FusedAddTopkDiv算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbFusedAddTopkDivGetWorkspaceSize(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                                               const aclTensor *mappingTable, uint32_t groupNum, uint32_t groupTopk,
                                               uint32_t n, uint32_t k, int activationType, bool isNorm, float scale,
                                               bool enableExpertMapping, aclTensor *y, aclTensor *indices,
                                               uint64_t *workspaceSize, atb::Operation **op, atb::Context *context);

//!
//! \brief 关于FusedAddTopkDiv算子使用aclnn风格调用的2段式接口的第2段，
//! 用于算子的推理调度阶段
//!
//! \param workspace 针对FusedAddTopkDiv算子申请的工作空间
//! \param workspaceSize FusedAddTopkDiv算子的workspace大小
//! \param op FusedAddTopkDiv算子的op handler
//! \param context FusedAddTopkDiv算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbFusedAddTopkDiv(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context);

//!
//! \brief 关于MLA算子使用aclnn风格调用的2段式接口的第1段，
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param qNope MLA算子的输入tensor
//! \param qRope MLA算子的输入tensor
//! \param ctKV MLA算子的输入tensor
//! \param kRope MLA算子的输入tensor
//! \param blockTables MLA算子的输入tensor
//! \param contextLens MLA算子的输入tensor
//! \param mask MLA算子的输入tensor
//! \param qSeqLen MLA算子的输入tensor
//! \param qkDescale MLA算子的输入tensor
//! \param pvDescale MLA算子的输入tensor
//! \param headNum MLA算子的query head数量
//! \param qkScale MLA算子Q*K^T的缩放系数
//! \param kvHeadNum MLA算子的kv head 数量
//! \param maskType MLA算子的mask类型
//! \param calcType MLA算子的calc类型
//! \param cacheMode MLA算子的cache类型
//! \param attenOut MLA算子的输出tensor
//! \param lse MLA算子的输出tensor
//! \param workspaceSize MLA算子的workspace大小
//! \param op MLA算子的handler
//! \param context MLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbMLAGetWorkspaceSize(const aclTensor *qNope, const aclTensor *qRope, const aclTensor *ctKV,
                                   const aclTensor *kRope, const aclTensor *blockTables, const aclTensor *contextLens,
                                   const aclTensor *mask, const aclTensor *qSeqLen, const aclTensor *qkDescale,
                                   const aclTensor *pvDescale, int32_t headNum, float qkScale, int32_t kvHeadNum,
                                   int maskType, int calcType, uint8_t cacheMode, aclTensor *attenOut, aclTensor *lse,
                                   uint64_t *workspaceSize, atb::Operation **op, atb::Context *context);

//!
//! \brief 关于MLA算子使用aclnn风格调用的2段式接口的第2段，
//! 用于算子的推理调度阶段
//!
//! \param workspace 针对MLA算子申请的工作空间
//! \param workspaceSize MLA算子的workspace大小
//! \param op MLA算子的op handler
//! \param context MLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbMLA(void* workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context);

//!
//! \brief MLA prefill 前处理接口
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param q MLA算子的输入tensor: query
//! \param qRope MLA算子的输入tensor: query rope
//! \param k MLA算子的输入tensor: key
//! \param kRope MLA算子的输入tensor: key rope
//! \param v MLA算子的输入tensor: value
//! \param qSeqLen MLA算子的输入tensor: query seq length
//! \param kvSeqLen MLA算子的输入tensor: key and value seq length
//! \param mask MLA算子的输入tensor: mask
//! \param headNum MLA算子的query head数量
//! \param qkScale MLA算子Q*K^T后的缩放系数
//! \param kvHeadNum MLA算子的kv head 数量
//! \param maskType MLA算子的mask类型
//! \param cacheMode MLA算子的cache类型
//! \param attenOut MLA算子的输出tensor
//! \param workspaceSize MLA算子的workspace大小
//! \param op MLA算子的handler
//! \param context MLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbMLAPreFillGetWorkspaceSize(const aclTensor *q, const aclTensor *qRope, const aclTensor *k,
    const aclTensor *kRope, const aclTensor *v, const aclTensor *qSeqLen, const aclTensor *kvSeqLen,
    const aclTensor *mask, int32_t headNum, float qkScale, int32_t kvHeadNum,
    int maskType, uint8_t cacheMode, aclTensor *attenOut,
    uint64_t *workspaceSize, atb::Operation **op, atb::Context *context);

//!
//! \brief MLA prefill 处理接口
//!
//! \param workspace 针对MLA算子申请的工作空间
//! \param workspaceSize MLA算子的work space大小
//! \param op MLA算子的op handler
//! \param context MLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbMLAPreFill(void* workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context);

//!
//! \brief 关于MlaPreprocess算子使用aclnn风格调用的2段式接口的第1段，
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param input MLA算子的输入tensor
//! \param gamma0 MLA算子的输入tensor
//! \param beta0 MLA算子的输入tensor
//! \param quantScale0 MLA算子的输入tensor
//! \param quantOffset0 MLA算子的输入tensor
//! \param wdqkv MLA算子的输入tensor
//! \param deScale0 MLA算子的输入tensor
//! \param bias0 MLA算子的输入tensor
//! \param gamma1 MLA算子的输入tensor
//! \param beta1 MLA算子的输入tensor
//! \param quantScale1 MLA算子的输入tensor
//! \param quantOffset1 MLA算子的输入tensor
//! \param wuq MLA算子的输入tensor
//! \param deScale1 MLA算子的输入tensor
//! \param bias1 MLA算子的输入tensor
//! \param gamma2 MLA算子的输入tensor
//! \param cos MLA算子的输入tensor
//! \param sin MLA算子的输入tensor
//! \param wuk MLA算子的输入tensor
//! \param kvCache MLA算子的输入tensor
//! \param kvCacheRope MLA算子的输入tensor
//! \param slotmapping MLA算子的输入tensor
//! \param ctkvScale MLA算子的输入tensor
//! \param qNopeScale MLA算子的输入tensor

//! \param wdqDim MLAPreprocess算子的经过matmul后拆分的dim大小
//! \param qRopeDim MLAPreprocess算子q传入rope的dim大小
//! \param kRopeDim MLAPreprocess算子k传入rope的dim大小
//! \param epsilon MLAPreprocess算子防止除0的参数
//! \param qRotaryCoeff MLAPreprocess算子的q旋转系数
//! \param kRotaryCoeff MLAPreprocess算子的k旋转系数
//! \param transposeWdq MLAPreprocess算子wdq是否转置
//! \param transposeWuq MLAPreprocess算子wuq是否转置
//! \param transposeWuk MLAPreprocess算子wdk是否转置
//! \param cacheMode MLAPreprocess算子的cache类别
//! \param quantMode MLAPreprocess算子的quant类别

//! \param qOut0 MLAPreprocess算子的输出tensor
//! \param kvCacheOut0 MLAPreprocess算子的输出tensor
//! \param qOut1 MLAPreprocess算子的输出tensor
//! \param kvCacheOut1 MLAPreprocess算子的输出tensor
//! \param workspaceSize MLA算子的workspace大小
//! \param op MLA算子的handler
//! \param context MLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbMLAPreprocessGetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *deScale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *deScale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotmapping, const aclTensor *ctkvScale, const aclTensor *qNopeScale,
    uint32_t wdqDim, uint32_t qRopeDim, uint32_t kRopeDim, float epsilon, uint32_t qRotaryCoeff, uint32_t kRotaryCoeff,
    bool transposeWdq, bool transposeWuq, bool transposeWuk, uint8_t cacheMode, uint16_t quantMode, aclTensor *qOut0,
    aclTensor *kvCacheOut0, aclTensor *qOut1, aclTensor *kvCacheOut1, uint64_t *workspaceSize, atb::Operation **op,
    atb::Context *context);

//!
//! \brief 关于MLAPreprocess算子使用aclnn风格调用的2段式接口的第2段，
//! 用于算子的推理调度阶段
//!
//! \param workspace 针对MLAPreprocess算子申请的工作空间
//! \param workspaceSize MLAPreprocess算子的workspace大小
//! \param op MLAPreprocess算子的op handler
//! \param context MLAPreprocess算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbMLAPreprocess(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context);

//!
//! \brief 关于PagedCacheLoad算子使用aclnn风格调用的2段式接口的第1段，
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param keyCache PagedCacheLoad算子的输入tensor
//! \param valueCache PagedCacheLoad算子的输入tensor
//! \param blockTables PagedCacheLoad算子的输入tensor
//! \param contextLens PagedCacheLoad算子的输入tensor
//! \param key PagedCacheLoad算子的输入/输出tensor
//! \param value PagedCacheLoad算子的输入/输出tensor
//! \param seqStarts PagedCacheLoad算子的输入tensor
//! \param kvCacheCfg keyCache和valueCache为ND还是NZ格式
//! \param isSeqLensCumsumType 是否使用batch输入为累加模式
//! \param hasSeqStarts 是否提供batch在blocktable中对应起始位置，对齐到blocktable
//! \param workspaceSize PagedCacheLoad算子的workspace大小
//! \param op PagedCacheLoad算子的handler
//! \param context PagedCacheLoad算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbPagedCacheLoadGetWorkspaceSize(const aclTensor *keyCache, const aclTensor *valueCache,
                                              const aclTensor *blockTables, const aclTensor *contextLens,
                                              const aclTensor *key, const aclTensor *value, const aclTensor *seqStarts,
                                              int8_t kvCacheCfg, bool isSeqLensCumsumType, bool hasSeqStarts,
                                              uint64_t *workspaceSize, atb::Operation **op, atb::Context *context);

//!
//! \brief 关于PagedCacheLoad算子使用aclnn风格调用的2段式接口的第2段，
//! 用于算子的推理调度阶段
//!
//! \param workspace 针对PagedCacheLoad算子申请的工作空间
//! \param workspaceSize PagedCacheLoad算子的workspace大小
//! \param op PagedCacheLoad算子的op handler
//! \param context PagedCacheLoad算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbPagedCacheLoad(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context);

//!
//! \brief 关于RingMLA算子使用aclnn风格调用的2段式接口的第1段，
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param querySplit1 RingMLA算子的输入tensor
//! \param querySplit2 RingMLA算子的输入tensor
//! \param keySplit1 RingMLA算子的输入tensor
//! \param keySplit2 RingMLA算子的输入tensor
//! \param value RingMLA算子的输入tensor
//! \param mask RingMLA算子的输入tensor
//! \param seqLen RingMLA算子的输入tensor
//! \param prevOut RingMLA算子的输入tensor（calcType为1时，需要置为nullptr）
//! \param prevLse RingMLA算子的输入tensor（calcType为1时，需要置为nullptr）

//! \param headNum RingMLA算子头大小
//! \param kvHeadNum RingMLA算子kv头大小
//! \param qkScale RingMLA算子tor值
//! \param kernelType RingMLA算子内核精度类型
//! \param maskType RingMLA mask类型
//! \param inputLayout RingMLA算子数据排布格式
//! \param calcType RingMLA算子计算类型

//! \param output RingMLA算子输出tensor
//! \param softmaxLse RingMLA算子输出tensor
//! \param workspaceSize RingMLA算子的workspace大小
//! \param op RingMLA算子的handler
//! \param context RingMLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbRingMLAGetWorkspaceSize(const aclTensor *querySplit1, const aclTensor *querySplit2,
                                       const aclTensor *keySplit1, const aclTensor *keySplit2, const aclTensor *value,
                                       const aclTensor *mask, const aclTensor *seqLen, const aclTensor *prevOut,
                                       const aclTensor *prevLse, int32_t headNum, int32_t kvHeadNum, float qkScale,
                                       int kernelType, int maskType, int inputLayout, int calcType, aclTensor *output,
                                       aclTensor *softmaxLse, uint64_t *workspaceSize, atb::Operation **op,
                                       atb::Context *context);

//!
//! \brief 关于RingMLA算子使用aclnn风格调用的2段式接口的第2段，
//! 用于算子的推理调度阶段
//!
//! \param workspace 针对RingMLA算子申请的工作空间
//! \param workspaceSize RingMLA算子的workspace大小
//! \param op RingMLA算子的op handler
//! \param context RingMLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbRingMLA(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context);

//!
//! \brief 关于SelfAttentionPrefixEncoder算子使用aclnn风格调用的2段式接口的第1段，
//! 用于workspaceSize的获取，以及输入输出tensors的准备等前处理
//!
//! \param query SelfAttentionPrefixEncoder算子的输入tensor
//! \param key SelfAttentionPrefixEncoder算子的输入tensor
//! \param value SelfAttentionPrefixEncoder算子的输入tensor
//! \param blockTables SelfAttentionPrefixEncoder算子的输入tensor
//! \param mask SelfAttentionPrefixEncoder算子的输入tensor（maskType为MASK_TYPE_CASUAL_MASK时，需要置为nullptr）
//! \param seqLen SelfAttentionPrefixEncoder算子的输入tensor
//! \param kvSeqLen SelfAttentionPrefixEncoder算子的输入tensor
//! \param slopes SelfAttentionPrefixEncoder算子的输入tensor（maskType不为MASK_TYPE_ALIBI_COMPRESS或MASK_TYPE_ALIBI_COMPRESS_SQRT时，需要置为nullptr）

//! \param maskType SelfAttentionPrefixEncoder mask类型
//! \param headNum SelfAttentionPrefixEncoder算子头大小
//! \param kvHeadNum SelfAttentionPrefixEncoder算子kv头大小
//! \param qkScale SelfAttentionPrefixEncoder算子tor值

//! \param attnOut SelfAttentionPrefixEncoder算子输出tensor
//! \param workspaceSize RingMLA算子的workspace大小
//! \param op RingMLA算子的handler
//! \param context RingMLA算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbSelfAttentionPrefixEncoderGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
                                                          const aclTensor *value, const aclTensor *blockTables,
                                                          const aclTensor *mask, const aclTensor *seqLen,
                                                          const aclTensor *kvSeqLen, const aclTensor *slopes,
                                                          int maskType, int32_t headNum, int32_t kvHeadNum,
                                                          float qkScale, aclTensor *attnOut, uint64_t *workspaceSize,
                                                          atb::Operation **op, atb::Context *context);

//!
//! \brief 关于SelfAttentionPrefixEncoder算子使用aclnn风格调用的2段式接口的第2段，
//! 用于算子的推理调度阶段
//!
//! \param workspace 针对SelfAttentionPrefixEncoder算子申请的工作空间
//! \param workspaceSize SelfAttentionPrefixEncoder算子的workspace大小
//! \param op SelfAttentionPrefixEncoder算子的op handler
//! \param context SelfAttentionPrefixEncoder算子的上下文参数
//!
//! \return 表示函数是否执行成功的状态码
atb::Status AtbSelfAttentionPrefixEncoder(void *workspace, uint64_t workspaceSize, atb::Operation *op,
                                          atb::Context *context);

#ifdef __cplusplus
}
#endif
#endif
