/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_INFEROPPARAM_H
#define ATB_INFEROPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

//!
//! \file infer_op_params.h
//!
//! \brief 定义加速库所有推理算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace infer {

//!
//! \enum InputLayout
//!
//! \brief 数据排布类型
//!
enum InputLayout : int {
    TYPE_BSND = 0, //!< 默认值，表示数据排布为BSND
    TYPE_BNSD      //!< 表示数据排布为BNSD
};

//!
//! \enum QuantType
//!
//! \brief 量化支持的类型
//!
enum QuantType : int {
    QUANT_UNDEFINED = 0, //!< 不量化
    QUANT_UNQUANT = 0,   //!< 不量化
    QUANT_INT4 = 1,      //!< 当前不支持
    QUANT_INT8 = 2,      //!< int8量化
    QUANT_INT16 = 3,     //!< 当前不支持
    QUANT_FLOAT8 = 4,    //!< 当前不支持
    QUANT_FLOAT16 = 5,   //!< 当前不支持
};

//!
//! \enum DynamicQuantType
//!
//! \brief 动态量化支持的类型
//!
enum DynamicQuantType : int {
    DYNAMIC_QUANT_UNDEFINED = 0, //!< 非动态量化
    DYNAMIC_QUANT_SYMMETRIC,     //!< 对称动态量化
    DYNAMIC_QUANT_ASYMMETRIC,    //!< 非对称动态量化，暂不支持
};

//!
//! \enum ActivationType
//!
//! \brief 激活支持的类型
//! ACTIVATION_FAST_GELU：快速运算的Gelu激活函数，对Tensor内每个element做Gelu激活函数近似计算，计算速度更快，同时保持较高的准确性。
//! ACTIVATION_SWIGLU_FORWARD: Swiglu正向激活函数。Atlas 推理系列产品中只支持32位对齐的数据。
//! ACTIVATION_FASTER_GELU_FORWARD: 简化后的FastGelu激活函数，计算速度更快。
//! ACTIVATION_SWIGLU_BACKWARD: Swiglu正向激活函数的反向，求梯度时使用。只支持Atlas 800I A2推理产品。
//!
enum ActivationType : int {
    ACTIVATION_UNDEFINED = 0,       //!< 未定义
    ACTIVATION_RELU,                //!< RELU激活类型
    ACTIVATION_GELU,                //!< GELU激活类型
    ACTIVATION_FAST_GELU,           //!< FAST_GELU激活类型
    ACTIVATION_SWISH,               //!< SWISH激活类型
    ACTIVATION_LOG,                 //!< LOG激活类型
    ACTIVATION_SWIGLU_FORWARD,      //!< SWIGLU_FORWARD激活类型
    ACTIVATION_SWIGLU_BACKWARD,     //!< SWIGLU_BACKWARD激活类型
    ACTIVATION_SIGMOID,             //!< SIGMOID激活类型
    ACTIVATION_FASTER_GELU_FORWARD, //!< FASTER_GELU_FORWARD激活类型
    ACTIVATION_MAX,                 //!< 枚举最大值, 非激活类型
};

//!
//! \enum CommMode
//!
//! \brief 通信算子支持的通信模式.
//!
enum CommMode : int {
    COMM_UNDEFINED = -1, //!< 未定义
    COMM_MULTI_PROCESS,  //!< 指定多进程通信
    COMM_MULTI_THREAD,   //!< 指定多线程通信
};

//!
//! \brief 激活函数。
//!
struct ActivationParam {
    //! \enum GeLUMode
    //! \brief GeLU激活函数可选的计算模式
    enum GeLUMode : int {
        TANH_MODE = 0, //!< 默认值，使用tanh估算
        NONE_MODE,     //!< 原GeLU计算公式
    };
    //! 激活函数类型，ActivationType类型枚举值.
    ActivationType activationType = ACTIVATION_UNDEFINED;
    //! SWISH激活函数的参数.
    float scale = 1.0f;
    //! SWIGLU激活函数的参数.
    int32_t dim = -1;
    //! GeLU模式选择参数
    GeLUMode geluMode = TANH_MODE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief InTensor根据指定参数，生成一个数据重新排布过的OutTensor.
//!
//! \warning 输出y基于输入x的总偏移量要求小于输入x的大小.
//!
struct AsStridedParam {
    //!
    //! \brief OutTensor的shape.
    //!
    //! \warning size的长度要求小于或等于8且各元素要求大于0.
    //!
    SVector<int64_t> size;
    //!
    //! \brief 用于从InTensor推导OutTensor的各维度的步长.
    //!
    //! \warning stride的长度要求与size一致，各元素要求大于或等于0.
    //!
    SVector<int64_t> stride;
    //!
    //! \brief OutTensor内存相对于InTensor内存的偏移，作为常数使用.
    //!
    //! \warning offset的长度要求为1且元素要求大于或等于0.
    //!
    SVector<int64_t> offset;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 后处理累积和计算.
//!
struct CumsumParam {
    //!
    //! \brief 指定axis轴(维度)上计算累加和，只能包含一个轴索引.
    //!
    //! \warning axes的值必须小于输入x的维度数。
    //!
    SVector<int64_t> axes;
    //!
    //! \brief 在某一个轴上的累加结果从第几个元素开始，默认为false.
    //!
    //! \note true：从第一个元素开始（暂不支持） false：从第0个元素开始.
    //!
    bool exclusive = false;
    //!
    //! \brief 正向累加或逆向累加，默认为false.
    //!
    //! \note true：输出逆向累加（暂不支持） false：输出正向累加.
    //!
    bool reverse = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[14] = {0};
};

//!
//! \brief 推理的长度大于训练长度时，embedding需要进行特殊处理。
//! 推理长度小于等于训练长度时，不进行插值；推理长度大于训练长度时，放大base动态插值。
//! 将输入的token序列的位置信息positionIds和inv_freq进行外积，再cos/sin运算得到最终的Rotary embedding的结果。
//!
struct DynamicNTKParam {
    //! 选择输出数据类型的参数
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief 从输入张量中根据索引收集切片，并将这些切片组合成一个新的张量.
//!
struct GatherParam {
    //!
    //! \brief 指定要收集切片的轴。默认值为0.
    //!
    //! \warning 该参数必须大于或等于0
    //!
    int64_t axis = 0;
    //!
    //! \brief  允许从一个batch的每个元素中收集不同的项目，默认值为0.
    //!
    //! \warning 该参数必须大于或等于0,且小于或等于axis.
    //!
    int64_t batchDims = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 采样功能。对最后一个轴进行采样，随机抽取numSamples个值，输出下标。
//!
//! \warning 用户需确保对最后一个轴进行归一化操作。
//!
struct MultinomialParam {
    //!
    //! \brief 随机采样数.
    //!
    //! \warning 小于等于输入张量对应的维度大小，最大为64。
    //!
    uint32_t numSamples = 1;
    //! \brief 随机数种子.
    uint32_t randSeed = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 对输入张量指定维度切成多个张量。
//!
struct SplitParam {
    //!
    //! \brief 指定切分的维度索引
    //!
    //! splitDim须位于输入张量x的维度范围内，即如果x的维度为xDim，则等长切分下splitDim的取值范围为[-xDim, xDim - 1]。
    //! 当splitDim为负数时，其含义是从最高维度开始访问，如splitDim = -1，x维度数为dimNum，则拆分维度为dimNum - 1。
    //! \warning 当使用不等长切分时，splitDim的取值范围为[0, xDim - 1]。
    //!
    int32_t splitDim = 0;
    //!
    //! \brief 切分次数,当前支持2或3.
    //!
    //! \warning 等长切分下输入张量x的维度须能够被splitNum整除,且当splitNum = 3时输入x要求是float16或者bf16数据类型。
    //!
    int32_t splitNum = 2;
    //!
    //! \brief 指定每个输出tensor在切分维度上的大小
    //!
    //! 不传入此参数时使用等长切分，传入此参数时使用splitV不等长切分
    //! \warning splitSizes中的每一个元素要求大于等于1。splitSizes中的元素之和等于切分维度的大小。
    //!
    SVector<int32_t> splitSizes = {};
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 将两个输入张量在指定维度拼接成一个输出张量
//!
struct ConcatParam {
    //!
    //! \brief 指定拼接的维度索引
    //!
    //! 当concatDim为负数时，其含义是从最高维度开始访问，如concatDim = -1，输入张量维度数为dimNum，则拼接维度为dimNum - 1。
    //!
    //! \warning 输入x和y的维数要求一致。输入x或y的维度大小，除了concatDim维外，其他维度要求相同。Atlas 推理系列产品中不支持bf16类型数据。
    //!
    int concatDim = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief 从输入张量某个起始位置中提取指定大小的切片
//!
struct SliceParam {
    //!
    //! \brief 每个维度切片的起始位置
    //!
    //! 当offsets[i]为负数时，其含义是第i维最高维度开始访问，如offsets= -1，输入x的维度为dimNum，则对应维度切片的起始位置为dimNum - 1。
    //!
    //! \warning 当offsets元素x小于0时，该元素对应的维度大小为dimNum，要求dimNum与x之和大于等于0。
    //!
    SVector<int64_t> offsets;
    //!
    //! \brief 每个维度切片的大小
    //!
    //! 当size = -1时，表示切片的结束位置是对应维度最后一个位置。如果对应维度大小为dimNum，则结束位置为dimNum - 1。
    //!
    //! \warning size中元素要求大于等于-1。对应维度offsets，以及offsets + size须在x的对应维度的大小范围内。
    //!
    SVector<int64_t> size;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief Softmax多分类激活函数，将多维（最大8维）Tensor数据在指定轴上映射到0到1之间，且softmax轴数值之和为1。
//!
struct SoftmaxParam {
    //!
    //! \brief 指定轴（维度），axes可以支持多个轴上进行处理
    //!
    //! \warning axes不能为空，当指定多个轴时，多个轴之间必须连续且从小到大排列。
    //! \warning axes的元素要求大于或等于-1且小于输入x的维度
    //!
    SVector<int64_t> axes;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 改变输入Tensor的排列顺序，在多个维度上进行转置
//!
struct TransposeParam {
    //! 指示输入维度的重排结果, 需要保证输入正确，维度和输入x一致
    SVector<int32_t> perm;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct ElewiseParam
//!
//! \brief 常用的逐元素数值计算集合
//!
//! ELEWISE_ADD、ELEWISE_MUL、ELEWISE_REALDIV、ELEWISE_SUB计算类型将会对输入进行广播后再进行指定操作。
//! 输入x、y对应维度的对应值要求相同或至少其中一个为1
//!
struct ElewiseParam {
    //!
    //! \enum ElewiseType
    //!
    //! \brief 计算类型
    //!
    enum ElewiseType : int {
        ELEWISE_UNDEFINED = 0,       //!< 默认值，未定义
        ELEWISE_CAST,                //!< 数据类型转换
        ELEWISE_MULS,                //!< 向量逐元素乘值
        ELEWISE_COS,                 //!< 逐元素计算余弦值
        ELEWISE_SIN,                 //!< 逐元素计算正弦值
        ELEWISE_NEG,                 //!< 逐元素取相反数
        ELEWISE_QUANT,               //!< 量化, 仅在Atlas 800I A2推理产品上支持
        ELEWISE_LOGICAL_NOT,         //!< 逐元素逻辑非
        ELEWISE_ADD,                 //!< 逐元素相加
        ELEWISE_MUL,                 //!< 向量与向量逐元素相乘
        ELEWISE_REALDIV,             //!< 向量与向量逐元素相除
        ELEWISE_LOGICAL_AND,         //!< 逐元素逻辑与
        ELEWISE_LOGICAL_OR,          //!< 逐元素逻辑或
        ELEWISE_LESS,                //!< 逐元素判断是否小于
        ELEWISE_GREATER,             //!< 逐元素判断是否大于
        ELEWISE_SUB,                 //!< 逐元素相减
        ELEWISE_EQUAL,               //!< 逐元素判断是否相等
        ELEWISE_QUANT_PER_CHANNEL,   //!< 每个通道量化
        ELEWISE_DEQUANT_PER_CHANNEL, //!< 每个通道反量化
        ELEWISE_DYNAMIC_QUANT,       //!< 逐行动态量化
        ELEWISE_TANH,                //!< 逐元素计算双曲正切值
        ELEWISE_TYPE_MAX             //!< 边界值，仅用于判断是否出界，所有情况不能取该值
    };

    //! 量化（非每通道）所需参数
    struct QuantParam {
        //! 量化的步长
        float inputScale = 1.0f;
        //! 动态量化的是否为非对称量化
        bool asymmetric = false; //!< false : symmetric，true : asymmetric
        //! 量化的偏移度
        int inputOffset = 0;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };

    //! 向量乘值所需参数
    struct MulsParam {
        //! 向量乘的值
        float varAttr = 0.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[12] = {0};
    };

    //! 计算方式
    ElewiseType elewiseType = ELEWISE_UNDEFINED;
    //! 量化参数
    QuantParam quantParam;
    //! 乘值参数
    MulsParam mulsParam;
    //! 指定数据类型转换输出的数据类型
    aclDataType outTensorType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct KvCacheParam
//!
//! \brief KVCache处理。
//!
struct KvCacheParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct GatingParam
//!
//! \brief 主要功能为将token和专家的映射关系反转为专家与token的映射关系。算子输入为MoE模型每个token选中专家的索引，算子输出为MoE模型每个专家对应的token的索引。
//!
//! \note 该算子支持TP和EP场景，当参数deviceExpert为空时，为TP场景，否则为EP场景。
//!
//! \warning 非Atlas 800I A2推理产品仅支持TP场景。
//!
struct GatingParam {
    //!
    //! \brief 每个token选中的专家数。
    //!
    //! \note 默认值为1。
    //!
    //! \warning 当cumSumNum为0时，取值为1；否则，取值范围为(0, cumSumNum]。
    //!
    int32_t topkExpertNum = 1;
    //!
    //! \brief 专家总数。
    //!
    //! \note 默认值为0。
    //!
    //! \warning 取值范围为[0, 200]。
    //!
    int32_t cumSumNum = 0;
    //!
    //! \brief 输出的cumSum的类型是否为int64。
    //!
    //! \note 默认值为false。
    //!
    //! \warning 当为false时，输出的cumSum类型为int32.
    //!
    bool cumSumInt64 = false;
    //!
    //! \brief 当前device上的专家索引列表。
    //!
    //! \note 默认为空。
    //!
    //! \warning 列表中各个元素取值范围为[0, cumSumNum)，且其中元素值不可重复。
    //!
    //! \warning 当cumSumNum为0时，不可为空。
    //!
    std::vector<int32_t> deviceExpert;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 遍历每个key和value，将key和value(num_heads, head_size)按照slotmapping填入key_cache/value_cache指定位置
//!
struct ReshapeAndCacheParam {
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    //! \note 默认值为COMPRESS_TYPE_UNDEFINED(0)，不开启压缩功能。
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为非COMPRESS_TYPE_UNDEFINED(0)的值
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0, //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD,        //!< alibi场景下压缩key_cache, value_cahe的kvHead维度
        COMPRESS_TYPE_KVHEAD_ROPE    //!< rope场景下压缩key_cache, value_cahe的kvHead维度
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置
    //!
    //! \note 默认值为K_CACHE_V_CACHE(0)，传入key_cache和value_cache
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为K_CACHE_V_BYPASS(1)
    //!
    enum KvCacheCfg : int {
        K_CACHE_V_CACHE = 0, //!< 默认值,传入key_cache和value_cache
        K_CACHE_V_BYPASS,    //!< 只传入key_cache
        K_CACHE_V_CACHE_NZ   //!< 传入key_cache和value_cache,且为NZ格式
    };

    //! 压缩方式
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvCacheCfg = K_CACHE_V_CACHE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 遍历每个key和value，将key和value(num_heads, head_size)按照slotmapping填入key_cache/value_cache指定位置
//!
struct ReshapeAndCacheWithStrideParam {
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    //! \note 默认值为COMPRESS_TYPE_UNDEFINED(0)，不开启压缩功能。
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为非COMPRESS_TYPE_UNDEFINED(0)的值
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0, //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD,        //!< alibi场景下压缩key_cache, value_cahe的kvHead维度
        COMPRESS_TYPE_KVHEAD_ROPE    //!< rope场景下压缩key_cache, value_cahe的kvHead维度
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置
    //!
    //! \note 默认值为K_CACHE_V_CACHE(0)，传入key_cache和value_cache
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为K_CACHE_V_BYPASS(1)
    //!
    enum KvCacheCfg : int {
        K_CACHE_V_CACHE = 0, //!< 默认值,传入key_cache和value_cache
        K_CACHE_V_BYPASS,    //!< 只传入key_cache
    };

    //! 压缩方式
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvCacheCfg = K_CACHE_V_CACHE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \struct LayerNormWithStrideParam
//!
//! \brief LayerNormWithStrideParam归一化处理。当前支持：NORM。
//!
//! \warning beginNormAxis维度小于等于输入x的维度。
//! 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct LayerNormWithStrideParam {
    //!
    //! \enum LayerNormType
    //!
    //! \brief 归一化类型：NORM、PRENORM、POSTNORM。
    //!
    enum LayerNormType : int {
        LAYER_NORM_UNDEFINED = 0, //!< 默认值，未定义
        LAYER_NORM_NORM,          //!< norm
        LAYER_NORM_PRENORM,       //!< prenorm
        LAYER_NORM_POSTNORM,      //!< postnorm
        LAYER_NORM_MAX,
    };
    //!
    //! \brief NORM参数。
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 归一化的维度，默认值为0，从第几维开始norm，同时决定输入gamma和beta维度。
        int32_t beginNormAxis = 0;
        //! \brief 归一化的维度，默认值为0，决定从第几维开始把后面的维度按轴合并。
        int32_t beginParamsAxis = 0;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //!
    //! \brief PRENORM参数
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNQUANT。
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        uint64_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //!
    //! \brief POSTNORM参数。
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        uint64_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //! \brief layerType
    LayerNormType layerType = LAYER_NORM_UNDEFINED;
    //! \brief normParam
    NormParam normParam;
    //! \brief preNormParam
    PreNormParam preNormParam;
    //! \brief postNormParam
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};


//!
//! \struct LayerNormParam
//!
//! \brief LayerNorm归一化处理。当前支持三种：NORM、PRENORM、POSTNORM。
//!
//! \warning beginNormAxis维度小于等于输入x的维度。
//! 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct LayerNormParam {
    //!
    //! \enum LayerNormType
    //!
    //! \brief 归一化类型：NORM、PRENORM、POSTNORM。
    //!
    enum LayerNormType : int {
        LAYER_NORM_UNDEFINED = 0, //!< 默认值，未定义
        LAYER_NORM_NORM,          //!< norm
        LAYER_NORM_PRENORM,       //!< prenorm
        LAYER_NORM_POSTNORM,      //!< postnorm
        LAYER_NORM_MAX,
    };
    //!
    //! \brief NORM参数。
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 归一化的维度，默认值为0，从第几维开始norm，同时决定输入gamma和beta维度。
        int32_t beginNormAxis = 0;
        //! \brief 归一化的维度，默认值为0，决定从第几维开始把后面的维度按轴合并。
        int32_t beginParamsAxis = 0;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //!
    //! \brief PRENORM参数
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNQUANT。
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        uint64_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //!
    //! \brief POSTNORM参数。
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        uint64_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //! \brief layerType
    LayerNormType layerType = LAYER_NORM_UNDEFINED;
    //! \brief normParam
    NormParam normParam;
    //! \brief preNormParam
    PreNormParam preNormParam;
    //! \brief postNormParam
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct RmsNormParam
//!
//! \brief RMS归一化处理。
//!
//! \warning 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct RmsNormParam {
    //!
    //! \brief RmsNormType
    //!
    enum RmsNormType : int {
        RMS_NORM_UNDEFINED = 0, //!< 默认值，未定义
        RMS_NORM_NORM,          //!< NORM参数。
        RMS_NORM_PRENORM,       //!< PRENORM参数。
        RMS_NORM_POSTNORM,      //!< POSTNORM参数
    };
    //!
    //! \brief PrecisionMode
    //!
    enum PrecisionMode : int {
        HIGH_PRECISION_MODE = 0, //!< 中间计算使用float类型
        HIGH_PERFORMANCE_MODE,   //!< 中间计算使用float16类型
    };
    //!
    //! \brief ModelType
    //!
    enum ModelType : int {
        LLAMA_MODEL = 0, //!< 默认值，使用Llama rmsnorm的公式
        GEMMA_MODEL,     //!< 使用Gemma rmsnorm的公式
    };
    //!
    //! \brief NormParam
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT, QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief Epsilon，默认为1e-5，暂时不使用。
        double layerNormEps = 1e-5;
        //! \brief 默认为False，设置为true时会使用训练的rmsnormforward算子。仅在Atlas 800I A2推理产品上支持该设置。
        //!  不支持和“precisionMode”，“modelType”同时设置。量化场景下不支持使用“rstd”。
        bool rstd = false;
        //! \brief 默认为HIGH_PRECISION_MODE。
        //! 支持参数如下：
        //! HIGH_PRECISION_MODE：默认值，中间计算使用float类型
        //! HIGH_PERFORMANCE_MODE： 中间计算使用float16类型
        //! 不支持和“rstd”，“modelType”同时设置。输入类型只支持float16。
        //! 量化场景下不支持使用“precisionMode”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        PrecisionMode precisionMode = HIGH_PRECISION_MODE;
        //! \brief 默认为LLAMA_MODEL，设置为GEMMA_MODEL时使用gemma模型的rmsnorm计算公式。
        //! 支持参数如下：
        //! LLAMA_MODEL：默认值， Llama的rms norm计算公式。
        //! GEMMA_MODEL：Gemma的rms norm计算公式。
        //! 不支持和“rstd”，“precisionMode”同时启用。
        //! 量化场景下不支持使用“modelType”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        ModelType modelType = LLAMA_MODEL;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[32] = {0};
    };
    //!
    //! \brief PreNormParam
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT
        //! QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。量化场景下不支持使用“hasBias”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //!
    //! \brief PostNormParam
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNQUANT。
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //! \brief 归一化类型，参数如下：
    //! RMS_NORM_UNDEFINED：默认值，未定义。
    //! RMS_NORM_NORM：NORM参数。
    //! RMS_NORM_PRENORM：PRENORM参数。
    //! RMS_NORM_POSTNORM：POSTNORM参数。
    RmsNormType layerType = RMS_NORM_UNDEFINED;
    //! \brief NORM参数。
    NormParam normParam;
    //! \brief PRENORM参数。
    PreNormParam preNormParam;
    //! \brief POSTNORM参数。
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct RmsNormWithStrideParam
//!
//! \brief RMS归一化处理。
//!
//! \warning 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct RmsNormWithStrideParam {
    //!
    //! \brief RmsNormType
    //!
    enum RmsNormType : int {
        RMS_NORM_UNDEFINED = 0, //!< 默认值，未定义
        RMS_NORM_NORM,          //!< NORM参数。
        RMS_NORM_PRENORM,       //!< PRENORM参数。
        RMS_NORM_POSTNORM,      //!< POSTNORM参数
    };
    //!
    //! \brief PrecisionMode
    //!
    enum PrecisionMode : int {
        HIGH_PRECISION_MODE = 0, //!< 中间计算使用float类型
        HIGH_PERFORMANCE_MODE,   //!< 中间计算使用float16类型
    };
    //!
    //! \brief ModelType
    //!
    enum ModelType : int {
        LLAMA_MODEL = 0, //!< 默认值，使用Llama rmsnorm的公式
        GEMMA_MODEL,     //!< 使用Gemma rmsnorm的公式
    };
    //!
    //! \brief NormParam
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT, QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief Epsilon，默认为1e-5，暂时不使用。
        double layerNormEps = 1e-5;
        //! \brief 默认为False，设置为true时会使用训练的rmsnormforward算子。仅在Atlas 800I A2推理产品上支持该设置。
        //!  不支持和“precisionMode”，“modelType”同时设置。量化场景下不支持使用“rstd”。
        bool rstd = false;
        //! \brief 默认为HIGH_PRECISION_MODE。
        //! 支持参数如下：
        //! HIGH_PRECISION_MODE：默认值，中间计算使用float类型
        //! HIGH_PERFORMANCE_MODE： 中间计算使用float16类型
        //! 不支持和“rstd”，“modelType”同时设置。输入类型只支持float16。
        //! 量化场景下不支持使用“precisionMode”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        PrecisionMode precisionMode = HIGH_PRECISION_MODE;
        //! \brief 默认为LLAMA_MODEL，设置为GEMMA_MODEL时使用gemma模型的rmsnorm计算公式。
        //! 支持参数如下：
        //! LLAMA_MODEL：默认值， Llama的rms norm计算公式。
        //! GEMMA_MODEL：Gemma的rms norm计算公式。
        //! 不支持和“rstd”，“precisionMode”同时启用。
        //! 量化场景下不支持使用“modelType”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        ModelType modelType = LLAMA_MODEL;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[32] = {0};
    };
    //!
    //! \brief PreNormParam
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNQUANT
        //! QUANT_INT8
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。量化场景下不支持使用“hasBias”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //!
    //! \brief PostNormParam
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNQUANT。
        QuantType quantType = QUANT_UNQUANT;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //! \brief 归一化类型，参数如下：
    //! RMS_NORM_UNDEFINED：默认值，未定义。
    //! RMS_NORM_NORM：NORM参数。
    //! RMS_NORM_PRENORM：PRENORM参数。
    //! RMS_NORM_POSTNORM：POSTNORM参数。
    RmsNormType layerType = RMS_NORM_UNDEFINED;
    //! \brief NORM参数。
    NormParam normParam;
    //! \brief PRENORM参数。
    PreNormParam preNormParam;
    //! \brief POSTNORM参数。
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct FillParam
//!
//! \brief 将指定位置设置为value值或者生成一个指定Shape的Tensor并填充为value。
//!
//! \warning 输入x不可以被broadcast。输入mask的元素只能是0或者1，且可以被broadcast。
//!
struct FillParam {
    //! \brief 是否Masked Fill。
    bool withMask = true;
    //! \brief 填充的元素，value是一个只含有一个元素的SVector。
    SVector<float> value;
    //! \brief withMask = false时，表示输出Tensor的Shape。
    SVector<int64_t> outDim;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct AllGatherParam
//!
//! \brief 将多个通信卡上的数据按所属rank号的顺序在第一维进行聚合，然后发送到每张卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllGatherParam {
    //! \brief 当前卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl",Atlas 推理系列产品仅支持backend为"hccl"。
    //!
    //! 当backend为"lccl"时，且若机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct AllGatherVParam
//!
//! \brief 将多个通信卡上的数据按所属rank号的顺序在第一维进行聚合，然后发送到每张卡上.支持每张卡的数据不等长
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllGatherVParam {
    //! \brief 当前卡所属通信编号, 默认值为-1, 代表没传rank参数
    int rank = -1;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl",Atlas 推理系列产品（Ascend 310P AI处理器）仅支持backend为"hccl"。
    //!
    //! 当backend为"lccl"时，且若机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const AllGatherVParam &left, const AllGatherVParam &right)
{
    return left.rank == right.rank && left.rankSize == right.rankSize && left.rankRoot == right.rankRoot &&
           left.hcclComm == right.hcclComm && left.commMode == right.commMode && left.backend == right.backend &&
           left.rankTableFile == right.rankTableFile && left.commDomain == right.commDomain;
}

//!
//! \struct AllReduceParam
//!
//! \brief 将多个通信卡上的数据进行计算，支持相加、取最大、最小、相乘四种计算，然后发送到每张卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllReduceParam {
    //! \brief 量化类型
    enum QuantType : int {
        QUANT_TYPE_UNQUANT = 0,     //!< 默认值
        QUANT_TYPE_UNDEFINED = 0,   //!< 默认值
        QUANT_TYPE_PER_TENSOR = 1,  //!< 对整个张量进行量化
        QUANT_TYPE_PER_CHANNEL = 2, //!< 对张量中每个channel分别进行量化
        QUANT_TYPE_MAX = 3,         //!< 枚举类型最大值
    };

    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 通信计算类型，支持"sum","prod","max"和"min".
    std::string allReduceType = "sum";
    //!
    //! \brief 通信计算类型，仅支持"hccl"和"lccl".Atlas 推理系列产品仅支持backend为"hccl"。
    //!
    //! backend为"hccl"时，支持"sum","prod","max"和"min"; backend为"lccl"时，支持"sum","max"和"min".
    //! 当backend为"hccl"时，allReduceType为"prod"时，不支持数据类型为int16和bf16。
    //! 当backend为"hccl"时，Atlas 推理系列产品不支持int64,bf16,int16只有allReduceType为"sum"时支持
    //! 当backend为"lccl"时，不支持数据类型int64，且若机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //! \brief 量化类型
    QuantType quantType = QUANT_TYPE_UNQUANT;
    //! 若为浮点AllReduce，参数outDataType配置为ACL_DT_UNDEFINED，表示输出tensor的数据类型与输入tensor一致；
    //! 若为量化AllReduce，输出tensor的数据类型与输入tensor不一致，则参数outDataType配置为用户预期输出tensor的数据类型，
    //! 量化只支持配置ACL_FLOAT16
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct BlockCopyParam
//!
//! \brief 将KVCache里通过src indices指定的block数据copy到dst indices指定的block位置上。
//!
struct BlockCopyParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \struct BroadcastParam
//!
//! \brief 将通信主卡上的数据广播到其他每张卡上, 该算子不支持Atlas 推理系列产品。
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!

struct BroadcastParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl"。
    std::string backend = "hccl";
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct ReduceScatterParam
//!
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct ReduceScatterParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 当前通信计算类型仅支持"sum","max"和"min",不支持"prod"。
    std::string reduceType = "sum";
    //! \brief HCCL通信域指针。
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子。
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 通信后端指示，当"backend"为lccl且机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    std::string backend = "lccl";
    //! \brief 集群信息的配置文件路径。
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct ReduceScatterVParam
//!
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct ReduceScatterVParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 表示发送数据量的数组.
    //! 例如，若发送的数据类型为float32，sendCounts[i] = n 表示本rank发给rank i n个float32数据。
    std::vector<int64_t> sendCounts;
    //! \brief 表示发送偏移量的数组.
    //! sdispls[i] = n表示本rank从相对于输入起始位置的的偏移量为n的位置开始发送数据给rank i
    std::vector<int64_t> sdispls;
    //! \brief 表示接收数据量.
    std::int64_t recvCount = 0;
    //!
    //! \brief 当前通信计算类型仅支持"sum","max"和"min",不支持"prod"。
    std::string reduceType = "sum";
    //! \brief HCCL通信域指针。 当前算子仅支持lccl,此参数为预留参数。
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子。
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 通信后端指示，当前算子仅支持"hccl"
    std::string backend = "hccl";
    //! \brief 集群信息的配置文件路径。
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识。
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const ReduceScatterVParam &left, const ReduceScatterVParam &right)
{
    return left.rank == right.rank && left.rankSize == right.rankSize && left.rankRoot == right.rankRoot &&
           left.sendCounts == right.sendCounts && left.sdispls == right.sdispls && left.recvCount == right.recvCount &&
           left.reduceType == right.reduceType && left.hcclComm == right.hcclComm && left.commMode == right.commMode &&
           left.backend == right.backend && left.rankTableFile == right.rankTableFile &&
           left.commDomain == right.commDomain;
}

//!
//! \struct LinearParam
//!
//! \brief 将A、B两个矩阵进行矩阵乘运算，同时可以选择对矩阵乘的运算结果进行叠加偏置、InplaceAdd融合或反量化操作。
//!
//! \note 算子本质上是接收x和weight两个输入tensor作为A矩阵和B矩阵进行矩阵乘运算，可通过参数transposeA与transposeB控制做矩
//! 阵乘前是否需要对A矩阵和B矩阵进行行列转置，根据参数转置后的A矩阵和B矩阵需满足矩阵乘维度关系。例如，当transposeA为false，
//! transposeB为true时，x和weight的shape可以分别为[m, k]和[n, k]。
//!
//! \note 该算子支持浮点和量化场景，当参数outDataType值为ACL_DT_UNDEFINED时为浮点场景，否则为量化场景。
//!
struct LinearParam {
    //! \brief Matmul所有计算类型。
    enum MatmulType : uint8_t {
        MATMUL_UNDEFINED = 0,
        MATMUL_EIN_SUM
    };
    //! \brief Matmul不同量化类型。
    enum QuantMode : uint8_t {
        QUANT_UNDEFINED,
        PER_CHANNEL,
        PER_TOKEN
    };
    //!
    //! \brief 是否转置A矩阵。
    //!
    //! \note 默认值为false，不转置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为false。
    //!
    bool transposeA = false;
    //!
    //! \brief 是否转置B矩阵。
    //!
    //! \note 默认值为true，转置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为true。
    //!
    bool transposeB = true;
    //!
    //! \brief 是否叠加偏置。
    //!
    //! \note 默认值为true，叠加偏置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为true。
    //!
    //! \warning enAccum为true时，仅支持配置为false。
    //!
    bool hasBias = true;
    //!
    //! \brief 输出数据类型。
    //!
    //! \note 默认值为ACL_DT_UNDEFINED。
    //!
    //! \warning 浮点场景下：支持配置为ACL_DT_UNDEFINED。
    //!
    //! \warning 量化场景下：Atlas 800I A2推理产品支持配置为ACL_FLOAT16/ACL_BF16，否则，仅支持配置为ACL_FLOAT16。
    //!
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 是否使能累加。
    //!
    //! \note 默认值为false，不使能累加。
    //!
    //! \warning 仅在Atlas 800I A2推理产品支持配置为true。
    //!
    //! \warning hasBias为true时，仅支持配置为false。
    //!
    //! \warning 量化场景下，仅支持配置为false。
    //!
    bool enAccum = false;
    //!
    //! \brief matmul计算类型
    //!
    //! \note 默认值为MATMUL_UNDEFINED，非爱因斯坦乘场景。
    //!
    //! \warning 取值范围为MATMUL_UNDEFINED/MATMUL_EIN_SUM。
    //!
    MatmulType matmulType = MATMUL_UNDEFINED;
    //!
    //! \brief matmul量化类型
    //!
    //! \note 默认值为QUANT_UNDEFINED。
    //!
    //! \warning 取值范围为PER_CHANNEL/PER_TOKEN。
    //!
    QuantMode quantMode = QUANT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[21] = {0};
};

//!
//! \struct LinearParallelParam
//!
//! \brief 通信计算并行算子,该算子功能为linear和通信算子组合
//!
//! 通信和计算是并行处理，和串行相比存在大幅度性能提升.
//!
//! \see LinearParam,AllReduceParam,AllGatherParam
//!
struct LinearParallelParam {
    //!
    //! \enum ParallelType
    //!
    //! \brief 通信类型
    //!
    enum ParallelType : int {
        UNDEFINED = -1,                       //!< 默认值
        LINEAR_ALL_REDUCE = 0,                //!< linear+AllReduce
        LINEAR_REDUCE_SCATTER = 1,            //!< linear+reduce_scatter
        ALL_GATHER_LINEAR = 2,                //!< AllGather+linear
        PURE_LINEAR = 3,                      //!< linear
        ALL_GATHER_LINEAR_REDUCE_SCATTER = 4, //!< AllGather+linear+reduce_scatter
        ALLTOALLVC_ALL_GATHER_GMM = 5,        //!< AllToAllvc+AllGather+GroupMatmul
        GMM_REDUCE_SCATTER_ALLTOALLVC = 6,    //!< GroupMatmul+ReduceScatter+AllToAllvc
        MAX = 7,                              //!< 枚举类型最大值
    };
    //!
    //! \enum QuantType
    //!
    //! \brief QuantType类型
    //!
    enum QuantType : int {
        QUANT_TYPE_UNDEFINED = -1,  //!< 默认值
        QUANT_TYPE_UNQUANT = -1,    //!< 默认值
        QUANT_TYPE_PER_TENSOR = 0,  //!< 对整个张量进行量化
        QUANT_TYPE_PER_CHANNEL = 1, //!< 对张量中每个channel分别进行量化
        QUANT_TYPE_PER_GROUP = 2,   //!< 将张量按quantGroupSize划分后，分别进行量化
        QUANT_TYPE_PER_TOKEN = 3,   //!< 对张量中每个token分别进行量化
        QUANT_TYPE_MAX = 4,         //!< 枚举类型最大值
    };
    //! \brief 权重是否需要转置，默认为true。
    bool transWeight = true;
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 是否叠加残差。配置为false时不叠加残差，为true时叠加残差。默认不叠加残差。
    bool hasResidual = false;
    //! \brief 通信后端指示。支持"hccl"，"lccl"，"lcoc", "mc2"。
    std::string backend = "hccl";
    //! \brief HCCL通信域接口获取的地址指针，仅当"hcclComm"不为nullptr时可用。
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景。
    std::string rankTableFile;
    //! \brief 权重并行类型。
    ParallelType type = LINEAR_ALL_REDUCE;
    //! \brief 是否返回中间结果，仅在使用ALL_GATHER_LINEAR时生效。
    bool keepIntermediate = false;
    //! \brief 量化类型。
    QuantType quantType = QUANT_TYPE_UNQUANT;
    //! \brief 量化类型为QUANT_TYPE_PER_GROUP时生效。
    int32_t quantGroupSize = 0;
    //!
    //! 若为浮点linear，参数outDataType配置为ACL_DT_UNDEFINED，表示输出tensor的数据类型与输入tensor一致,
    //! 若为量化linear，输出tensor的数据类型与输入tensor不一致，则参数outDataType配置为用户预期输出tensor的数据类型,
    //! 如ACL_FLOAT16/ACL_BF16
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //! \brief AllGather_Matmul_ReduceScatter算子参数结构体
    struct TwoDimTPInfo {
        //! \brief 表示ag轴卡数，规定x轴方向是非连续卡号
        uint16_t agDim = 0;
        //! \brief 表示rs轴卡数，规定y轴方向是连续卡号
        uint16_t rsDim = 0;
        //! \brief 是否沿着内轴进行allgather通信
        uint8_t innerDimIsAg = 1;
        //! \brief 填充满8字节
        uint8_t rsv[3] = {0};
    };
    //! \brief AllGather_Matmul_ReduceScatter算子参数
    TwoDimTPInfo twoDimTPInfo;
    //! \brief Moe 算子参数结构体
    struct MoeInfo {
        //! \brief MOE场景，每个NPU处理的expert数量
        int16_t localExpertNums = 1;
        //! \brief MOE场景，EP通信域大小
        int8_t epSize = 1;
        //! \brief MOE场景，TP通信域大小
        int8_t tpSize = 1;
    };
    //! \brief AllGather_Matmul_ReduceScatter算子参数
    MoeInfo moeInfo;

    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[52] = {0};
};

//!
//! \struct LinearSparseParam
//!
//! \brief 稀疏量化linear
//!
//! 该算子实现功能与量化linear类似。不同点在于稀疏量化算子会使用压缩工具提前对weight输入进行压缩，
//! 以此提升算子性能。参数tilingK和tilingN由压缩算法决定，目前均只支持取值为8.
//! 目前该算子仅支持在Atlas 推理系列产品中进行运算。
//!
struct LinearSparseParam {
    //! \brief 是否转置A矩阵，默认不转置。当前仅支持transposeA = false。
    bool transposeA = false;
    //! \brief 是否转置B矩阵，默认转置。当前仅支持transposeB = true。
    bool transposeB = true;
    //! \brief 压缩参数，由外部压缩算法决定，默认为8，目前仅支持取值为8。
    uint32_t tilingK = 8;
    //! \brief 压缩参数，由外部压缩算法决定，默认为8，目前仅支持取值为8。
    uint32_t tilingN = 8;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief 旋转位置编码。hiddenSizeQ必须是hiddenSizeK的整数倍且满足hiddenSizeQ = headDim * headNum。
//!
struct RopeParam {
    //! \brief rope，旋转系数，对半旋转是2，支持配置2、4或headDim / 2。
    int32_t rotaryCoeff = 4;
    //! \brief 训练用参数，支持配置0或1
    int32_t cosFormat = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const RopeParam &left, const RopeParam &right)
{
    return left.rotaryCoeff == right.rotaryCoeff && left.cosFormat == right.cosFormat;
}

//!
//! \brief 旋转位置编码后进行concat操作。hiddenSizeQ必须是hiddenSizeK的整数倍且满足hiddenSizeQ = headDim * headNum。
//!
struct RopeQConcatParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const RopeQConcatParam &left, const RopeQConcatParam &right)
{
    (void)left;
    (void)right;
    return true;
}

//!
//! \struct RelayAttentionParam
//!
//! \brief 通过减少共享组的kv搬运来优化模型吞吐量
//!
//!
struct RelayAttentionParam {
    //!
    //! \brief head数量
    //!
    //! \note 默认值为0
    //!
    int32_t headNum = 0;
    //!
    //! \brief 算子tor值
    //!
    //! \note 默认值为1.0
    //!
    float qkScale = 1;
    //!
    //! \brief kv头数量
    //! \warning 取值范围为[0,8]
    //! \note 默认值为0
    //!
    int32_t kvHeadNum = 0;
    //!
    //! \enum MaskType
    //!
    //! \brief mask类型
    //!
    enum MaskType : int {
        MASK_TYPE_UNDEFINED = 0, //!< 默认值，全0mask
        MASK_TYPE_NORM,          //!< 倒三角mask
    };
    //!
    //! \brief mask类型
    //!
    //! \note 默认值为MASK_TYPE_UNDEFINED
    //!
    MaskType maskType = MASK_TYPE_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[32] = {0};
};

//!
//! \brief KVCache+KVCache+Muls+FlashAttention.
//!
struct SelfAttentionParam {
    //!
    //! \enum CalcType
    //!
    //! \brief 计算类型
    //!
    enum CalcType : int {
        UNDEFINED = 0,  //!< decoder&encoder for flashAttention
        ENCODER,        //!< encoder for flashAttention
        DECODER,        //!< decoder for flashAttention
        PA_ENCODER,     //!< encoder for pagedAttention
        PREFIX_ENCODER, //!< prefix encoder for flashAttention
    };
    //!
    //! \enum KernelType
    //!
    //! \brief 算子内核精度类型
    //!
    enum KernelType : int {
        KERNELTYPE_DEFAULT = 0,   //!< i:float16, bmm:float16, o:float16
        KERNELTYPE_HIGH_PRECISION //!< i:float16, bmm:float, o:float16
    };
    //!
    //! \enum ClampType
    //!
    //! \brief clamp类型
    //!
    enum ClampType : int {
        CLAMP_TYPE_UNDEFINED = 0, //!< 不做clamp
        CLAMP_TYPE_MIN_MAX        //!< 做clamp，同时指定最大最小值
    };
    //!
    //! \enum MaskType
    //!
    //! \brief mask类型
    //!
    enum MaskType : int {
        MASK_TYPE_UNDEFINED = 0,             //!< 默认值，全0mask
        MASK_TYPE_NORM,                      //!< 倒三角mask
        MASK_TYPE_ALIBI,                     //!< alibi mask
        MASK_TYPE_NORM_COMPRESS,             //!< 倒三角压缩mask
        MASK_TYPE_ALIBI_COMPRESS,            //!< alibi压缩mask
        MASK_TYPE_ALIBI_COMPRESS_SQRT,       //!< alibi压缩开平方mask
        MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN, //!< alibi压缩mask左对齐,只支持Atlas 800I A2推理产品
        MASK_TYPE_SLIDING_WINDOW_NORM,       //!< sliding window attention mask
        MASK_TYPE_SLIDING_WINDOW_COMPRESS,   //!< sliding window attention压缩mask
        MASK_TYPE_CAUSAL_MASK,               //!< mask内部生成
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置,不支持calcType为PA_ENCODER
    //!
    enum KvCacheCfg : int {
        K_CACHE_V_CACHE = 0, //!< 默认值,进行kvcache处理
        K_BYPASS_V_BYPASS,   //!< 直接传入kvcache
    };
    //!
    //! \enum ScaleType
    //!
    //! \brief The type values of ScaleType.
    //!
    enum ScaleType : int {
        SCALE_TYPE_TOR = 0, //!< 默认值，不开启LogN缩放
        SCALE_TYPE_LOGN,    //!< 注意力使用LogN缩放，quantType只能是0
        SCALE_TYPE_MAX      //!< 边界值，仅用于判断是否出界
    };

    //! \enum QuantType
    //!
    //! \brief quant类型
    //!
    enum QuantType : int {
        TYPE_QUANT_UNDEFINED = 0,   //!< 默认值，不与量化融合，此时q，k，v为bf16/float16
        TYPE_QUANT_UNQUANT = 0,     //!< 默认值，不与量化融合，此时q，k，v为bf16/float16
        TYPE_DEQUANT_FUSION = 1,    //!< 与反量化融合, 预留类型，当前不能够取此值。
        TYPE_QUANT_QKV_OFFLINE = 2, //!< 离线INT8量化, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_ONLINE = 3   //!< 在线INT8量化, 只支持Atlas 800I A2推理产品
    };
    //!
    //! \enum CacheType
    //!
    //! \brief cache内部排布类型, 为CACHE_TYPE_SWA开启SWA KVCache优化，只储存后windowSize个token的KVCache，
    //!  控制KVCache的长度不超过windowSize, 以此减少显存占用
    //!
    enum CacheType : int8_t {
        CACHE_TYPE_NORM = 0, //!< 正常cache
        CACHE_TYPE_SWA = 1   //!< 固定长度cache
    };
    //!
    //! 量化类型(只支持PA_ENCODER)：
    //! 当值为TYPE_QUANT_QKV_OFFLINE或TYPE_QUANT_QKV_ONLINE时q，k，v为int8。key,value的headsize等长，范围为（0, 256]，
    //! 且32对齐。outdatatype需要配置，只能是ACL_FLOAT16或ACL_BF16。inputLayout只支持TYPE_BSND，calcType只能为PA_ENCODER。
    QuantType quantType = TYPE_QUANT_UNQUANT;

    //! output数据类型：只支持PA_ENCODER,且QuantType不为TYPE_QUANT_UNQUANT（格式为aclDataType）
    aclDataType outDataType = ACL_DT_UNDEFINED;

    //! query头大小, 需大于0
    int32_t headNum = 0;
    //! kv头数量, 该值需要用户根据使用的模型实际情况传入
    //! kvHeadNum = 0时，keyCache的k_head_num，valueCache的v_head_num与query的num_heads一致，均为num_heads的数值
    //! kvHeadNum != 0时，keyCache的k_head_num， valueCache的v_head_num与kvHeadNum值相同
    int32_t kvHeadNum = 0;
    //! query缩放系数
    float qScale = 1;
    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1;
    //! 是否开启动态batch
    bool batchRunStatusEnable = false;
    //! 是否开启倒三角优化, 只有mask为倒三角的时候才能开启优化
    uint32_t isTriuMask = 0;
    //! 计算类型
    CalcType calcType = UNDEFINED;
    //! 内核精度类型
    KernelType kernelType = KERNELTYPE_DEFAULT;
    //! clamp类型
    ClampType clampType = CLAMP_TYPE_UNDEFINED;
    //! clamp功能最小值
    float clampMin = 0;
    //! clamp功能最大值
    float clampMax = 0;
    //! mask类型
    MaskType maskType = MASK_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvcacheCfg = K_CACHE_V_CACHE;
    //! scale类型
    ScaleType scaleType = SCALE_TYPE_TOR;
    //! 数据排布格式默认为BSND
    InputLayout inputLayout = TYPE_BSND;
    //! \brief 大于0时开启MLA合并kvcache功能，表示kv合并传入时v的head_size
    //! \note 默认值为0
    //! \warning 取值范围为[0,576]
    uint32_t mlaVHeadSize = 0;
    //! \brief cache内部排布，开启SWA特性并设置为CACHE_TYPE_SWA可以开启SWA cache优化
    //! \note 默认值为CACHE_TYPE_NORM
    //! \warning 只有开启SWA特性后才可以是CACHE_TYPE_SWA
    CacheType cacheType = CACHE_TYPE_NORM;
    //! \brief windowSize大于0时开启SWA特性，开启SWA特性后表示sliding window 大小
    //! \note 默认值为0
    //! \warning windowSize大于0时需要将maskType设置为MASK_TYPE_SLIDING_WINDOW_NORM或MASK_TYPE_SLIDING_WINDOW_COMPRESS
    uint32_t windowSize = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \brief PagedAttention.
//!
//! 一个Q有多个token，一个token对应多个KV的token，以token0为例，block_table代表其对应的KV的block_id，-1代表截止，
//! 所以第二行和第四行为其目标block，context_lens则表示KV有多少个token，则代表仅有block_id为(3,4,5,9,10)是需要与Q进行计算的。
//!
struct PagedAttentionParam {
    //! query 头大小
    int32_t headNum = 0;
    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1.0;
    //! kv头数量
    int32_t kvHeadNum = 0;
    //!
    //! \enum MaskType
    //!
    //! \brief The type values of MaskType.
    //!
    enum MaskType : int {
        UNDEFINED = 0,   //!< 默认值，全0的mask
        MASK_TYPE_NORM,  //!< 倒三角mask
        MASK_TYPE_ALIBI, //!< alibi mask
        MASK_TYPE_SPEC,   //!< 并行解码mask
        MASK_TYPE_MASK_FREE //! mask_free 只支持fp16
    };
    //! mask类型
    MaskType maskType = UNDEFINED;
    //! 是否开启动态batch
    bool batchRunStatusEnable = false;
    //!
    //! \enum QuantType
    //!
    //! \brief quant类型
    //!
    enum QuantType : int {
        TYPE_QUANT_UNDEFINED = 0,   //!< 默认值，不与量化融合，此时q，k，v为bf16/float16
        TYPE_QUANT_UNQUANT = 0,     //!< 默认值，不与量化融合，此时q，k，v为bf16/float16
        TYPE_DEQUANT_FUSION = 1,    //!< 与反量化融合, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_OFFLINE = 2, //!< 离线INT8量化, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_ONLINE = 3   //!< 在线INT8量化, 只支持Atlas 800I A2推理产品
    };
    //!
    //! 量化类型：
    //! 为TYPE_QUANT_UNQUANT时q，keyCache，valueCache为bf16/float16。
    //! 为TYPE_DEQUANT_FUSION时q为bf16/float16，keyCache，valueCache为int8。
    //! 为TYPE_QUANT_QKV_OFFLINE或TYPE_QUANT_QKV_ONLINE时q，keyCache，valueCache为int8。
    //! keyCache,valueCache的headsize等长，范围为（0, 256]，且block_size * head_size ≤ 128 * 128。
    //! outdatatype需要配置，只能是ACL_FLOAT16或ACL_BF16。inputLayout只支持TYPE_BSND。
    QuantType quantType = TYPE_QUANT_UNQUANT;

    //! output数据类型（格式为aclDataType）
    aclDataType outDataType = ACL_DT_UNDEFINED;

    //! 开启量化功能后是否使用offset
    bool hasQuantOffset = false;
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0, //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD,        //!< 压缩key_cache, value_cache的kvHead维度, 只支持Atlas 800I A2推理产品。
        COMPRESS_TYPE_KVHEAD_ROPE,   //!< rope场景压缩key_cache, value_cache的kvHead维度, 只支持Atlas 800I A2推理产品。
        COMPRESS_TYPE_MAX            //!< 压缩类型边界值，仅用于判断是否出界，所有情况不能取该值。
    };
    //!
    //! 压缩方式
    //! 为COMPRESS_TYPE_KVHEAD时，不支持quanttype为2和3。
    //! 为COMPRESS_TYPE_KVHEAD_ROPE时, maskType需传0。不支持quanttype为2和3。
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //!
    //! \enum CalcType
    //!
    //! \brief The type values of CalcType.
    //!
    enum CalcType : int {
        CALC_TYPE_UNDEFINED = 0, //!< 默认值，不开启并行解码
        CALC_TYPE_SPEC           //!< 此计算模式支持传入长度大于1的qseqlen，启用并行解码功能
    };
    //! 计算类型
    CalcType calcType = CALC_TYPE_UNDEFINED;

    //!
    //! \enum ScaleType
    //!
    //! \brief The type values of ScaleType.
    //!
    enum ScaleType : int {
        SCALE_TYPE_TOR = 0, //!< 默认值，不开启LogN缩放
        SCALE_TYPE_LOGN,    //!< 注意力使用LogN缩放
        SCALE_TYPE_MAX      //!< 边界值，仅用于判断是否出界
    };
    //! scale类型
    //! 为SCALE_TYPE_LOGN时，不支持quanttype为2和3。
    ScaleType scaleType = SCALE_TYPE_TOR;

    //! 数据排布格式默认为BSND
    InputLayout inputLayout = TYPE_BSND;
    //! \brief 大于0时开启MLA合并kvcache功能，表示kv合并传入时v的head_size
    //! \note 默认值为0
    //! \warning 取值范围为[0,576]
    uint32_t mlaVHeadSize = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[68] = {0};
};

//!
//! \brief 数据格式转换处理。
//!
//! 使用的NZ的dims约定表示方式：{b, n1, m1m0, n0}，对应的ND的dims是{b, m, n}，
//! 其中：b表示batch，如果batch为1，该维度为1，不可省略。如果batch有多个，该维度为所有batch维度合轴的结果。
//! m0/n0表示对齐位，float16时，n0与m0都为16, int8时，n0为32，m0为16，m1m0表示原始ND的m维度经过对齐位向上对齐，
//! n1表示原始ND的n维度经过对齐位向上对齐后，除以n0的商。例如原始ND的dims为{8, 100, 30}，则其对应的NZ的dims为{8, 2, 112, 16}。
//!
//! \warning outCrops的长度要求是2，其值须满足以下要求：
//! - 如果m0m1落在区间(k1 × 16, (k1 + 1) × 16]（其中k1为正整数）内，那么该区间即为outCrops[0]的取值范围要求。
//! - 如果n0*n1落在区间(k2 × 16, (k2 + 1) × 16]（其中k2为正整数）内，那么该区间即为outCrops[1]的取值范围要求。
//!
struct TransdataParam {
    //!
    //! \enum TransdataType
    //!
    //! \brief TransdataType类型值
    //!
    enum TransdataType : int {
        UNDEFINED = 0,    //!< 默认
        FRACTAL_NZ_TO_ND, //!< FRACTAL_NZ转ND
        ND_TO_FRACTAL_NZ  //!< ND转FRACTAL_NZ
    };
    //! \brief 数据格式转换类型，支持FRACTAL_NZ和ND互相转换。
    TransdataType transdataType = UNDEFINED;
    //! \brief 仅当FRACTAL_NZ转ND时使用，表示原ND数据格式Shape的最后两维。
    SVector<int64_t> outCrops = {0, 0};
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 三目运算。
//!
//! 输入张量为cond,x,y, 输出张量 z = cond ? x : y;
//! 输入cond的元素只能是0或者1
//! 输出z的维度为输入x与y广播后的结果。要求cond, x, y必须是可广播的。
//!
struct WhereParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 将输入Tensor的Shape，按指定轴扩展指定的倍数。
//!
//! \warning 输出y的维度和multiples维度一致，每个维度大小为输入x广播到multiples维度后和multiples对应维度的乘积。
//!
struct RepeatParam {
    //!
    //! \brief 每一维度上扩展的倍数。
    //!
    //! \warning
    //! - 支持在不超过两个维度上进行扩展
    //! - multiples的维度小于等于8且需大于或等于输入x的维度，每一个元素要求大于0。
    //!
    SVector<int64_t> multiples;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct SetValueParam
//!
//! \brief 将输入源张量中的内容拷贝到输入目标张量指定位置中.
//!
//! 该拷贝为原地拷贝，最终结果修改在输入目标张量中.<br>
//! 输入目标张量 dst: [a,b,c], 输入源张量src: [d,e,f].
//! dst[starts[0]: ends[0], starts[1]: ends[1], starts[2]: ends[2]] = src.<br>
//! 其中 ends[0]-starts[0]需为src第0维的维度大小,ends[1]-starts[1]需为为src第1维的维度大小,ends[2]-starts[2]需为src第2维的维度大小。
//!
//! \warning 输入src和输入dst的维数须相同.<br>
//! 输入src的各维度大小要求小于或等于输入dst对应维度大小.<br>
//! 输入src和输入dst的各维度要求有一个或两个维度不相同，且需要满足：
//!   - 如果有一个维度不相同，则这个维度不能是最高维（第0维）。
//!   - 如果有两个维度不相同，则其中一个不同的维度必须是最高维（第0维）。
//
struct SetValueParam {
    //! \brief 每一维拷贝起始位置
    SVector<int64_t> starts;
    //! \brief 每一维拷贝结束位置后一个位置，拷贝到该位置前一个位置为止
    SVector<int64_t> ends;
    //! \brief 每一维拷贝步长，当前仅支持strides为全1.
    SVector<int64_t> strides;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 在指定维度上求和、取最大值或最小值，并消除这个维度。
//!
struct ReduceParam {
    //!
    //! \enum ReduceType
    //!
    //! \brief ReduceType支持的值
    //!
    enum ReduceType {
        REDUCE_UNDEFINED = 0, //!< 未定义。
        REDUCE_MAX,           //!< 求最大值。
        REDUCE_MIN,           //!< 求最小值。
        REDUCE_SUM,           //!< 求和。
    };
    //! \brief reduceType
    ReduceType reduceType = REDUCE_UNDEFINED;
    //!
    //! \brief 指定轴（维度）。
    //!
    //! \warning axis不能为空且长度要求小于等于输入x的维度。<br>
    //! axis可以支持多个轴上进行处理，各元素要求小于x的维度且大于等于0
    //!
    SVector<int64_t> axis;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 依据给定的词表概率以及top-p，设置随机种子及top-k保留词数，选择最合适的词及对应概率作为输出。
//!  支持batch级别随机种子、top-k取样，支持exponential取样
//! \warning probs必须是两维张量。
//!
struct TopkToppSamplingParam {
    //! \brief 取样处理类型
    enum TopkToppSamplingType {
        SAMPLING_UNDEFINED = -1,                  //!< 未定义
        SINGLE_TOPK_SAMPLING,                     //!< 非batch级别随机种子、Topk的取样
        BATCH_TOPK_MULTINOMIAL_SAMPLING,          //!< batch级别随机种子、Topk的multinomial取样
        BATCH_TOPK_EXPONENTIAL_SAMPLING,          //!< batch级别随机种子、Topk的exponential取样
        BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING, //!< batch级别随机种子、Topk的multinomial 增加log_Probs取样
        BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING, //!< batch级别随机种子、Topk的exponential 增加log_Probs取样
        SAMPLING_MAX,                             //!< 枚举最大值
    };
    //! \brief 采样类型，默认为非batch级别随机种子、Topk的取样
    TopkToppSamplingType topkToppSamplingType = SINGLE_TOPK_SAMPLING;
    //! \brief 当 topkToppSamplingType为BATCH_TOPK_MULTINOMIAL_SAMPLING时使用
    //! \brief 每个batch下top-p阶段随机抽样使用的随机数种子。
    //! \brief 维度与batch大小一致。
    std::vector<uint32_t> randSeeds;
    //! \brief 当 topkToppSamplingType为SINGLE_TOPK_SAMPLING时使用
    //! \brief top-p阶段随机抽样使用的随机数种子。
    uint32_t randSeed = 0;
    //! \brief 当 topkToppSamplingType为SINGLE_TOPK_SAMPLING时使用
    //! \brief top-k阶段保留的词的个数,需要小于词表的词数。
    //! \brief top-k必须大于0且小于或等于输入probs最后一维的大小。
    uint32_t topk = 100;
    //!
    //! \brief logProb logprobSwitch=true时有效
    //!
    int32_t logProbsSize = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const TopkToppSamplingParam &left, const TopkToppSamplingParam &right)
{
    return left.topkToppSamplingType == right.topkToppSamplingType && left.randSeeds == right.randSeeds &&
           left.randSeed == right.randSeed && left.topk == right.topk && left.logProbsSize == right.logProbsSize;
}

//!
//! \struct PadParam
//!
//! \brief 对于输入input_ids，取出每个batch最后一个有效token的embedding向量
//!
struct PadParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct UnpadParam
//!
//! \brief 对于输入input_ids，把所有有效的token拼接在一起，并在最后补0
//!
struct UnpadParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct SortParam
//!
//! \brief 后处理计算功能。实现输入tensor在最后一维上降序排列，并保留最大的num个元素，输出排序后的tensor及各元素对应的索引。
//!
struct SortParam {
    //!
    //! \brief 排序后保留的最大的元素的数量。
    //!
    //! \warning num是一个仅含有一个值的SVector，该值需大于0且小于等于输入x最后一维的大小。
    //!
    SVector<int32_t> num;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const SortParam &left, const SortParam &right)
{
    return left.num == right.num;
}

//!
//! \struct NonzeroParam
//!
//! \brief 输出非零值索引。
//!
//! \warning 仅在Atlas 800I A2推理产品上支持
//!
struct NonzeroParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct SwiGluQuantParam
//!
//! \brief 输出非零值索引。
//!
//! \warning 仅在Atlas 800I A2推理产品上支持
//!
struct SwigluQuantParam {
    //!
    //! \enum QuantType
    //!
    //! \brief 量化支持的类型
    //!
    enum QuantType : int {
        QUANT_TYPE_PER_TOKEN = 0, //!< PER_TOKEN量化
    };

    //! \brief 量化类型。默认为QUANT_TYPE_PER_TOKEN量化。
    QuantType quantType = QUANT_TYPE_PER_TOKEN;

    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};


//!
//! \struct OnehotParam
//!
//! \brief onehot编码。
//!
struct OnehotParam {
    //! \brief depth所在下标。可为负数。
    int64_t axis = 0;
    //! \brief 类别数。
    int64_t depth = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct IndexAddParam
//!
//! \brief 固定维度的指定下标加上某个特定值。
//!
struct IndexAddParam {
    //!
    //! \enum IndexType
    //!
    //! \brief 指定下标需要执行的操作类型。
    //!
    enum IndexType {
        INDEX_UNDEFINED = 0, //!< 默认值。不支持。
        INDEX_ADD,           //!< 加
        INDEX_ADD_VALID,     //!< 有效长度内加。不支持Atlas 推理系列产品。
    };
    //!
    //! \brief 指定下标需要执行的操作类型。
    //!
    //! \note 默认值为INDEX_UNDEFINED。
    //!
    //! \warning 目前支持取值为INDEX_ADD/INDEX_ADD_VALID。
    //!
    IndexType indexType = INDEX_UNDEFINED;
    //!
    //! \brief 输入Tensor需加上updates更新值的轴。
    //!
    //! \note 默认值为0。
    //!
    //! \warning 当indexType为INDEX_ADD时，可为负数，取值范围为[-varDimNum, varDimNum - 1]。varDimNum为inTensor0的维度数。
    //!
    //! \warning 当indexType为INDEX_ADD_VALID时，仅支持取值为0。
    //!
    int64_t axis = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \struct SendParam
//!
//! \brief 将当前通信卡的输入发送至指定通信卡上,当前只支持仅Atlas 800I A2推理产品.Send和Recv需要配套使用
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize, 0 ≤ destRank < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct SendParam {
    //! \brief 当前卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信域内数据接收端的rank编号.
    uint32_t destRank = 1;
    //! \brief 通信后端指示，仅支持"hccl".
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct RecvParam
//!
//! \brief 从当前通信卡接收来自指定通信卡的数据,当前只支持仅Atlas 800I A2推理产品,Send和Recv需要配套使用
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize, 0 ≤ srcRank < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct RecvParam {
    //! \brief 当前卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信域内数据发送端的rank编号.
    uint32_t srcRank = 1;
    //! \brief 通信后端指示，仅支持"hccl".
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct AllToAllParam
//!
//! \brief 向通信域内所有通信卡发送相同数据量(输入切分成ranksize份)的数据，并从所有通信卡接收相同数据量的数据，当前只支持仅Atlas 800I A2推理产品.
//!
struct AllToAllParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //!
    //! \brief 通信计算类型。仅Atlas 800 A3推理产品支持配置为"lccl"。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //! \brief 通信结果对输入进行转置。
    //! 仅当backend为"lccl"时生效
    bool transpose = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[62] = {0};
};

//!
//! \struct AllToAllVParam
//!
//! \brief 向通信域内所有通信卡发送数据（数据量可以定制），并从所有通信卡接收数据，当前只支持仅Atlas 800I A2推理产品.
//!
struct AllToAllVParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 表示发送数据量的数组.
    //! 例如，若发送的数据类型为float32，sendCounts[i] = n 表示本rank发给rank i n个float32数据。
    std::vector<int64_t> sendCounts;
    //! \brief 表示发送偏移量的数组.
    //! sdispls[i] = n表示本rank从相对于输入起始位置的的偏移量为n的位置开始发送数据给rank i
    std::vector<int64_t> sdispls;
    //! \brief 表示接收数据量的数组.
    //! 例如，若发送的数据类型为float32，recvCounts[i] = n 表示本rank从rank i收到n个float32数据。
    std::vector<int64_t> recvCounts;
    //! \brief 表示接收偏移量的数组.
    // rdispls[i] = n表示本rank从相对于输出起始位置的的偏移量为n的位置开始接收rank i的数据
    std::vector<int64_t> rdispls;
    //!
    //! \brief 通信计算类型，仅支持"hccl".
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。
    //! 当backend为"lccl"时，commMode为多进程时，commDomain需要设置为0-65535的数字。
    //! commMode为多线程时，不支持确定性计算，"LCCL_DETERMINISTIC"需要为0或者false。
    //! LCCL在多进程/多线程多通信域并发场景下，"LCCL_PARALLEL"需要设置为1或者true。
    //! 多通信域并行功能使用结束后，"LCCL_PARALLEL"需要设置为0或者false，否则会导致基础场景性能下降。
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct AllToAllVV2Param
//!
//! \brief 向通信域内所有通信卡发送数据（数据量可以定制），并从所有通信卡接收数据，当前只支持仅Atlas 800I A2推理产品.
//!
struct AllToAllVV2Param {
    //! \brief 当前卡所属通信编号.
    int rank = -1;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //!
    //! \brief 通信计算类型，仅支持"hccl".
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    //! ranktable配置参考
    //! https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/tfmigr1/tfmigr1_000029.html
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const AllToAllVV2Param &left, const AllToAllVV2Param &right)
{
    return left.rank == right.rank && left.rankSize == right.rankSize && left.rankRoot == right.rankRoot &&
           left.hcclComm == right.hcclComm && left.commMode == right.commMode && left.backend == right.backend &&
           left.rankTableFile == right.rankTableFile && left.commDomain == right.commDomain;
}

//!
//! \struct GroupTopkParam
//!
//! \brief GroupTopk算子超参数。将输入inTensor0中维度1（inTensor0有2个维度：维度0和维度1）数据分groupNum个组，每组取最大值，然后选出每组最大值中前k个，最后将非前k个组的数据全部置零。
//!
//! \note
//!
//! \warning
//!
struct GroupTopkParam {
    //!
    //! \brief 每个token分组数量。注：“专家总数”为inTensor0Desc.shape.dims[1]的值。
    //!
    //! \note 必传，默认值为1，取值范围为[1, 专家总数]。
    //!
    //! \warning groupNum需要保证可以被inTensor0Desc.shape.dims[1]整除。
    //!
    int32_t groupNum = 1;
    //!
    //! \brief 选择top K组数量。
    //!
    //! \note 必传，默认值为0，取值范围为[1, groupNum]。
    //!
    //! \warning
    //!
    int32_t k = 0;
    //!
    //! \enum GroupMultiFlag
    //!
    //! \brief 指定GroupTopk每组中取值计算的方式。
    //!
    //! \warning
    //!
    enum GroupMultiFlag : uint16_t {
        UNDEFINED = 0, //!< 默认方式，每组内取最大值。
        SUM_MULTI_MAX  //!< 每组内取n个最大值求和，需要设置参数n
    };
    //!
    //! \brief 指定GroupTopk每组中取值计算的方式。
    //!
    //! \note 默认值为UNDEFINED。
    //!
    //! \warning 取值为SUM_MULTI_MAX时需要传入参数n。
    //!
    GroupMultiFlag groupMultiFlag = UNDEFINED;
    //!
    //! \brief 每组内取值的个数。
    //!
    //! \note 默认值为1，取值范围为[1,expert_num/groupNum]。
    //!
    //! \warning 只有当groupMultiFlag为SUM_MULTI_MAX时有效
    //!
    uint16_t n = 1;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \struct GroupedMatmulWithRoutingParam
//!
//! \brief 实现了GroupedMatmulWithRouting算子的Up和Down方法,将topK个专家权重与token激活值做矩阵乘法计算。
//!
//! \warning 仅Atlas 800I A2推理产品支持该算子
//!

struct GroupedMatmulWithRoutingParam {
    //!
    //! \enum GroupedMatmulType
    //!
    //! \brief 指定GroupedMatmulWithRouting算子需要执行的操作类型。
    //!
    enum GroupedMatmulType : int {
        GROUPED_MATMUL_UP = 0, //!< 默认值。up类型。
        GROUPED_MATMUL_DOWN    //!< down类型。
    };
    //! \brief 是否转置B矩阵（专家权重）。
    bool transposeB = true;
    //! \brief 选取的topK专家个数
    int32_t topK = 0;
    //!
    //! \brief 指定GroupedMatmulWithRouting算子需要执行的操作类型。
    //!
    //! \note 默认值为GROUPED_MATMUL_UP。
    //!
    //! \warning 目前支持取值为GROUPED_MATMUL_UP/GROUPED_MATMUL_DOWN。
    //!
    GroupedMatmulType groupedMatmulType = GROUPED_MATMUL_UP;
    //!
    //! \brief 指定输出值的反量化类型。
    //!
    //! \note 默认值为ACL_DT_UNDEFINED。
    //!
    //! \warning 非量化场景下：仅支持配置为ACL_DT_UNDEFINED。量化场景下支持ACL_FLOAT16/ACL_BF16
    //!
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \struct GroupedMatmulInplaceAddParam
//!
//! \brief 将A、B两个矩阵按照规则进行分组矩阵乘运算，并累加在矩阵C上作为输出。
//!
//! \note 算子本质上是接收x和weight两个输入tensor作为A矩阵和B矩阵进行分组矩阵乘运算并累加在矩阵C上，可通过参数transposeA与transposeB控制做矩
//! 阵乘前是否需要对A矩阵和B矩阵进行行列转置，根据参数转置后的A矩阵和B矩阵需满足矩阵乘维度关系。例如，当transposeA为false，
//! transposeB为true时，x和weight的shape可以分别为[m, k]和[n, k]。
//!
struct GroupedMatmulInplaceAddParam {
    //!
    //! \brief 是否转置A矩阵。
    //!
    //! \note 默认值为false，不转置。
    //!
    bool transposeA = false;
    //!
    //! \brief 是否转置B矩阵。
    //!
    //! \note 默认值为false，不转置，当前仅支持false。
    //!
    bool transposeB = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[22] = {0};
};

//!
//! \struct CohereLayerNormParam
//!
//! \brief CohereLayerNorm可以将网络层输入根据最后一维归一化到[0, 1]之间。
//!
//! \note 针对Command R Plus模型，对多batch数据用于表示根据最后一维进行归一化操作。
//!
struct CohereLayerNormParam {
    //!
    //! \brief epsilon,放在分母上防止除0。
    //!
    //! \note 默认值为1e-5。
    //!
    //! \warning epsilon的取值要求大于0。
    float epsilon = 1e-5;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[32] = {0};
};

//!
//! \struct GatherPreRmsNormParam
//!
//! \brief 首先对ResIn进行Gather索引操作，然后与X相加，最后进行RmsNorm计算。
//!
//! \warning 仅Atlas 800I A2推理产品支持该算子
//!
struct GatherPreRmsNormParam {
    //!
    //! \brief epsilon,放在分母上防止除0。
    //!
    //! \note 默认值为1e-5。
    //!
    //! \warning epsilon的取值要求大于0。
    float epsilon = 1e-5;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[28] = {0};
};

//!
//! \struct NormRopeReshapeParam
//!
//! \brief 融合rmsnorm、rope、reshapeAndCache。
//!
//! \warning 仅Atlas 800I A2推理产品支持该算子
//!
struct NormRopeReshapeParam {
    //! \brief precisionMode，精度模式。
    uint32_t precisionMode = 0;
    //! \brief rotaryCoeff，算子内Rope部分计算的旋转系数。
    uint32_t rotaryCoeff = 2;
    //! \brief epsilon，归一化时加在分母上防止除零。
    float epsilon = 1e-5;
    //!
    //! \brief 预留参数
    //!
    //! \note 默认值为1e-5。
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \struct FusedAddTopkDivParam
//!
//! \brief Deepseek融合算子：Sigmoid+Add+GroupTopk+Gather+ReduceSum，RealDiv，Muls。
//!
//! \note OP详细描述。
//!
//! \warning 当前仅支持Atlas 800I A2 推理产品、Atlas A2 训练系列产品和Atlas A3 训练系列产品。
//!
struct FusedAddTopkDivParam {
    //!
    //! \brief 分组数量。
    //!
    //! \note 默认值为1。
    //!
    //! \warning 取值大于0。
    //!
    uint32_t groupNum = 1;
    //!
    //! \brief 选择k个组。
    //!
    //! \note 默认值为1。
    //!
    //! \warning 取值范围为(0, groupNum]。
    //!
    uint32_t groupTopk = 1;
    //!
    //! \brief 组内选取n个最大值求和。
    //!
    //! \note 默认值为1。
    //!
    //! \warning 取值大于0。
    //!
    uint32_t n = 1;
    //!
    //! \brief topk选择前k个值。
    //!
    //! \note 默认值为1。
    //!
    //! \warning 取值大于0。
    //!
    uint32_t k = 1;
    //!
    //! \brief 激活类型。
    //!
    //! \note 默认值为ACTIVATION_SIGMOID。
    //!
    //! \warning 取值范围为ACTIVATION_SIGMOID。
    //!
    ActivationType activationType = ACTIVATION_SIGMOID;
    //!
    //! \brief 是否归一化。
    //!
    //! \note 默认值为true。
    //!
    //! \warning 取值范围为true。
    //!
    bool isNorm = true;
    //!
    //! \brief 归一化后的乘系数。
    //!
    //! \note 默认值为1.0。
    //!
    //! \warning 取值范围为任意值。
    //!
    float scale = 1.0f;
    //!
    //! \brief 是否使能物理专家向逻辑专家的映射。
    //!
    //! \note 默认值为false。
    //!
    //! \warning 取值范围为false/true。
    //!
    bool enableExpertMapping = false;
    //!
    //! \brief 预留参数。
    //!
    //! \note 默认为全0的数组。
    //!
    //! \warning 数组元素必须均为0。
    //!
    uint8_t rsv[27] = {0};
};

//!
//! \struct MlaPreprocessParam
//!
//! \brief 融合rmsNormQuant、matmul、rope、reshapeAndCache，用于MLA预处理。
//!
//! \warning 所有参数目前均为未使用的预留参数，需支持泛化后启用，仅Atlas 800I A2推理产品支持该算子
//!
struct MlaPreprocessParam {
    //!
    //! \brief 经过matmul后拆分的dim大小
    //!
    uint32_t wdqDim = 0;
    //!
    //! \brief q传入rope的dim大小
    //!
    uint32_t qRopeDim = 0;
    //!
    //! \brief k传入rope的dim大小
    //!
    uint32_t kRopeDim = 0;
    //!
    //! \brief epsilon,放在分母上防止除0。
    //!
    float epsilon = 1e-5;
    //!
    //! \brief q旋转系数，对半旋转是2，支持配置2、4或headDim。
    //!
    int32_t qRotaryCoeff = 2;
    //!
    //! \brief k旋转系数，对半旋转是2，支持配置2、4或headDim。
    //!
    int32_t kRotaryCoeff = 2;
    //!
    //! \brief wdq是否转置
    //!
    bool transposeWdq = true;
    //!
    //! \brief wuq是否转置
    //!
    bool transposeWuq = true;
    //!
    //! \brief wuk是否转置
    //!
    bool transposeWuk = true;
    //!
    //! \enum CacheMode
    //!
    //! \brief 指定cache的类型。
    //!
    enum CacheMode : uint8_t {
        KVCACHE = 0,
        KROPE_CTKV,
        INT8_NZCACHE,
        NZCACHE,
    };
    //!
    //! \brief 指定cache的类型。
    //!
    CacheMode cacheMode = KVCACHE;
    //!
    //! \enum QuantMode
    //!
    //! \brief 指定RmsNorm量化的类型。
    //!
    enum QuantMode : uint16_t {
        PER_TENSOR_QUANT_ASYMM = 0,
        PER_TOKEN_QUANT_SYMM,
        PER_TOKEN_QUANT_ASYMM,
        UNQUANT,
    };
    //!
    //! \brief 指定RmsNorm量化的类型。
    //!
    QuantMode quantMode = PER_TENSOR_QUANT_ASYMM;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[34] = {0};
};

//!
//! \struct ReshapeAndCacheOmniParam
//!
//! \brief omni压缩配套使用的reshapeAndCache
//!
//! \warning 仅Atlas 800I A2推理产品支持该算子
//!
struct ReshapeAndCacheOmniParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief MultiLatentAttention.
//!
struct MultiLatentAttentionParam {
    //!
    //! \brief query头大小
    //!
    int32_t headNum = 0;
    //!
    //! \brief 算子tor值, 在Q*K^T后乘
    //!
    float qkScale = 1.0;
    //!
    //! \brief kv头数量
    //!
    int32_t kvHeadNum = 0;
    //!
    //! \enum MaskType
    //!
    //! \brief The type values of MaskType.
    //!
    enum MaskType : int {
        UNDEFINED = 0,       //!< 默认值，全0的mask
        MASK_TYPE_SPEC,      //!< qseqlen > 1时的mask
        MASK_TYPE_MASK_FREE, //!< mask free
        MASK_TYPE_CAUSAL_MASK, //!< 内部生成mask
        MASK_TYPE_SWA_NORM
    };
    //!
    //! \brief mask类型
    //!
    MaskType maskType = UNDEFINED;
    //!
    //! \enum CalcType
    //!
    //! \brief The type values of CalcType.
    //!
    enum CalcType : int {
        CALC_TYPE_UNDEFINED = 0, // 默认值
        CALC_TYPE_SPEC,          // 支持传入大于1的qseqlen
        CALC_TYPE_RING,          // ringAttention
        CALC_TYPE_SPEC_AND_RING,          // 支持传入大于1的qseqlen ringAttention
        CALC_TYPE_PREFILL,       // 全量场景
    };
    //!
    //! \brief CalcType类型
    //!
    CalcType calcType = CALC_TYPE_UNDEFINED;
    //!
    //! \enum CacheMode
    //!
    //! \brief 指定cache的类型。
    //!
    enum CacheMode : uint8_t {
        KVCACHE = 0,  // 拼接cache
        KROPE_CTKV,   // 分离cache，默认值
        INT8_NZCACHE, // 高性能分离cache
        NZCACHE,      // 非量化NZcache
    };
    //!
    //! \brief 指定cache的类型。
    //!
    CacheMode cacheMode = KVCACHE;
    //!
    //! \note cacheMode uint8_t，占1字节后编译器会4字节补齐，此处有3字节浪费

    //!
    //! \brief 滑动窗口大小
    //!
    uint32_t windowSize = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[36] = {0};
};

//!
//! \struct RazorFusionAttentionParam
//!
//! \brief 多模态场景
//!
struct RazorFusionAttentionParam {
    //!
    //! \brief 算子headSize值, query头大小
    //!
    int32_t headNum = 1;
    //!
    //! \brief 算子kvHead值, kv头数量
    //!
    int32_t kvHeadNum = 1;
    //!
    //! \brief 算子tor值, 在Q*K^T后乘
    //!
    float qkScale = 1;
    //!
    //! \brief 图片的长度
    //!
    int32_t razorLen = 0;
    //!
    //! \brief 用于稀疏计算，表示attention需要和前几个Token计算关联，128的倍数
    //!
    int32_t preTokens = 0;
    //!
    //! \brief 用于稀疏计算，表示attention需要和前几个Token计算关联，128的倍数
    //!
    int32_t nextTokens = 0;
    //!
    //! \brief Q方向上图片的个数
    //!
    int32_t tileQ = 0;
    //!
    //! \brief Kv方向图片的个数
    //!
    int32_t tileKv = 0;
    //!
    //! \brief Q方向文本Token数量
    //!
    int32_t textQLen = 0;
    //!
    //! \brief Kv方向文本Token数量
    //!
    int32_t textKvLen = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct FaUpdateParam
//!
//! \brief 主要功能为将flash attention输出的中间结果rowmax, rowsum, attention out三个局部结果更新成全局结果
//!
struct FaUpdateParam {
    //!
    //! \enum FaUpdateType
    //!
    //! \brief 指定下标需要执行的操作类型。
    //!
    enum FaUpdateType {
        DECODE_UPDATE = 0, //!< 默认值。decode_update。
    };
    //!
    //! \brief 指定下标需要执行的操作类型。
    //!
    //! \warning 目前支持取值为DECODE_UPDATE。
    //!
    FaUpdateType faUpdateType = DECODE_UPDATE;
    //!
    //! \brief 序列并行的并行度SP。
    //!
    //! \note 默认值为1。
    //!
    uint32_t sp = 1;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct PagedCacheLoadParam
//!
//! \brief reshapeandcache反向
//!
struct PagedCacheLoadParam {
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置
    //!
    //! \note 默认值为K_CACHE_V_CACHE(0)，传入key_cache和value_cache
    //!
    enum KvCacheCfg : int8_t {
        K_CACHE_V_CACHE_NZ = 0, //!< 默认值,传入key_cache和value_cache,且为NZ格式
        K_CACHE_V_CACHE_ND,     //!< 传入key_cache和value_cache,且为ND格式
    };
    //! KvCacheCfg 配置
    KvCacheCfg kvCacheCfg = K_CACHE_V_CACHE_NZ;
    //!
    //! \brief 支持输入SeqLens为累加和模式，即第n个batch中所需提取的元素数量为前n个batch的所需提取元素数量的累加和
    //!
    bool isSeqLensCumsumMode = false;
    //!
    //! \brief 提供SeqStart用于为每个batch提供在blockTable中的初始位置（类似于offset）
    //!
    bool hasSeqStarts = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[61] = {0};
};

//!
//! \struct ScatterElementsV2Param
//!
//! \brief 将张量update中的所有的值，按照indices位置，写到自身张量input_tensor中。
//!
struct ScatterElementsV2Param {
    //!
    //! \enum ReductionType
    //!
    //! \brief ReductionType
    enum ReductionType {
        NONE = 0, //!< 默认值。none。
        ADD, //!< add 原地累加。
    };
    //!
    //! \brief 指定要收集切片的轴。默认值为0.
    //!
    //! \warning 该参数必须大于或等于0
    //!
    int32_t axis = -1;
    //!
    //! \brief  允许从一个batch的每个元素中收集不同的项目，默认值为0.
    //!
    //! \warning 该参数必须大于或等于0,且小于或等于axis.
    //!
    ReductionType reduction = NONE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[24] = {0};
};

//!
//! \struct GmmDeqSwigluQuantGmmDeqParam
//!
//! \brief GroupedMatmul1 + Dequant1 + Swiglu + Quant + GroupedMatmul2 + Dequant2 融合算子
//!
struct GmmDeqSwigluQuantGmmDeqParam {
    //!
    //! \enum OutputType
    //!
    //! \brief 指定输出类型
    //!
    enum OutputType {
        OUTPUT_FLOAT16 = 0,
        OUTPUT_BFLOAT16,
        OUTPUT_INVALID
    };

    //!
    //! \enum GroupListType
    //!
    //! \brief 指定 group list 类型
    //!
    enum GroupListType {
        GROUP_LIST_CUMSUM = 0,
        GROUP_LIST_SINGLE,
        GROUP_LIST_INVALID
    };

    //!
    //! \enum WeightUpPermuteType
    //!
    //! \brief weight1 和 scale 1 重排类型
    //!
    enum WeightUpPermuteType {
        PERMUTE_N256 = 0,
        PERMUTE_N128,
        PERMUTE_INVALID
    };

    //!
    //! \brief 输出数据类型
    //!
    OutputType outputType = OUTPUT_FLOAT16;
    //!
    //! \brief groupList 形式
    //!
    GroupListType groupListType = GROUP_LIST_CUMSUM;
    //!
    //! \brief Weight1 和 scale1 的重排方式
    //!
    WeightUpPermuteType weightUpPermuteType = PERMUTE_N256;
    //!
    //! \brief Weight1 是否转置
    //!
    bool transposeWeightUp = false;
    //!
    //! \brief Weight2 是否转置
    //!
    bool transposeWeightDown = true;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[42] = {0};
};

//!
//! \struct MmDeqSwigluQuantMmDeqParam
//!
//! \brief Matmul1 + Dequant1 + Swiglu + Quant + Matmul2 + Dequant2 融合算子
//!
struct MmDeqSwigluQuantMmDeqParam {
    //!
    //! \enum OutputType
    //!
    //! \brief 指定输出类型
    //!
    enum OutputType {
        OUTPUT_FLOAT16 = 0,
        OUTPUT_BFLOAT16,
        OUTPUT_INVALID
    };

    //!
    //! \enum WeightUpPermuteType
    //!
    //! \brief weight1 和 scale 1 重排类型
    //!
    enum WeightUpPermuteType {
        PERMUTE_N256 = 0,
        PERMUTE_N128,
        PERMUTE_INVALID
    };

    //!
    //! \brief 输出数据类型
    //!
    OutputType outputType = OUTPUT_FLOAT16;
    //!
    //! \brief Weight1 和 scale1 的重排方式
    //!
    WeightUpPermuteType weightUpPermuteType = PERMUTE_N256;
    //!
    //! \brief Weight1 是否转置
    //!
    bool transposeWeightUp = false;
    //!
    //! \brief Weight2 是否转置
    //!
    bool transposeWeightDown = true;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[46] = {0};
};

//!
//! \struct RingMLAParam
//!
//! \warning 仅Atlas 800I/T A2/A3推理产品支持该算子
//!
struct RingMLAParam {
    //!
    //! \enum CalcType
    //!
    //! \brief 计算类型
    //!
    enum CalcType : int {
        CALC_TYPE_DEFAULT = 0, // 默认，非首末卡场景，有prev_lse, prev_o传入，生成softmaxLse输出
        CALC_TYPE_FISRT_RING,  // 首卡场景，无prev_lse, prev_o传入，生成softmaxLse输出
        CALC_TYPE_MAX
    };
    //!
    //! \enum KernelType
    //!
    //! \brief 算子内核精度类型
    //!
    enum KernelType : int {
        KERNELTYPE_DEFAULT = 0,   //!< i:float16, bmm:float16, o:float16
        KERNELTYPE_HIGH_PRECISION //!< i:float16, bmm:float, o:float16
    };

    //!
    //! \enum MaskType
    //!
    //! \brief mask类型
    //!
    enum MaskType : int {
        NO_MASK = 0,    //!< 全0mask
        MASK_TYPE_TRIU, //!< 默认值，上三角mask
    };

    //! 计算类型
    CalcType calcType = CalcType::CALC_TYPE_DEFAULT;

    //! query头大小, 需大于0
    int32_t headNum = 0;
    //! kv头数量, 该值需要用户根据使用的模型实际情况传入
    //! kvHeadNum = 0时，keyCache的k_head_num，valueCache的v_head_num与query的num_heads一致，均为num_heads的数值
    //! kvHeadNum != 0时，keyCache的k_head_num， valueCache的v_head_num与kvHeadNum值相同
    int32_t kvHeadNum = 0;

    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1;

    //! 内核精度类型
    KernelType kernelType = KERNELTYPE_HIGH_PRECISION; // 预留

    //! mask类型
    MaskType maskType = MASK_TYPE_TRIU;

    //! 数据排布格式默认为BSND
    InputLayout inputLayout = TYPE_BSND;

    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};
} // namespace infer
} // namespace atb
#endif
