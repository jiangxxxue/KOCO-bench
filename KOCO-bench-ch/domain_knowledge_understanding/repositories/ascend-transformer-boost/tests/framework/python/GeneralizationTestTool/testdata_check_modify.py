#
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import logging
import os
import random
from functools import reduce
import torch
import torch_npu
import re
import numpy as np


def get_soc_version(only_910C=False):
    device_name = torch.npu.get_device_name()
    if (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name,
                                                                                           re.I):
        if not only_910C:
            soc_version = "Ascend910B"
        else:
            if re.search("Ascend910B", device_name, re.I) and len(device_name) > 10:
                soc_version = "Ascend910B"
            elif re.search("Ascend910_93", device_name, re.I):
                soc_version = "Ascend910C"
    elif re.search("Ascend310P", device_name, re.I):
        soc_version = "Ascend310P"
    elif (re.search("Ascend910ProB", device_name, re.I) or re.search("Ascend910B", device_name, re.I) or
          re.search("Ascend910PremiumA", device_name, re.I) or re.search("Ascend910ProA", device_name, re.I) or
          re.search("Ascend910A", device_name, re.I)):
        soc_version = "Ascend910A"
    else:
        logging.error("device_name {} is not supported".format(device_name))
        quit(1)
    return soc_version


class TensorDesc:
    def __init__(self):
        self.nums = 0
        self.dtypes = []
        self.formats = []
        self.shapes = []
        self.data_gen_ranges = []
        self.data_gen_types = []


class OperationValidation:
    @staticmethod
    def op_param_check_modify(op_param):
        return True

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        return True

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        return True

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype):
        return True

    @staticmethod
    def in_format_check_modify(op_param, in_format):
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        return True

    def __get_dtype_size(dtype):
        if dtype in ["float16", "int16", "uint16", "bf16"]:
            return 2
        elif dtype in ["float", "int32", "uint32"]:
            return 4
        elif dtype in ["int64", "uint64", "double", "complex64"]:
            return 8
        elif dtype in ["complex128"]:
            return 16
        else:
            return 1

    @staticmethod
    def input_data_volumn_check(op_param, tensor_desc: TensorDesc):
        MAX_SHAPE_DATA_VOLUMN = 2 ** 32 - 1
        input_data_volumn = 0
        for i in range(tensor_desc.nums):
            input_data_volumn += reduce(lambda x, y: x * y,
                                        tensor_desc.shapes[i]) * OperationValidation.__get_dtype_size(
                tensor_desc.dtypes[i])
        if input_data_volumn > MAX_SHAPE_DATA_VOLUMN:
            logging.debug("input_data_volumn {} is larger than max volumn {}, shapes: {}, dtypes {}".format(
                input_data_volumn, MAX_SHAPE_DATA_VOLUMN, tensor_desc.shapes, tensor_desc.dtypes))
            return False
        return True


class SplitOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        splitDim, splitNum = op_param['splitDim'], op_param['splitNum']
        if (splitDim + len(in_shape) < 0) or (splitDim >= len(in_shape)):
            return False
        if in_shape[splitDim] % splitNum != 0:
            return False
        return True

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_dtype))
        splitNum = op_param['splitNum']
        if get_soc_version() == 'Ascend310P' and in_dtype == 'bf16':
            return False
        if splitNum == 3 and not (in_dtype == 'float16' or in_dtype == 'bf16'):
            return False
        return True


class CumsumOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_shape: {}".format(op_param, in_shape))
        axes = op_param['axes']
        if axes[0] >= len(in_shape):
            return False
        return True


class ConcatOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        concatDim = op_param['concatDim']
        shape1, shape2 = tensor_desc.shapes[0], tensor_desc.shapes[1]
        dim_num1, dim_num2 = len(shape1), len(shape2)
        logging.debug("concatDim: %d, dim_num1: %d", concatDim, dim_num1)
        if concatDim < 0:
            concatDim = concatDim + dim_num1
        if concatDim < 0 or concatDim >= dim_num1:
            return False
        if dim_num1 != dim_num2:
            return False
        for i in range(0, dim_num1):
            if i == concatDim:
                continue
            if shape1[i] != shape2[i]:
                return False
        return True


class FillOperation(OperationValidation):
    @staticmethod
    def in_num_check_modify(op_param, in_num):
        logging.debug("op_param: %s, in_num: %d", op_param, in_num)
        withMask, outDim = op_param['withMask'], op_param['outDim']
        if withMask and in_num != 2:
            return False
        if withMask == False and in_num != 0:
            return False
        if withMask == False and len(outDim) == 0:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if tensor_desc.nums != 2:
            return True
        if tensor_desc.dtypes[0] != 'float16' and tensor_desc.dtypes[0] != 'int32':
            return False
        if tensor_desc.dtypes[1] != 'bool' and tensor_desc.dtypes[1] != 'int8':
            return False
        tensor_desc.data_gen_ranges[1] = "0,1"
        shape1, shape2 = tensor_desc.shapes[0], tensor_desc.shapes[1]
        dim_num1, dim_num2 = len(shape1), len(shape2)
        if dim_num1 < dim_num2:
            return False
        diff_num = dim_num1 - dim_num2
        for j in range(dim_num2):
            i = j + diff_num
            dim1, dim2 = shape1[i], shape2[j]
            if dim1 != dim2 and dim2 != 1:
                return False
        return True


class AsStridedOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if len(op_param['size']) != len(op_param['stride']):
            return False
        logging.debug("op_param: %s", op_param)
        return True


class MultinomialOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        numSamples = op_param['numSamples']
        if numSamples > in_shape[-1]:
            return False
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.data_gen_types[0] = 'customize'
        return True


class ReduceOperation(OperationValidation):
    soc_version = get_soc_version()

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        axis = op_param['axis']
        if len(axis) > len(in_shape):
            return False
        for aix in axis:
            if aix >= len(in_shape):
                return False
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        return True

    def in_dtype_check_modify(op_param, in_dtype, soc_version=soc_version):
        reduceType = op_param['reduceType']
        if soc_version == 'Ascend310P' and in_dtype == 'bf16':
            return False
        if reduceType == 1 and in_dtype == 'bf16':
            return False
        if reduceType == 1 and in_dtype == 'float16':
            return False
        if reduceType == 2 and in_dtype == 'bf16':
            return False
        if reduceType == 2 and in_dtype == 'float16':
            return False
        if reduceType == 3 and in_dtype == 'int32':
            return False
        return True


class RopeOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.dtypes[0] = 'float16'
        tensor_desc.dtypes[1] = 'float16'
        tensor_desc.dtypes[2] = 'float16'
        tensor_desc.dtypes[3] = 'float16'
        tensor_desc.dtypes[4] = 'int32'
        tensor_desc.data_gen_ranges[4] = "4,4"

        shapes = [tensor_desc.shapes[0], tensor_desc.shapes[1], tensor_desc.shapes[2], tensor_desc.shapes[3]]
        shapeCheck = True
        for shape in shapes:
            if len(shape) != 2:
                shapeCheck = False
        if len(tensor_desc.shapes[4]) != 1:
            shapeCheck = False
        if shapeCheck == False:
            tensor_desc.shapes[0] = [4, 4096]
            tensor_desc.shapes[1] = [4, 4096]
            tensor_desc.shapes[2] = [4, 128]
            tensor_desc.shapes[3] = [4, 128]
            tensor_desc.shapes[4] = [1]

        tensor_desc.shapes[1] = tensor_desc.shapes[0]
        tensor_desc.shapes[3] = tensor_desc.shapes[2]
        tensor_desc.shapes[3][0] = tensor_desc.shapes[0][0]

        if tensor_desc.shapes[1][1] % tensor_desc.shapes[2][1] != 0:
            tensor_desc.shapes[2][1] = tensor_desc.shapes[1][1] // 2
        if tensor_desc.shapes[0][0] % tensor_desc.shapes[4][0] != 0:
            tensor_desc.shapes[4][0] = 1

        if op_param['rotaryCoeff'] == 64:
            tensor_desc.data_gen_types[2] = 'customize'
            tensor_desc.data_gen_types[3] = 'customize'
            tensor_desc.shapes[2][1] = 64
            tensor_desc.shapes[3][1] = 64

        return True


class RopeQConcatOperation(OperationValidation):

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        ntokens = random.randint(1, 128)
        tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
        tensor_desc.dtypes[2] = tensor_desc.dtypes[0]
        tensor_desc.dtypes[3] = tensor_desc.dtypes[0]
        head_dim = random.choice([16, 32, 48, 64])
        head_num = random.choice([16, 32, 64, 128, 256, 320, 512, 768, 1024, 5120])
        concat_size = random.choice([16, 32, 64, 128, 256, 320, 512, 768, 1024, 5120])
        hidden_size_q = head_dim * head_num
        tensor_desc.shapes[0] = [ntokens, hidden_size_q]
        tensor_desc.shapes[1] = [ntokens, head_dim]
        tensor_desc.shapes[2] = tensor_desc.shapes[1]
        tensor_desc.shapes[3] = [ntokens, head_num, concat_size]
        return True

class SliceOperation(OperationValidation):
    soc_version = get_soc_version()

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype, soc_version=soc_version):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_dtype))
        if soc_version == 'Ascend310P' and in_dtype == 'bf16':
            return False
        return True

    def op_param_check_modify(op_param):
        if len(op_param['offsets']) != len(op_param['size']):
            return False
        for i, offset in enumerate(op_param['offsets']):
            size = op_param['size'][i]
            if size != -1 and offset > -size:
                return False
        logging.debug("op_param: %s", op_param)
        return True

    def in_shape_check_modify(op_param, in_shape):
        offsets = op_param['offsets']
        sizeList = op_param['size']
        if len(offsets) != len(in_shape):
            return False
        for i, offset in enumerate(offsets):
            size = sizeList[i]
            dim = in_shape[i]
            if size == -1:
                size = dim
            if offset < 0:
                offset = dim
            if offset + size > dim:
                in_shape[i] = offset + size
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        return True


class OnehotOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if tensor_desc.dtypes[0] != tensor_desc.dtypes[1] or tensor_desc.dtypes[0] != tensor_desc.dtypes[2] or \
                tensor_desc.dtypes[1] != tensor_desc.dtypes[2]:
            tensor_desc.dtypes[0] = 'int64'
            tensor_desc.dtypes[1] = 'int64'
            tensor_desc.dtypes[2] = 'int64'
        tensor_desc.shapes[1] = [1]
        tensor_desc.shapes[2] = [1]
        tensor_desc.data_gen_types[0] = "random"
        tensor_desc.data_gen_types[1] = "one"
        tensor_desc.data_gen_types[2] = "zero"
        logging.debug("op_param: %s", op_param)
        return True

    def in_shape_check_modify(op_param, in_shape):
        axis = op_param['axis']
        axis = axis if axis >= 0 else len(in_shape) + 1 + axis
        if axis >= len(in_shape) + 1:
            return False

        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        return True


class NonzeroOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        return True


class IndexAddOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        axis, indexType = op_param['axis'], op_param['indexType']
        if indexType == 2:
            if axis != 0:
                return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        axis, indexType = op_param['axis'], op_param['indexType']
        tensor_desc.shapes[3] = [1]
        tensor_desc.data_gen_types = ["random", "customize", "random", "random"]
        if indexType == 1:
            tensor_desc.dtypes[0] = 'float16'
            tensor_desc.dtypes[1] = 'int32'
            tensor_desc.dtypes[2] = 'float16'
            tensor_desc.dtypes[3] = 'float16'
            # tensor_desc.shapes[3] = [1]
            # tensor_desc.data_gen_types = ["random", "customize", "random", "random"]
            if axis >= len(tensor_desc.shapes[0]) or axis < -len(tensor_desc.shapes[0]):
                return False
            if len(tensor_desc.shapes[1]) != 1:
                return False
            if len(tensor_desc.shapes[0]) != len(tensor_desc.shapes[2]):
                return False

            dim = tensor_desc.shapes[0][axis]
            if tensor_desc.shapes[1][0] > dim:
                return False

            for i in range(len(tensor_desc.shapes[2])):
                if i == axis:
                    tensor_desc.shapes[2][i] = tensor_desc.shapes[1][0]
                else:
                    tensor_desc.shapes[2][i] = tensor_desc.shapes[0][i]
        if indexType == 2:
            if len(tensor_desc.shapes[0]) != len(tensor_desc.shapes[2]) or len(tensor_desc.shapes[0]) != 2:
                return False
            if len(tensor_desc.shapes[1]) != 1:
                return False
            if tensor_desc.dtypes[1] != 'int32':
                return False
            tensor_desc.dtypes[0] = tensor_desc.dtypes[2] = random.choice(["float16","bf16"])
            if tensor_desc.dtypes[1] != tensor_desc.dtypes[3]:
                return False
            tensor_desc.shapes[0][1] = tensor_desc.shapes[2][1]
            tensor_desc.shapes[1][0] = tensor_desc.shapes[2][0]
            tensor_desc.data_gen_ranges[1] = "0,0"
            tensor_desc.data_gen_ranges[3] = "0,0"
        return True


class ReshapeAndCacheOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.dtypes[0] = 'float16'
        tensor_desc.dtypes[1] = 'float16'
        tensor_desc.dtypes[2] = 'float16'
        tensor_desc.dtypes[3] = 'float16'
        tensor_desc.dtypes[4] = 'int32'

        if len(tensor_desc.shapes[0]) != 3 or len(tensor_desc.shapes[1]) != 3:
            tensor_desc.shapes[0] = [2, 32, 128]
            tensor_desc.shapes[1] = [2, 32, 128]
        if len(tensor_desc.shapes[2]) != 4 or len(tensor_desc.shapes[3]) != 4:
            tensor_desc.shapes[2] = [64, 16, 32, 128]
            tensor_desc.shapes[3] = [64, 16, 32, 128]
        if len(tensor_desc.shapes[4]) != 1:
            tensor_desc.shapes[4] = [2]
        tensor_desc.shapes[1][0] = tensor_desc.shapes[0][0]
        tensor_desc.shapes[1][1] = tensor_desc.shapes[0][1]
        tensor_desc.shapes[4][0] = tensor_desc.shapes[0][0]
        tensor_desc.shapes[2][2] = tensor_desc.shapes[0][1]
        tensor_desc.shapes[3][2] = tensor_desc.shapes[1][1]
        tensor_desc.shapes[1][2] = tensor_desc.shapes[0][2]
        tensor_desc.shapes[2][3] = tensor_desc.shapes[0][2]
        tensor_desc.shapes[3][3] = tensor_desc.shapes[0][2]
        tensor_desc.shapes[3][0] = tensor_desc.shapes[2][0]
        tensor_desc.shapes[3][1] = tensor_desc.shapes[2][1]

        if get_soc_version() == 'Ascend310P':
            tensor_desc.formats = ["nd", "nd", "fractal_nz", "fractal_nz", "nd"]
            if (tensor_desc.shapes[0][1] * tensor_desc.shapes[0][2]) < 16:
                return False
            else:
                tensor_desc.shapes[2][1] = (tensor_desc.shapes[0][1] * tensor_desc.shapes[0][2]) // 16
                tensor_desc.shapes[3][1] = (tensor_desc.shapes[1][1] * tensor_desc.shapes[1][2]) // 16
                tensor_desc.shapes[2][3] = tensor_desc.shapes[3][3] = 16
            if tensor_desc.shapes[0][0] > (tensor_desc.shapes[2][0] * tensor_desc.shapes[2][2]):
                return False
            if (tensor_desc.shapes[2][2] % 16 !=0) or (tensor_desc.shapes[3][2] % 16 !=0):
                return False
        else:
            tensor_desc.formats = ["nd", "nd", "nd", "nd", "nd"]
            if tensor_desc.shapes[0][0] > (tensor_desc.shapes[2][0] * tensor_desc.shapes[2][1]):
                return False
        return True


class TransposeOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_shape: {}".format(op_param, in_shape))
        perm = op_param['perm']
        if len(perm) != len(in_shape):
            return False
        return True


class KvCacheOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if get_soc_version() == 'Ascend910B':
            tensor_desc.dtypes[0] = 'float16'
            tensor_desc.dtypes[1] = 'int32'
            tensor_desc.dtypes[2] = 'float16'
            tensor_desc.dtypes[3] = 'int32'
            tensor_desc.dtypes[4] = 'int32'

            layer = random.choice([28, 35, 80])
            batch = random.choice([2, 4, 8])
            max_seqlen = 384
            hidden_size = 1024

            seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
            token_offset = seqlen
            ntokens = np.sum(seqlen)

            tensor_desc.shapes[0] = [ntokens, hidden_size]
            tensor_desc.shapes[1] = [1]
            tensor_desc.shapes[2] = [layer, batch, max_seqlen, hidden_size]
            tensor_desc.shapes[3] = [batch]
            tensor_desc.shapes[4] = [batch]
        else:
            tensor_desc.dtypes[0] = 'float16'
            tensor_desc.dtypes[1] = 'int32'
            tensor_desc.dtypes[2] = 'float16'
            tensor_desc.dtypes[3] = 'int32'
            tensor_desc.dtypes[4] = 'int32'

            tensor_desc.formats = ["fractal_nz", "nd", "fractal_nz", "nd", "nd"]

            layer = random.choice([28, 35, 80])
            batch = random.choice([2, 4, 8])
            max_seqlen = 384
            hidden_size = 1024

            seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
            token_offset = seqlen
            ntokens = np.sum(seqlen)

            tensor_desc.shapes[0] = [ntokens, hidden_size]
            tensor_desc.shapes[1] = [1]
            tensor_desc.shapes[2] = [layer, batch, max_seqlen, hidden_size]
            tensor_desc.shapes[3] = [batch]
            tensor_desc.shapes[4] = [batch]
        return True


class DemoOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.

        Returns:
            Bool: checking results.
        """
        return True

    def in_num_check_modify(op_param, in_num):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_num (Int): 'InNum' column in csv file.

        Returns:
            Bool: check result.
        """
        return True

    def in_shape_check_modify(op_param, in_shape):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_shape (List[int]): 'InShape' column in csv file.

        Returns:
            Bool: checking results.
        """
        return True

    def in_dtype_check_modify(op_param, in_dtype):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_dtype (List[int]): 'InDType' column in csv file.

        Returns:
            Bool: checking results.
        """
        return True

    def in_format_check_modify(op_param, in_format):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_format (List[int]): 'InFormat' column in csv file.

        Returns:
            Bool: checking results.
        """
        return True


class GatherOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param) -> bool:
        batchDims = op_param["batchDims"]
        axis = op_param['axis']
        if batchDims > axis:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc) -> bool:
        batchDims = op_param["batchDims"]
        axis = op_param['axis']
        if axis == batchDims and len(tensor_desc.shapes[0]) != len(tensor_desc.shapes[1]):
            return False
        if axis == batchDims:
            for i in range(len(tensor_desc.shapes[0]) - 1):
                tensor_desc.shapes[1][i] = tensor_desc.shapes[0][i]
        if axis >= len(tensor_desc.shapes[0]):
            return False
        tensor_desc.dtypes[0] = 'float16'
        tensor_desc.dtypes[1] = 'int64'
        tensor_desc.data_gen_ranges[1] = "0," + str(tensor_desc.shapes[0][axis] - 1)
        return True


class SetValueOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        ends = op_param['ends']
        starts = op_param['starts']
        strides = op_param['strides']
        if len(strides) != len(starts) or len(strides) != len(ends):
            return False
        return True

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        ends = op_param['ends']
        if len(in_shape) != len(ends):
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        ends = op_param['ends']
        starts = op_param['starts']
        strides = op_param['strides']
        tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
        dstDimNum = len(tensor_desc.shapes[0])
        for i in range(dstDimNum):
            tensor_desc.shapes[1][i] = ends[i] - starts[i]
        for i in range(dstDimNum):
            if tensor_desc.shapes[0][i] < tensor_desc.shapes[1][i]:
                return False
        count = 0
        for i in range(1, dstDimNum):
            if tensor_desc.shapes[0][i] != tensor_desc.shapes[1][i]:
                count += 1
                if count > 1:
                    return False
        if count == 0:
            return False
        return True


class UnpadOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.dtypes = ["int64", "int32", "int64", "int32"]
        tensor_desc.shapes[1][0] = tensor_desc.shapes[0][0]
        tensor_desc.shapes[3][0] = tensor_desc.shapes[0][0]
        return True


class PadOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.dtypes = ["float16", "int32", "int32", "int64"]
        tensor_desc.shapes[0][1] = 1600
        if tensor_desc.shapes[2][0] != tensor_desc.shapes[3][0]:
            return False
        if tensor_desc.shapes[3][1] >= tensor_desc.shapes[0][1]:
            return False
        tensor_desc.shapes[3][1] = 64
        tensor_desc.shapes[1][0] = 1
        tensor_desc.shapes[2][1] = 1
        return True


class AllGatherOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        backend = op_param['backend']
        soc_version = get_soc_version()
        ranksize = op_param['rankSize']
        rankRoot = op_param['rankRoot']
        if soc_version == 'Ascend310P' and backend == 'lccl':
            return False
        if ranksize <= rankRoot:
            return False
        return True

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype):
        soc_version = get_soc_version()
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_dtype))
        if soc_version == 'Ascend310P' and in_dtype == 'bf16':
            return False
        return True


class AllReduceOperation(OperationValidation):

    @staticmethod
    def op_param_check_modify(op_param):
        soc_version = get_soc_version()
        ranksize = op_param['rankSize']
        rankRoot = op_param['rankRoot']
        backend = op_param['backend']
        allReduceType = op_param['allReduceType']
        if backend == "lccl" and allReduceType == "prod":
            return False
        if ranksize <= rankRoot:
            return False
        if soc_version == 'Ascend310P' and op_param['backend'] == 'lccl':
            return False
        return True

    def in_dtype_check_modify(op_param, in_dtype):
        soc_version = get_soc_version()
        backend = op_param['backend']
        allReduceType = op_param['allReduceType']
        if backend == "lccl" and in_dtype == 'int64':
            return False
        if backend == "hccl" and allReduceType == "prod":
            if in_dtype in ["int16", "bf16"]:
                return False
        if backend == "hccl" and in_dtype == "int16" and soc_version == 'Ascend310P':
            if allReduceType in ["prod", "max", "min"]:
                return False
        if soc_version == 'Ascend310P' and in_dtype == 'bf16':
            return False
        if soc_version == 'Ascend310P' and in_dtype == 'int64':
            return False
        return True


class ReduceScatterOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        return True

    def in_shape_check_modify(op_param, in_shape):
        ranksize = op_param['rankSize']
        if in_shape[0] % ranksize != 0:
            in_shape[0] = in_shape[0] + 1
        return True


class DynamicNTKOperation(OperationValidation):

    @staticmethod
    def op_param_check_modify(op_param):
        if get_soc_version() == "Ascend310P":
            op_param["outputType"] = 1
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.dtypes[0] = 'int32'
        tensor_desc.dtypes[1] = 'float'
        tensor_desc.dtypes[2] = 'int32'
        batch = random.choice([x for x in range(1, 17)])
        ntokens = random.choice([x for x in range(1, 25601)])
        headDim = random.choice([x for x in range(1, 2049) if x % 32 == 0])
        tensor_desc.shapes[0] = [ntokens]
        tensor_desc.shapes[1] = [batch, int(headDim / 2)]
        tensor_desc.shapes[2] = [batch]
        return True


class BroadcastOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        ranksize = op_param['rankSize']
        rankRoot = op_param['rankRoot']
        if ranksize <= rankRoot:
            return False
        return True


class ElewiseOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        elewiseType = op_param["elewiseType"]
        input_scale = op_param["inputScale"]
        asymmetric = op_param["asymmetric"]
        inputOffset = op_param["inputOffset"]
        outType = op_param["outTensorType"]
        soc_version = get_soc_version()
        op_param["mulsParam"] = {"varAttr": op_param["varAttr"]}
        if elewiseType in [17, 18, 19]:
            op_param["quantParam"] = {"inputScale": input_scale}
            op_param["quantParam"]["asymmetric"] = asymmetric
            op_param["quantParam"]["inputOffset"] = inputOffset
        else:
            if "quantParam" in op_param:
                del op_param["quantParam"]
        if soc_version == "Ascend310P":
            if elewiseType in [6, 18]:
                return False
            # 310P平台ELEWISE_CAST不支持outType取3：INT32和9：INT64
            if elewiseType in [1] and outType in [3, 9, 27]:
                return False
        return True

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_num (Int): 'InNum' column in csv file.

        Returns:
            Bool: check result.
        """
        elewiseType = op_param["elewiseType"]
        if in_num == 1:
            if elewiseType not in [1, 2, 3, 4, 5, 6, 7, 19, 20]:
                return False
        elif in_num == 2:
            if elewiseType in [1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20]:
                return False
        elif in_num == 3:
            if elewiseType not in [17, 18]:
                return False
        return True

    def in_dtype_check_modify(op_param, in_dtype):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_dtype (List[int]): 'InDType' column in csv file.

        Returns:
            Bool: checking results.
        """
        elewiseType = op_param["elewiseType"]
        soc_version = get_soc_version()
        if elewiseType in [2, 3, 4, 8, 16]:
            return in_dtype in ['float', 'float16']
        if elewiseType in [10]:
            if soc_version == "Ascend310P":
                return in_dtype in ['float', 'float16']
            else:
                return in_dtype in ['float', 'float16', 'bf16']
        if elewiseType in [13, 14]:
            return in_dtype in ['int64', 'float16', 'float']
        if elewiseType in [15]:
            return in_dtype in ['float16', 'int64']
        if elewiseType in [17]:
            return in_dtype in ['float16', 'int8']
        if elewiseType in [18]:
            return in_dtype in ['float16', 'int8']
        if elewiseType in [19, 20]:
            return in_dtype in ['float16']
        return True

    def in_shape_check_modify(op_param, in_shape):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_shape (List[int]): 'InShape' column in csv file.

        Returns:
            Bool: checking results.
        """
        elewiseType = op_param["elewiseType"]
        soc_version = get_soc_version()
        if elewiseType == 6:
            if len(in_shape) != 2:
                return False
        elif elewiseType == 6 and soc_version == "Ascend310P":
            return False
        elif elewiseType == 17 or elewiseType == 18:
            if len(in_shape) == 2:
                in_shape[1] = random.randint(1, 100) * 32
            elif len(in_shape) == 3:
                in_shape[1] = random.randint(1, 100) * 32
                in_shape[2] = random.randint(1, 100) * 32
        elif elewiseType == 19:
            if random.random() < 0.1:
                in_shape.extend([random.randint(1, 4096)])
            elif random.random() < 0.1:
                in_shape.extend([random.randint(1, 30),
                                 random.randint(1, 4096)])
            elif random.random() < 0.15:
                in_shape.extend([random.randint(1, 30),
                                 random.randint(1, 30),
                                 random.randint(1, 4096)])
            elif random.random() < 0.2:
                in_shape.extend([random.randint(1, 30),
                                 random.randint(1, 30),
                                 random.randint(1, 30),
                                 random.randint(1, 4096)])
            elif random.random() < 0.8:
                in_shape.extend([random.randint(1, 30),
                                 random.randint(1, 30),
                                 random.randint(1, 30),
                                 random.randint(1, 30),
                                 random.randint(1, 4096)])

            if soc_version == "Ascend310P":
                in_shape[-1] = ((in_shape[-1] + 31) // 32) * 32
        return True

    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        elewiseType = op_param["elewiseType"]
        if elewiseType in [5, 6, 9]:
            tensor_desc.dtypes[0] = 'float16'
        if elewiseType in [7]:
            tensor_desc.dtypes[0] = 'int8'
            tensor_desc.data_gen_ranges = ["-1, 1"]
        if elewiseType in [6]:
            tensor_desc.data_gen_ranges = ["-1, 1"]
        if elewiseType in [10]:
            tensor_desc.data_gen_ranges = ["-7, 7", "1, 2"]
        if elewiseType in [19]:
            tensor_desc.dtypes = ['float16']
            tensor_desc.data_gen_ranges = ["-5, 10"]
        if elewiseType in [11, 12]:
            tensor_desc.dtypes = ['int8', 'int8']
            tensor_desc.data_gen_ranges = ["-1, 1", "-1, 1"]
        if elewiseType in [1]:
            soc_version = get_soc_version()
            outType = op_param["outTensorType"]
            if outType == 0:
                if soc_version == "Ascend310P":
                    tensor_desc.dtypes[0] = "float16"
                else:
                    tensor_desc.dtypes[0] = random.choice(["float16", "bf16"])
            if outType == 1:
                tensor_desc.dtypes[0] = 'float'
            if outType == 3:
                tensor_desc.dtypes[0] = 'int64'
            if outType == 9:
                tensor_desc.dtypes[0] = "int32"
            if outType == 27:
                tensor_desc.dtypes[0] = "float"
        if elewiseType not in [1, 2, 3, 4, 5, 6, 7, 19, 20]:
            tensor_desc.shapes[1] = tensor_desc.shapes[0]
            tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
        if elewiseType in [17]:
            tensor_desc.dtypes = ['float16', 'float16', 'int8']
            tensor_desc.data_gen_ranges = ["-1000, 1000", "0.1, 100", "-10, 10"]
            if len(tensor_desc.shapes[0]) == 2:
                tensor_desc.shapes[1] = tensor_desc.shapes[0][1:]
                tensor_desc.shapes[2] = tensor_desc.shapes[0][1:]
            elif len(tensor_desc.shapes[0]) == 3:
                tensor_desc.shapes[1] = tensor_desc.shapes[0][1:]
                tensor_desc.shapes[2] = tensor_desc.shapes[0][1:]
        if elewiseType in [18]:
            tensor_desc.dtypes = ['int8', 'float16', 'int8']
            tensor_desc.data_gen_ranges = ["-128, 127", "0, 1", "-128, 127"]
            if len(tensor_desc.shapes[0]) == 2:
                tensor_desc.shapes[1] = tensor_desc.shapes[0][1:]
                tensor_desc.shapes[2] = tensor_desc.shapes[0][1:]
            elif len(tensor_desc.shapes[0]) == 3:
                tensor_desc.shapes[1] = tensor_desc.shapes[0][1:]
                tensor_desc.shapes[2] = tensor_desc.shapes[0][1:]
        return True


import random


class SelfAttentionOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.

        Returns:
            Bool: checking results.
        """
        soc_version = get_soc_version()
        op_param["kernelType"] = 0
        if soc_version == "Ascend310P":
            if op_param["calcType"] != 3:
                return False
        if op_param["calcType"] == 3:
            op_param["maskType"] = 0
            op_param["kernelType"] = 0
            if op_param["headNum"] != 32:
                return False
            if op_param["kvHeadNum"] != 32:
                return False
            if op_param["qkScale"] != 0.08838834764831843:
                return False
            if op_param["clampType"] == 1:
                return False
        else:
            op_param["maskType"] = 1
            if op_param["kvHeadNum"] != 0:
                return False
            if op_param["clampType"] == 1:
                if op_param["qkScale"] != 1:
                    return False
        return True

    def in_num_check_modify(op_param, in_num):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.
            in_num (Int): 'InNum' column in csv file.

        Returns:
            Bool: check result.
        """
        soc_version = get_soc_version()
        if op_param["calcType"] == 3:
            return in_num == 4
        else:
            return in_num == 9
        return True

    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        soc_version = get_soc_version()
        if op_param["calcType"] == 3:
            tensor_desc.nums = 4
            batch = random.choice([128, 64, 32])
            tensor_desc.dtypes = ["float16", "float16", "float16", "int32"]
            tensor_desc.shapes = [[32, 128], [32, 128], [32, 128], [batch]]
        elif op_param["clampType"] == 1:
            tensor_desc.nums = 9
            max_seq_len = random.choice([1024, 2048])
            layer = random.choice([8, 28, 80])
            tensor_desc.dtypes = ["float16", "float16", "float16", "float16", "float16", "float16", "int32", "int32",
                                  "int32"]
            tensor_desc.shapes = [[4, 3, 32, 128], [4, 3, 32, 128], [4, 3, 32, 128], [layer, 3, max_seq_len, 4096],
                                  [layer, 3, max_seq_len, 4096], [max_seq_len, max_seq_len], [3], [3], [1]]
        else:
            max_seq_len = random.choice([1024, 2048])
            layer = random.choice([8, 28, 80])
            tensor_desc.dtypes = ["float16", "float16", "float16", "float16", "float16", "float16", "int32", "int32",
                                  "int32"]
            tensor_desc.shapes = [[4, 3, 32, 128], [4, 3, 32, 128], [4, 3, 32, 128], [layer, 3, max_seq_len, 4096],
                                  [layer, 3, max_seq_len, 4096], [max_seq_len, max_seq_len], [3], [3], [1]]

        return True


class PagedAttentionOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.nums = 5
        tensor_desc.dtypes = ["float16", "float16", "float16", "float16", "int32"]
        tensor_desc.shapes = [[4, 32, 32], [4, 256, 32, 32], [4, 256, 32, 32], [4, 32], [4]]
        tensor_desc.shapes[0] = [4, 32, 32]  # query
        num_blocks = random.choice([1, 2, 3, 4])
        tensor_desc.shapes[1] = [num_blocks, 256, 32, 32]  # key cache
        tensor_desc.shapes[2] = [num_blocks, 256, 32, 32]  # value cache
        tensor_desc.shapes[3] = [4, 32]  # blocktables
        tensor_desc.shapes[4] = [4]  # num tokens

        return True


class ActivationOperation(OperationValidation):
    @staticmethod
    def in_num_check_modify(op_param, in_num):
        soc_version = get_soc_version()
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_num))
        activationType = op_param['activationType']
        if activationType < 7 and in_num != 1:
            return False
        if activationType == 7 and in_num != 2:
            return False
        if activationType == 8 and in_num != 1:
            return False
        if activationType == 7 and soc_version == 'Ascend310P':
            return False
        return True

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype):
        soc_version = get_soc_version()
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_dtype))
        activationType = op_param['activationType']
        if activationType == 1 and in_dtype != 'float':
            return False
        if activationType == 3 and in_dtype != 'float16':
            return False
        if activationType == 4 and in_dtype != 'float16':
            return False
        if activationType == 5 and in_dtype == 'bf16':
            return False
        if activationType == 8 and in_dtype == 'float':
            return False
        if soc_version == 'Ascend310P' and in_dtype == 'bf16':
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        activationType = op_param['activationType']
        split_dim = -1
        # 6: ACTIVATION_SWIGLU_FORWARD, 7: ACTIVATION_SWIGLU_BACKWARD
        if activationType in (6, 7):
            split_dim = op_param['dim']
        x_dim = random.choice([2048, 3072, 1376, 3904, 4096])
        if activationType < 5 and tensor_desc.data_gen_ranges[0] == "-2,2":
            return False
        if activationType == 5 and tensor_desc.data_gen_ranges[0] != "0.1,100":
            return False
        if activationType == 6:
            if tensor_desc.data_gen_ranges[0] != "-2,2":
                return False
            tensor_desc.shapes[0] = [4096, 4, 2048]
            tensor_desc.shapes[0][split_dim] = x_dim * 2
        if activationType == 7:
            if tensor_desc.data_gen_ranges[0] != "-2,2" and tensor_desc.data_gen_ranges[1] != "-2,2":
                return False
            tensor_desc.nums = 2
            tensor_desc.shapes[0] = [4096, 4, 2048]
            tensor_desc.shapes[1] = [4096, 4, 4096]
            tensor_desc.shapes[0][split_dim] = x_dim
            tensor_desc.shapes[1][split_dim] = x_dim * 2
            tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
        return True


class RepeatOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        if len(op_param['multiples']) < len(in_shape):
            return False
        return True


class WhereOperation(OperationValidation):
    @staticmethod
    def broadcast_dim(x_shape, y_shape):
        x_len, y_len = len(x_shape), len(y_shape)
        result_dim = []
        while x_len > 0 and y_len > 0:
            size1, size2 = x_shape[x_len - 1], y_shape[y_len - 1]
            if size1 != size2 and size1 != 1 and size2 != 1:
                return None
            result_dim.insert(0, max(size1, size2))
            x_len -= 1
            y_len -= 1
        while x_len > 0:
            result_dim.insert(0, x_shape[x_len - 1])
            x_len -= 1
        while y_len > 0:
            result_dim.insert(0, y_shape[y_len - 1])
            y_len -= 1
        return tuple(result_dim)

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        cond_shape = tensor_desc.shapes[0]
        x_shape = tensor_desc.shapes[1]
        y_shape = tensor_desc.shapes[2]
        cond_shape_len = min(len(x_shape), len(y_shape))
        tensor_desc.dtypes[0] = "int8"
        tensor_desc.dtypes[1] = "float16"
        tensor_desc.dtypes[2] = "float16"
        if WhereOperation.broadcast_dim(x_shape, y_shape) is None:
            return False
        times = 0
        while WhereOperation.broadcast_dim(cond_shape, x_shape) != tuple(x_shape) or \
                WhereOperation.broadcast_dim(cond_shape, y_shape) != tuple(y_shape):
            times += 1
            cond_shape = [random.randint(1, 16) for _ in range(cond_shape_len)] if \
                times < 10000 else [1 for _ in range(cond_shape_len)]
        tensor_desc.shapes[0] = cond_shape
        return True


class TopkToppSamplingOperation(OperationValidation):

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_num))
        topkToppSamplingType = op_param['topkToppSamplingType']
        if topkToppSamplingType == 0 and in_num != 2:
            return False
        if topkToppSamplingType == 1 and in_num != 3:
            return False
        if topkToppSamplingType == 2 and in_num != 4:
            return False
        if topkToppSamplingType == 3 and in_num != 4:
            return False
        if topkToppSamplingType == 4 and in_num != 4:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        topkToppSamplingType = op_param['topkToppSamplingType']
        tensor_desc.shapes[1][0] = tensor_desc.shapes[0][0]
        if tensor_desc.shapes[0][0] > 7 or tensor_desc.shapes[0][1] < 100:
            return False
        if topkToppSamplingType == 1:
            if len(tensor_desc.dtypes) != 3:
                return False
            if tensor_desc.shapes[1][0] != tensor_desc.shapes[2][0]:
                return False
            if tensor_desc.shapes[0][0] != len(op_param['randSeeds']):
                return False
            tensor_desc.dtypes[0] = "float16"
            tensor_desc.dtypes[1] = "int32"
            tensor_desc.dtypes[2] = "float16"
            tensor_desc.data_gen_types = ["customize", "random", "random"]
            tensor_desc.data_gen_ranges = ["-5,5", "10,20", "0,1"]
            tensor_desc.shapes[1][1] = 1
            tensor_desc.shapes[2][1] = 1
            tensor_desc.shapes[2][0] = tensor_desc.shapes[0][0]
        if topkToppSamplingType == 2:
            if len(tensor_desc.dtypes) != 4:
                return False
            tensor_desc.dtypes[0] = "float16"
            tensor_desc.dtypes[1] = "int32"
            tensor_desc.dtypes[2] = "float16"
            tensor_desc.dtypes[3] = "float16"
            tensor_desc.data_gen_types = ["customize", "random", "random", "customize"]
            tensor_desc.data_gen_ranges = ["-5,5", "10,20", "0,1", "-5,5"]
            tensor_desc.shapes[1][1] = 1
            tensor_desc.shapes[2][1] = 1
            tensor_desc.shapes[2][0] = tensor_desc.shapes[0][0]
            tensor_desc.shapes[3][1] = tensor_desc.shapes[0][1]
            tensor_desc.shapes[3][0] = tensor_desc.shapes[0][0]

        if topkToppSamplingType == 3:
            if len(tensor_desc.dtypes) != 4:
                return False
            if op_param['logProbsSize'] > 16384 or op_param['logProbsSize'] < 0:
                return False
            tensor_desc.dtypes[0] = "float16"
            tensor_desc.dtypes[1] = "int32"
            tensor_desc.dtypes[2] = "float16"
            tensor_desc.dtypes[3] = "float"
            tensor_desc.data_gen_types = ["customize", "random", "random", "random"]
            tensor_desc.data_gen_ranges = ["-5,5", "10,20", "0,1", "0,1"]
            tensor_desc.shapes[1][1] = 1
            tensor_desc.shapes[2][1] = 1
            tensor_desc.shapes[2][0] = tensor_desc.shapes[0][0]
            tensor_desc.shapes[3][0] = tensor_desc.shapes[0][0]
            tensor_desc.shapes[3][1] = 1

        if topkToppSamplingType == 4:
            if len(tensor_desc.dtypes) != 4:
                return False
            if op_param['logProbsSize'] > 16384 or op_param['logProbsSize'] < 0:
                return False
            tensor_desc.dtypes[0] = "float16"
            tensor_desc.dtypes[1] = "int32"
            tensor_desc.dtypes[2] = "float16"
            tensor_desc.dtypes[3] = "float16"
            tensor_desc.data_gen_types = ["customize", "random", "random", "customize"]
            tensor_desc.data_gen_ranges = ["-5,5", "10,20", "0,1", "-5,5"]
            tensor_desc.shapes[1][1] = 1
            tensor_desc.shapes[2][1] = 1
            tensor_desc.shapes[2][0] = tensor_desc.shapes[0][0]
            tensor_desc.shapes[3][1] = tensor_desc.shapes[0][1]
            tensor_desc.shapes[3][0] = tensor_desc.shapes[0][0]

        if topkToppSamplingType == 0:
            if tensor_desc.shapes[0][1] < op_param['topk']:
                return False
            if len(tensor_desc.dtypes) != 2:
                return False
            tensor_desc.dtypes[0] = "float16"
            tensor_desc.dtypes[1] = "float16"
            tensor_desc.data_gen_types = ["customize", "random"]
            tensor_desc.data_gen_ranges = ["-5,5", "0,1"]
            tensor_desc.shapes[1][1] = 1
        return True


class TransdataOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        if op_param['transdataType'] == 1 and len(in_shape) != 4:
            return False
        if op_param['transdataType'] == 2 and len(in_shape) == 4:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if op_param['transdataType'] == 1:
            tensor_desc.formats[0] = "fractal_nz"
            tensor_desc.dtypes[0] = "float16"
            tensor_desc.shapes[0][1] = 2
            tensor_desc.shapes[0][2] = 32
            tensor_desc.shapes[0][3] = 16
            op_param['outCrops'][0] = random.randint(17, 32)
            op_param['outCrops'][1] = random.randint(17, 32)
        else:
            tensor_desc.formats[0] = "nd"
            op_param['outCrops'] = [0, 0]
        return True


class MatmulCommon(OperationValidation):
    soc_version = get_soc_version()
    param = None

    class Op:
        LINEAR = 0
        LINEAR_PARALLEL = 1
        LINEAR_SPARSE = 2
        MATMUL = 3

    class LinearType:
        FP16FP16_FP32_FP16 = 0
        BF16BF16_FP32_BF16 = 1
        INT8INT8_INT32_FP16 = 2
        INT8INT8_INT32_BF16 = 3

    class Param:
        def __init__(self, op_param: dict, op: int):
            self.op = op
            self.transpose_a: bool = op_param.get("transposeA", False)
            self.transpose_b: bool = op_param.get("transposeB", True)
            self.has_bias: bool = op_param.get("hasBias", True)
            self.linear_type: int = op_param.get("linearType", -1)
            self.backend: str = op_param.get("backend", "")
            self.tiling_k: int = op_param.get("tilingK", 0)
            self.tiling_n: int = op_param.get("tilingN", 0)
            if self.op == MatmulCommon.Op.LINEAR_PARALLEL:
                self.transpose_b = op_param.get("transWeight", True)
                self.has_bias = op_param["bias"] != "None"
            elif self.op == MatmulCommon.Op.LINEAR_SPARSE:
                self.linear_type = MatmulCommon.LinearType.INT8INT8_INT32_FP16
            elif self.op == MatmulCommon.Op.MATMUL:
                self.has_bias = False

    @staticmethod
    def op_param_check_modify_common(op_param, op: int):
        param = MatmulCommon.Param(op_param, op)
        if MatmulCommon.__is_quant(param.linear_type):
            if not MatmulCommon.__is_910b() and (param.transpose_a or not param.transpose_b):
                return False
        return True

    @staticmethod
    def in_num_check_modify_common(op_param, in_num, op: int):
        if op == MatmulCommon.Op.LINEAR_SPARSE or op == MatmulCommon.Op.MATMUL:
            return True
        param = MatmulCommon.Param(op_param, op)
        if MatmulCommon.__is_quant(param.linear_type):
            if not (not param.has_bias and in_num == 3):
                return False
            if not (param.has_bias and in_num == 4):
                return False
        else:
            if not (not param.has_bias and in_num == 2):
                return False
            if not (param.has_bias and in_num == 3):
                return False
        return True

    @staticmethod
    def tensor_desc_check_modify_common(op_param, tensor_desc: TensorDesc, op: int):
        # 稀疏量化仅支持310P
        if op == MatmulCommon.Op.LINEAR_SPARSE and not MatmulCommon.__is_310p():
            return False
        param = MatmulCommon.Param(op_param, op)
        # 非910B量化当前hasBias必须为true
        if MatmulCommon.__is_910b() and MatmulCommon.__is_quant(param.linear_type) and not param.has_bias:
            return False
        # set dtypes
        MatmulCommon.__tensor_desc_check_modify_dtype(param, tensor_desc)
        # set formats
        MatmulCommon.__tensor_desc_check_modify_format(param, tensor_desc)
        # set shapes
        MatmulCommon.__tensor_desc_check_modify_shape(param, tensor_desc)
        if tensor_desc.formats[1] == "fractal_nz":
            MatmulCommon.__tensor_desc_check_modify_nz_weight(param, tensor_desc)
        # set data_gen_ranges
        MatmulCommon.__tensor_desc_check_modify_data_gen_ranges(param, tensor_desc)
        MatmulCommon.param = None
        return True

    @staticmethod
    def __is_quant(linear_type: int):
        return linear_type == MatmulCommon.LinearType.INT8INT8_INT32_FP16 or linear_type == MatmulCommon.LinearType.INT8INT8_INT32_BF16

    @staticmethod
    def __is_910b() -> bool:
        return MatmulCommon.soc_version == "Ascend910B"

    @staticmethod
    def __is_310p() -> bool:
        return MatmulCommon.soc_version == "Ascend310P"

    @staticmethod
    def __align_up(value: int, align: int):
        return (value - 1 + align) // align * align

    @staticmethod
    def __rand_flag() -> bool:
        return random.choice([True, False])

    @staticmethod
    def __rand_within_1000() -> int:
        values = []
        weights = []
        for num in range(2, 1002):
            values.append(num)
            if num < 102:
                weights.append(0.006)  # 0.6
            elif num < 202:
                weights.append(0.003)  # 0.3
            else:
                weights.append(0.000125)  # 0.1
        return random.choices(values, weights)[0]

    @staticmethod
    def __rand_within_10000() -> int:
        values = []
        weights = []
        for num in range(2, 10002):
            values.append(num)
            if num < 102:
                weights.append(0.008)  # 0.8
            elif num < 502:
                weights.append(0.00025)  # 0.2
            elif num < 2002:
                weights.append(0.00004)  # 0.06
            else:
                weights.append(0.000005)  # 0.04
        return random.choices(values, weights)[0]

    @staticmethod
    def __tensor_desc_check_modify_dtype(param: Param, tensor_desc: TensorDesc):
        if MatmulCommon.__is_quant(param.linear_type):
            tensor_desc.dtypes[0] = "int8"
        else:
            if param.linear_type == MatmulCommon.LinearType.FP16FP16_FP32_FP16:
                tensor_desc.dtypes[0] = "float16"
            elif param.linear_type == MatmulCommon.LinearType.BF16BF16_FP32_BF16:
                tensor_desc.dtypes[0] = "bf16"
            else:
                tensor_desc.dtypes[0] = random.choice(["float16", "bf16"])
        tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
        if MatmulCommon.__is_quant(param.linear_type):
            in_tensor_id = 2
            if param.has_bias:
                tensor_desc.dtypes[in_tensor_id] = "int32"
                in_tensor_id = in_tensor_id + 1
            tensor_desc.dtypes[in_tensor_id] = random.choice(["uint64", "int64"])
            if param.op == MatmulCommon.Op.LINEAR_SPARSE:
                tensor_desc.dtypes[4] = "int8"
        else:
            if param.has_bias:
                tensor_desc.dtypes[2] = tensor_desc.dtypes[0]

    @staticmethod
    def __tensor_desc_check_modify_format(param: Param, tensor_desc: TensorDesc):
        tensor_desc.formats[0] = "nd"
        tensor_desc.formats[1] = random.choice(["nd", "fractal_nz"])
        # 910B硬件上，torch_npu不支持INT8数据类型的tensor转NZ
        if MatmulCommon.__is_910b() and MatmulCommon.__is_quant(param.linear_type):
            tensor_desc.formats[1] = "nd"
        # linear_parallel lcoc权重不支持NZ
        if param.backend == "lcoc":
            tensor_desc.formats[1] = "nd"
        # linear_sparse压缩前需为ND
        if param.op == MatmulCommon.Op.LINEAR_SPARSE:
            tensor_desc.formats[1] = "nd"
        in_tensor_id = 2
        if param.has_bias:
            tensor_desc.formats[in_tensor_id] = "nd"
            in_tensor_id = in_tensor_id + 1
        if MatmulCommon.__is_quant(param.linear_type):
            tensor_desc.formats[in_tensor_id] = "nd"
        if param.op == MatmulCommon.Op.LINEAR_SPARSE:
            tensor_desc.formats[4] = "nd"

    @staticmethod
    def __tensor_desc_check_modify_shape(param: Param, tensor_desc: TensorDesc):
        x_dim_num = random.choice([2, 3])
        y_dim_num = random.choice([2, 3]) if param.op == MatmulCommon.Op.MATMUL else 2
        if param.transpose_a and y_dim_num == 2:
            x_dim_num = 2
        if y_dim_num == 3:
            x_dim_num = 3
        batch = MatmulCommon.__rand_within_1000()
        m = MatmulCommon.__rand_within_1000()
        k = MatmulCommon.__rand_within_10000()
        n = MatmulCommon.__rand_within_10000()
        if param.op == MatmulCommon.Op.LINEAR_SPARSE:
            x_dim_num = 2
            k = (k // 64 + 1) * 64
            n = (n // 64 + 1) * 64
            if m > 256:
                m = 256
            if k <= 256:
                k = 320
            if n < 128:
                n = 128
        # set shape
        if x_dim_num == 3:
            tensor_desc.shapes[0] = [batch, k, m] if param.transpose_a else [batch, m, k]
        else:
            tensor_desc.shapes[0] = [k, m] if param.transpose_a else [m, k]
        if y_dim_num == 3:
            tensor_desc.shapes[1] = [batch, n, k] if param.transpose_b else [batch, k, n]
        else:
            tensor_desc.shapes[1] = [n, k] if param.transpose_b else [k, n]
        tensor_id = 2
        if param.has_bias:
            tensor_desc.shapes[tensor_id] = [n] if MatmulCommon.__rand_flag() else [1, n]
            tensor_id = tensor_id + 1
        if MatmulCommon.__is_quant(param.linear_type):
            tensor_desc.shapes[tensor_id] = [n] if MatmulCommon.__rand_flag() else [1, n]
        if param.op == MatmulCommon.Op.LINEAR_SPARSE:
            tensor_desc.shapes[4] = [1]

    @staticmethod
    def __tensor_desc_check_modify_nz_weight(param: Param, tensor_desc: TensorDesc):
        is_w_dim_num_4 = MatmulCommon.__rand_flag()
        if param.op == MatmulCommon.Op.LINEAR_PARALLEL:
            is_w_dim_num_4 = True
        elif param.op == MatmulCommon.Op.MATMUL:
            if len(tensor_desc.shapes[0]) == 3 and len(tensor_desc.shapes[1]) == 2:
                is_w_dim_num_4 = False
        if is_w_dim_num_4:
            align = 32 if MatmulCommon.__is_quant(param.linear_type) else 16
            if len(tensor_desc.shapes[1]) == 2:
                tensor_desc.shapes[1] = [1, MatmulCommon.__align_up(tensor_desc.shapes[1][1], align) // align,
                                         MatmulCommon.__align_up(tensor_desc.shapes[1][0], 16), align]
            else:
                tensor_desc.shapes[1] = [tensor_desc.shapes[1][0],
                                         MatmulCommon.__align_up(tensor_desc.shapes[1][2], align) // align,
                                         MatmulCommon.__align_up(tensor_desc.shapes[1][1], 16), align]

    @staticmethod
    def __tensor_desc_check_modify_data_gen_ranges(param: Param, tensor_desc: TensorDesc):
        tensor_desc.data_gen_ranges = ["-5,5", "-5,5"]
        if param.has_bias:
            tensor_desc.data_gen_ranges.append("-10,10")
        if MatmulCommon.__is_quant(param.linear_type):
            tensor_desc.data_gen_ranges.append("-2,2")
        if param.op == MatmulCommon.Op.LINEAR_SPARSE:
            tensor_desc.data_gen_ranges.append("0,0")


class LinearOperation(MatmulCommon):
    @staticmethod
    def op_param_check_modify(op_param):
        transposeA = op_param['transposeA']
        hasBias = op_param['hasBias']
        outDataType = op_param['outDataType']
        enAccum = op_param['enAccum']
        matmulType = op_param['matmulType']
        if op_param["hasBias"] is True and op_param["enAccum"] is True:
            return False
        if get_soc_version() == "Ascend310P":
            if op_param["enAccum"] is True:
                return False
            if op_param["outDataType"] == 27:
                return False
            if op_param["outDataType"] == 1:
                if op_param["transposeA"] is True:
                    return False
                if op_param["transposeB"] is False:
                    return False
                if op_param["hasBias"] is False:
                    return False
        if matmulType == 1:
            if transposeA is True:
                return False
            if hasBias is True:
                return False
            if outDataType == 1 or outDataType == 27:
                return False
            if enAccum is True:
                return False
        # elif matmulType == 0:
        #     return False
        return True

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        hasBias = op_param['hasBias']
        outDataType = op_param['outDataType']
        enAccum = op_param['enAccum']
        if in_num == 2 and hasBias is True:
            return False
        if in_num == 4:
            if outDataType == -1:
                return False
            if enAccum is True:
                return False
            else:
                if hasBias is False:
                    return False
        if enAccum is True and in_num == 2:
            return False
        if outDataType == 1 or outDataType == 27:
            if in_num == 2:
                return False
            if enAccum is True:
                return False
            if hasBias is True and in_num == 3:
                return False
            if hasBias is False and in_num == 4:
                return False
        if enAccum is False and hasBias is False and outDataType == -1 and in_num != 2:
            return False
        if op_param["matmulType"] == 1:
            if in_num != 2:
                return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        transposeA = op_param['transposeA']
        transposeB = op_param['transposeB']
        hasBias = op_param['hasBias']
        outDataType = op_param['outDataType']
        enAccum = op_param['enAccum']
        matmulType = op_param['matmulType']
        def transposeA_fun(l1):
            if len(l1) == 2:
                return [l1[1], l1[0]]
            elif len(l1) == 3:
                return [l1[0], l1[2], l1[1]]

        def transposeB_fun(l1):
            if len(l1) == 2:
                return [l1[1], l1[0]]
            elif len(l1) == 3:
                return [l1[0], l1[2], l1[1]]
            elif len(l1) == 4:
                if l1[0] == 1:
                    if l1[3] == 16:
                        return [1, int(l1[2] / 16), int(l1[1] * 16), 16]
                    elif l1[3] == 32:
                        return [1, int(l1[2] / 32), int(l1[1] * 32), 32]
                if l1[0] != 1:
                    if l1[3] == 16:
                        return [l1[0], int(l1[2] / 16), int(l1[1] * 16), 16]
                    elif l1[3] == 32:
                        return [l1[0], int(l1[2] / 32), int(l1[1] * 32), 32]

        m = random.randint(1, 10) * 32
        k = random.randint(10, 20) * 32
        n = random.randint(1, 10) * 32
        batch = random.choice(range(2, 10))
        shape_comb1 = [[m, k], [k, n], random.choice([[1, n], [n]])]
        shape_comb2 = [[batch, m, k], [k, n], random.choice([[1, n], [n]])]
        shape_comb3 = [[batch, m, k], [batch, k, n], [batch, n], [batch, n]]
        shape_comb4 = [[m, k], random.choice([[k, n], [1, int(n / 16), k, 16]]), random.choice([[1, n], [n]])]
        shape_comb5 = [[batch, m, k], random.choice([[k, n], [1, int(n / 16), k, 16]]), random.choice([[1, n], [n]])]
        shape_comb6 = [[batch, m, k], random.choice([[batch, k, n], [batch, int(n / 16), k, 16]]), [batch, n]]
        shape_comb7 = [[m, k], random.choice([[k, n], [1, int(n / 32), k, 32]]), random.choice([[1, n], [n]]),
                       random.choice([[1, n], [n]])]
        shape_comb8 = [[batch, m, k], random.choice([[k, n], [1, int(n / 32), k, 32]]), random.choice([[1, n], [n]]),
                       random.choice([[1, n], [n]])]
        shape_comb9 = [[batch, m, k], random.choice([[batch, k, n], [batch, int(n / 32), k, 32]]), [batch, n],
                       [batch, n]]
        shape_comb13 = [[m, k], [k, n], random.choice([[1, n], [n]]), random.choice([[1, n], [n]])]
        shape_comb14 = [[batch, m, k], [k, n], random.choice([[1, n], [n]]), random.choice([[1, n], [n]])]
        shape_comb15 = [[batch, m, k], [batch, k, n], [batch, n], [batch, n], [batch, n]]
        shape_comb10 = [[m, k], [k, n], [m, n]]
        shape_comb11 = [[batch, m, k], [k, n], [batch, m, n]]
        shape_comb12 = [[batch, m, k], [batch, k, n], [batch, m, n]]
        shape_list1 = list()
        shape_list1.append(shape_comb1)
        shape_list1.append(shape_comb2)
        shape_list1.append(shape_comb3)
        shape_list1.append(shape_comb4)
        shape_list1.append(shape_comb5)
        shape_list1.append(shape_comb6)
        shape_list2 = list()
        shape_list2.append(shape_comb13)
        shape_list2.append(shape_comb14)
        shape_list2.append(shape_comb15)
        shape_list2.append(shape_comb7)
        shape_list2.append(shape_comb8)
        shape_list2.append(shape_comb9)
        shape_list3 = list()
        shape_list3.append(shape_comb1)
        shape_list3.append(shape_comb2)
        shape_list3.append(shape_comb3)
        shape_list4 = list()
        shape_list4.append(shape_comb10)
        shape_list4.append(shape_comb11)
        shape_list4.append(shape_comb12)
        shape_list5 = [[1, int(n / 16), k, 16], [batch, int(n / 16), k, 16]]
        shape_list6 = [[1, int(n / 32), k, 32], [batch, int(n / 32), k, 32]]
        shape_list7 = [[k, n], [1, n / 16, k, 16], [1, n / 32, k, 32]]
        tensor_desc.formats[1] = random.choice(["nd", "fractal_nz"])
        if matmulType == 0:
            if outDataType == -1:
                if len(tensor_desc.shapes) == 2:
                    tensor_desc.shapes = (random.choice(shape_list1))[0:2]
                elif len(tensor_desc.shapes) == 3:
                    tensor_desc.shapes = (random.choice(shape_list1))[0:3]
                d_type = random.choice(["bf16", "float16"])
                tensor_desc.dtypes = [d_type for _ in tensor_desc.dtypes]
                if get_soc_version() == "310P" and d_type == "bf16":
                    return False
                if enAccum is False:
                    if hasBias is True and len(tensor_desc.shapes[1]) == 3:
                        tensor_desc.dtypes[2] = random.choice([d_type, "float"])
                        if tensor_desc.dtypes[2] == "float":
                            tensor_desc.formats[1] = "nd"
                            tensor_desc.shapes = (random.choice(shape_list3))[0:3]
                            if get_soc_version() == "Ascend310P":
                                return False
                else:
                    if get_soc_version() == "Ascend310P":
                        return False
                    if hasBias is True:
                        return False
                    if len(tensor_desc.shapes[0]) != len(tensor_desc.shapes[2]):
                        return False
                    tensor_desc.shapes = (random.choice(shape_list4))[0:3]
                    tensor_desc.formats[1] = "nd"
                    tensor_desc.dtypes[2] = "float"

            elif outDataType == 1 or outDataType == 27:
                if len(tensor_desc.shapes) == 2:
                    tensor_desc.shapes = (random.choice(shape_list2))[0:2]
                elif len(tensor_desc.shapes) == 3:
                    tensor_desc.shapes = (random.choice(shape_list2))[0:3]
                elif len(tensor_desc.shapes) == 4:
                    tensor_desc.shapes = (random.choice(shape_list2))[0:4]
                tensor_desc.dtypes[0] = "int8"
                tensor_desc.dtypes[1] = "int8"
                if hasBias is True:
                    tensor_desc.dtypes[2] = "int32"
                    tensor_desc.dtypes[3] = random.choice(["int64", "uint64", "float"])
                else:
                    tensor_desc.dtypes[2] = random.choice(["int64", "uint64", "float"])

                if hasBias is True:
                    if tensor_desc.dtypes[3] == "float" and get_soc_version() == "Ascend310P":
                        return False
                else:
                    if tensor_desc.dtypes[2] == "float" and get_soc_version() == "Ascend310P":
                        return False
            if transposeA is True:
                if tensor_desc.shapes[0] == [batch, m, k] and tensor_desc.shapes[1] in shape_list7:
                    return False
            if tensor_desc.dtypes[0] == "bf16" and get_soc_version() == "Ascend310P":
                return False
            if tensor_desc.shapes[1] in shape_list5 or tensor_desc.shapes[1] in shape_list6:
                tensor_desc.formats[1] = "fractal_nz"
            if transposeA is True:
                tensor_desc.shapes[0] = transposeA_fun(tensor_desc.shapes[0])
            if transposeB is True:
                tensor_desc.shapes[1] = transposeB_fun(tensor_desc.shapes[1])
        elif matmulType == 1:
            EIN_shape = [[m, batch, k], [batch, k, n]]
            tensor_desc.shapes = EIN_shape
            tensor_desc.dtypes = ["float16", "float16"]
            tensor_desc.formats = ["nd", "nd"]
            if transposeB is True:
                tensor_desc.shapes[1] = [batch,n, k]
        return True


class LinearParallelOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if op_param['rankSize'] <= op_param['rankRoot']:
            return False

        backend = op_param['backend']
        if get_soc_version() != 'Ascend910B' and backend in {"lccl", "lcoc"}:
            return False

        if backend in {"lccl", "hccl"} and op_param['quantType'] != -1:
            return False

        return True

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        logging.debug("op_param: %s, in_num: %d", op_param, in_num)
        expected_num = 2 + 2 * (op_param['quantType'] != -1) + op_param['hasResidual']
        return in_num == expected_num

    @staticmethod
    def rand_multiple(n, max_value=1000) -> int:
        return random.choice(range(n, max_value + 1, n))

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        rank_size = op_param['rankSize']
        backend = op_param['backend']
        trans_weight = op_param['transWeight']
        out_dtype = op_param['outDataType']
        quant_type = op_param['quantType']
        has_residual = op_param['hasResidual']

        def set_dtype(index, primary_dtype, secondary_dtype=None):
            tensor_desc.dtypes[index] = primary_dtype if secondary_dtype is None else secondary_dtype

        def set_shape(index, shape):
            tensor_desc.shapes[index] = shape

        primary_dtype = "float16" if out_dtype == 1 else "bf16"
        if quant_type != -1 and backend == "lcoc":
            primary_dtype = "int8" if quant_type == 1 else primary_dtype
            secondary_dtype = "int8"
        else:
            secondary_dtype = primary_dtype

        set_dtype(0, primary_dtype)
        set_dtype(1, secondary_dtype)

        m = LinearParallelOperation.rand_multiple(rank_size)
        k = LinearParallelOperation.rand_multiple(rank_size)
        n = LinearParallelOperation.rand_multiple(rank_size)

        set_shape(0, [m, k])
        set_shape(1, [n, k] if trans_weight else [k, n])

        in_idx = 2
        if quant_type != -1 and backend == "lcoc":
            bias_shape = {0: [1], 1: [n], 2: [int(k / op_param['quantGroupSize']), n]}
            scale_dtype = {"int8": "int64", "float16": primary_dtype, "bf16": primary_dtype}
            tensor_desc.data_gen_ranges[3] = "1,2"
            if quant_type in bias_shape:
                if quant_type == 2 and k % op_param['quantGroupSize'] != 0:
                    return False
                set_shape(in_idx, bias_shape[quant_type])
                set_shape(in_idx + 1, bias_shape[quant_type])
                set_dtype(in_idx, "int32" if primary_dtype == "int8" else primary_dtype)
                set_dtype(in_idx + 1, scale_dtype[primary_dtype])
                in_idx += 2

        if has_residual:
            set_shape(in_idx, [n])
            set_dtype(in_idx, "float16" if out_dtype == 1 else "bf16")

        return True


class LinearSparseOperation(MatmulCommon):
    @staticmethod
    def op_param_check_modify(op_param):
        if op_param["tilingK"] != op_param["tilingN"]:
            return False
        return MatmulCommon.op_param_check_modify_common(op_param, MatmulCommon.Op.LINEAR_SPARSE)

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        return MatmulCommon.in_num_check_modify_common(op_param, in_num, MatmulCommon.Op.LINEAR_SPARSE)

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        return MatmulCommon.tensor_desc_check_modify_common(op_param, tensor_desc, MatmulCommon.Op.LINEAR_SPARSE)


class RmsNormOperation(OperationValidation):
    soc_version = get_soc_version()

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype, soc_version=soc_version):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_dtype))
        if soc_version == 'Ascend310P' and in_dtype == 'bf16':
            return False
        return True

    @staticmethod
    def op_param_check_modify(op_param, soc_version=soc_version):
        layer_type, quant_type = op_param['layerType'], op_param['quantType']
        epsilon = op_param["epsilon"]
        has_bias = op_param["hasBias"]
        dynamic_quant_type = op_param["dynamicQuantType"]
        if soc_version == 'Ascend310P' and layer_type == 3 and quant_type == 2:
            return False
        if layer_type == 1:
            op_param["normParam"] = {"quantType": quant_type}
            op_param["normParam"]["epsilon"] = epsilon
            precision_mode = op_param["precisionMode"]
            modelType = op_param["modelType"]
            rstd = op_param["rstd"]
            if rstd and precision_mode == 1 or rstd and modelType == 1 or precision_mode == 1 and modelType == 1:
                return False
            if quant_type == 2 and (rstd or precision_mode == 1 or modelType == 1):
                return False
            op_param["normParam"]["precisionMode"] = precision_mode
            op_param["normParam"]["modelType"] = modelType
            op_param["normParam"]["dynamicQuantType"] = dynamic_quant_type
        elif layer_type == 2:
            op_param["preNormParam"] = {"quantType": quant_type}
            op_param["preNormParam"]["epsilon"] = epsilon
            op_param["preNormParam"]["hasBias"] = has_bias
            if "normParam" in op_param:
                del op_param["normParam"]
        elif layer_type == 3:
            op_param["postNormParam"] = {"quantType": quant_type}
            op_param["postNormParam"]["epsilon"] = epsilon
            op_param["postNormParam"]["hasBias"] = has_bias
            if "preNormParam" in op_param:
                del op_param["preNormParam"]
        if layer_type == 2 and quant_type == 2 and has_bias:
            return False
        if layer_type == 3 and quant_type == 2 and has_bias:
            return False
        return True

    @staticmethod
    def remove_redundant_key(op_param):
        if "dynamicQuantType" in op_param:
            del op_param["dynamicQuantType"]
        if "epsilon" in op_param:
            del op_param["epsilon"]
        if "hasBias" in op_param:
            del op_param["hasBias"]
        if "precisionMode" in op_param:
            del op_param["precisionMode"]
        if "modelType" in op_param:
            del op_param["modelType"]
        if "quantType" in op_param:
            del op_param["quantType"]
        if "rstd" in op_param:
            del op_param["rstd"]

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_num))
        layer_type = op_param['layerType']
        if layer_type == 1:
            dynamic_quant_type = op_param["normParam"]["dynamicQuantType"]
            quant_type = op_param["normParam"]["quantType"]
            if quant_type == 0 and in_num != 2:
                return False
            if quant_type == 2 and dynamic_quant_type == 0 and in_num != 5:
                return False
            if quant_type == 2 and dynamic_quant_type == 1 and in_num != 3:
                return False
        elif layer_type == 2:
            has_bias = op_param["preNormParam"]['hasBias']
            quant_type = op_param["preNormParam"]["quantType"]
            if quant_type == 0 and has_bias and in_num != 4:
                return False
            if quant_type == 0 and not has_bias and in_num != 3:
                return False
            if quant_type == 2 and in_num != 6:
                return False
        elif layer_type == 3:
            has_bias = op_param["postNormParam"]['hasBias']
            quant_type = op_param["postNormParam"]["quantType"]
            if quant_type == 0 and has_bias and in_num != 4:
                return False
            if quant_type == 0 and not has_bias and in_num != 3:
                return False
            if quant_type == 2 and in_num != 5:
                return False
        RmsNormOperation.remove_redundant_key(op_param)
        return True

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        layerType = op_param['layerType']
        if (in_shape[-1] % 32 != 0) or (in_shape[-1] > 8192):
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        layer_type = op_param['layerType']
        expect_last_dim_pivot = tensor_desc.shapes[0][-1]

        def set_dtype_from_list(l):
            for index in l:
                tensor_desc.dtypes[index] = tensor_desc.dtypes[0]

        if layer_type == 1:
            dynamic_quant_type = op_param["normParam"]["dynamicQuantType"]
            precision_mode = op_param["normParam"]["precisionMode"]
            quant_type = op_param["normParam"]["quantType"]
            if precision_mode == 1:
                tensor_desc.dtypes[0] = "float16"
            if quant_type == 2 and dynamic_quant_type == 0:
                tensor_desc.shapes[1] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[2] = tensor_desc.shapes[1]
                tensor_desc.shapes[3] = [1]
                tensor_desc.shapes[4] = [1]
                set_dtype_from_list([1, 2, 3])
                tensor_desc.dtypes[4] = "int8"
            elif quant_type == 2 and dynamic_quant_type == 1:
                tensor_desc.shapes[1] = [1] * (len(tensor_desc.shapes[1]) - 1) + [expect_last_dim_pivot]
                tensor_desc.shapes[2] = [1] * (len(tensor_desc.shapes[1]) - 1) + [expect_last_dim_pivot]
                tensor_desc.dtypes[0] = "float16"
                set_dtype_from_list([1, 2])
            else:
                tensor_desc.shapes[1] = [1] * (len(tensor_desc.shapes[1]) - 1) + [expect_last_dim_pivot]
                tensor_desc.data_gen_ranges = ["-100,16", "-100,16"]
                tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
        elif layer_type == 2:
            has_bias = op_param["preNormParam"]['hasBias']
            quant_type = op_param["preNormParam"]["quantType"]
            if quant_type == 2:
                tensor_desc.shapes[1] = tensor_desc.shapes[0]
                tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[3] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[4] = [1]
                tensor_desc.shapes[5] = [1]
                tensor_desc.dtypes[0] = "float16"
                set_dtype_from_list([1, 2, 3, 4])
                tensor_desc.dtypes[5] = "int8"
            elif has_bias:
                tensor_desc.shapes[1] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[2] = tensor_desc.shapes[0]
                tensor_desc.shapes[3] = [1, expect_last_dim_pivot]
                set_dtype_from_list([1, 2, 3])
            else:
                tensor_desc.shapes[1] = tensor_desc.shapes[0]
                tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
                set_dtype_from_list([1, 2])
        elif layer_type == 3:
            has_bias = op_param["postNormParam"]['hasBias']
            quant_type = op_param["postNormParam"]["quantType"]
            if has_bias:
                tensor_desc.shapes[1] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[2] = tensor_desc.shapes[0]
                tensor_desc.shapes[3] = [1, expect_last_dim_pivot]
                set_dtype_from_list([1, 2, 3])
            else:
                if quant_type == 0:
                    tensor_desc.shapes[1] = tensor_desc.shapes[0]
                    tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
                    set_dtype_from_list([1, 2])
                else:
                    tensor_desc.shapes[1] = tensor_desc.shapes[0]
                    tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
                    tensor_desc.shapes[3] = [1]
                    tensor_desc.shapes[4] = [1]
                    set_dtype_from_list([1, 2, 3])
                    tensor_desc.dtypes[4] = "int8"

        return True


class RmsNormBackwardOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.shapes[1] = tensor_desc.shapes[0]
        tensor_desc.shapes[2] = tensor_desc.shapes[0].copy()
        tensor_desc.shapes[2][-1] = 1
        tensor_desc.shapes[3] = [tensor_desc.shapes[0][-1]]
        tensor_desc.dtypes[2] = "float"
        return True


class LayerNormOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        layer_type, quant_type = op_param['layerType'], op_param['quantType']
        dynamic_quant_type = op_param["dynamicQuantType"]
        epsilon = op_param["epsilon"]
        begin_norm_axis = op_param['beginNormAxis']
        begin_params_axis = op_param['beginParamsAxis']
        op_mode = op_param['opMode']
        zoom_scale_value = op_param['zoomScaleValue']
        if layer_type == 1:
            op_param["normParam"] = {"quantType": quant_type}
            op_param["normParam"]["epsilon"] = epsilon
            op_param["normParam"]['beginNormAxis'] = begin_norm_axis
            op_param["normParam"]['beginParamsAxis'] = begin_params_axis
            op_param["normParam"]["dynamicQuantType"] = dynamic_quant_type
        elif layer_type == 2:
            op_param["preNormParam"] = {"quantType": quant_type}
            op_param["preNormParam"]['epsilon'] = epsilon
            op_param["preNormParam"]['opMode'] = op_mode
            op_param["preNormParam"]['zoomScaleValue'] = zoom_scale_value
            if "normParam" in op_param:
                del op_param["normParam"]
            if quant_type == 2:
                return False
        elif layer_type == 3:
            op_param["postNormParam"] = {"quantType": quant_type}
            op_param["postNormParam"]['epsilon'] = epsilon
            op_param["postNormParam"]['opMode'] = op_mode
            op_param["postNormParam"]['zoomScaleValue'] = zoom_scale_value
            if "preNormParam" in op_param:
                del op_param["preNormParam"]
        else:
            return False

        return True

    @staticmethod
    def remove_redundant_key(op_param):
        if "beginNormAxis" in op_param:
            del op_param["beginNormAxis"]
        if "beginParamsAxis" in op_param:
            del op_param["beginParamsAxis"]
        if "dynamicQuantType" in op_param:
            del op_param["dynamicQuantType"]
        if "epsilon" in op_param:
            del op_param["epsilon"]
        if "opMode" in op_param:
            del op_param["opMode"]
        if "quantType" in op_param:
            del op_param["quantType"]
        if "zoomScaleValue" in op_param:
            del op_param["zoomScaleValue"]

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_num))
        layer_type = op_param['layerType']
        if layer_type == 1:
            dynamic_quant_type = op_param["normParam"]["dynamicQuantType"]
            quant_type = op_param["normParam"]["quantType"]
            if quant_type == 0 and in_num != 3:
                return False
            if quant_type == 2 and dynamic_quant_type == 0 and in_num != 5:
                return False
            if quant_type == 2 and dynamic_quant_type == 1 and in_num != 3:
                return False
        if layer_type == 2 and in_num != 4:
            return False
        if layer_type == 3:
            quant_type = op_param["postNormParam"]['quantType']
            if quant_type == 0 and in_num != 4:
                return False
            if quant_type == 2 and in_num != 6:
                return False
        LayerNormOperation.remove_redundant_key(op_param)
        return True

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        layer_type = op_param['layerType']
        if layer_type == 1 and op_param["normParam"]["beginNormAxis"] >= len(in_shape):
            return False
        if (in_shape[-1] % 32 != 0):
            return False
        return True

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_dtype))
        layer_type = op_param['layerType']
        if (layer_type == 2 or layer_type == 3) and in_dtype != 'float16':
            return False
        if layer_type == 1 and op_param["normParam"]["quantType"] == 2 and in_dtype != 'float16':
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        layer_type = op_param['layerType']
        expect_last_dim_pivot = tensor_desc.shapes[0][-1]

        def set_dtype_from_list(l):
            for index in l:
                tensor_desc.dtypes[index] = tensor_desc.dtypes[0]

        def set_ranges_from_list(l):
            for index in l:
                tensor_desc.data_gen_ranges[index] = "-1e-5,5e-5"

        if layer_type == 1:
            quant_type = op_param["normParam"]['quantType']
            dynamic_quant_type = op_param["normParam"]["dynamicQuantType"]
            beginNormAxis = op_param["normParam"]["beginNormAxis"]
            if quant_type == 2 and dynamic_quant_type == 0:
                if len(tensor_desc.shapes[0]) > 1:
                    tensor_desc.shapes[0][-2] = 1
                tensor_desc.shapes[1] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[3] = [1]
                tensor_desc.shapes[4] = [1]
                set_ranges_from_list([1, 2])
                tensor_desc.data_gen_ranges[3] = "-2,2"
                tensor_desc.data_gen_ranges[4] = "-128,127"
                set_dtype_from_list([1, 2, 3])
                tensor_desc.dtypes[4] = "int8"
            elif quant_type == 2 and dynamic_quant_type == 1:
                if len(tensor_desc.shapes[0]) > 1:
                    tensor_desc.shapes[0][-2] = 1
                tensor_desc.shapes[1] = [1, expect_last_dim_pivot]
                tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
                set_dtype_from_list([1, 2])
            else:
                tensor_desc.shapes[1] = tensor_desc.shapes[0][beginNormAxis:]
                tensor_desc.shapes[2] = tensor_desc.shapes[0][beginNormAxis:]
                set_dtype_from_list([1, 2])
                set_ranges_from_list([1, 2])
        elif layer_type == 2:
            if len(tensor_desc.shapes[0]) > 1:
                tensor_desc.shapes[0][-2] = 1
            tensor_desc.shapes[1] = tensor_desc.shapes[0]
            tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
            tensor_desc.shapes[3] = [1, expect_last_dim_pivot]
            set_dtype_from_list([1, 2, 3])
            set_ranges_from_list([1, 2, 3])
        elif layer_type == 3:
            quant_type = op_param["postNormParam"]['quantType']
            if len(tensor_desc.shapes[0]) > 1:
                tensor_desc.shapes[0][-2] = 1
            tensor_desc.shapes[1] = tensor_desc.shapes[0]
            tensor_desc.shapes[2] = [1, expect_last_dim_pivot]
            tensor_desc.shapes[3] = [1, expect_last_dim_pivot]
            set_dtype_from_list([1, 2, 3])
            set_ranges_from_list([1, 2, 3])
            if quant_type == 2:
                tensor_desc.shapes[4] = [1]
                tensor_desc.shapes[5] = [1]
                tensor_desc.dtypes[4] = "float16"
                tensor_desc.dtypes[5] = "int8"
                tensor_desc.data_gen_ranges[4] = "-2,2"
                tensor_desc.data_gen_ranges[5] = "-128,127"
        return True


class StridedBatchMatmulOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_shape: {}".format(op_param, in_shape))
        batch = op_param['batch']
        if len(op_param['m']) != batch:
            return False
        if len(op_param['k']) != batch:
            return False
        if len(op_param['n']) != batch:
            return False
        if len(op_param['lda']) != batch:
            return False
        if len(op_param['ldb']) != batch:
            return False
        if len(op_param['ldc']) != batch:
            return False
        if len(op_param['strideA']) != batch:
            return False
        if len(op_param['strideB']) != batch:
            return False
        if len(op_param['strideC']) != batch:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        tensor_desc.shapes[0][0] = sum(op_param['k']) if op_param["transA"] else sum(op_param['m'])
        tensor_desc.shapes[0][1] = sum(op_param['m']) * op_param["headNum"] if op_param["transA"] else sum(
            op_param['k']) * op_param["headNum"]
        tensor_desc.shapes[1][0] = sum(op_param['n']) if op_param["transB"] else sum(op_param['k'])
        tensor_desc.shapes[1][1] = sum(op_param['k']) * op_param["headNum"] if op_param["transB"] else sum(
            op_param['n']) * op_param["headNum"]
        return True


class GenAttentionMaskOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        headNum, seqLen = op_param['headNum'], op_param['seqLen']
        if (headNum <= 0) or len(seqLen) <= 0 or len(seqLen) > 32:
            return False
        maxseqlen = max(seqLen)
        if in_shape[0] != len(seqLen):
            return False
        if in_shape[1] != 1:
            return False
        if in_shape[2] < maxseqlen:
            return False
        if in_shape[3] < maxseqlen:
            return False
        if in_shape[3] != in_shape[2]:
            return False
        return True


class RopeGradOperation(OperationValidation):
    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        qSeqLen = op_param['qSeqLen']
        if len(qSeqLen) <= 0:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        seq_len_list = op_param['qSeqLen']
        logging.debug(
            "tensor_desc_check_modify op_param: {}, tensor_desc.shapes: {}".format(op_param, tensor_desc.shapes))
        q_shape = tensor_desc.shapes[0]
        k_shape = tensor_desc.shapes[1]
        cos_shape = tensor_desc.shapes[2]
        sin_shape = tensor_desc.shapes[3]
        sumSeqLen = sum(seq_len_list)
        maxseqlen = max(seq_len_list)
        tensor_desc.shapes[0][0] = sumSeqLen
        tensor_desc.shapes[1][0] = sumSeqLen
        if tensor_desc.shapes[2][0] < maxseqlen:
            tensor_desc.shapes[2][0] = maxseqlen
        tensor_desc.shapes[3][0] = tensor_desc.shapes[2][0]
        tensor_desc.shapes[2][1] = 128
        tensor_desc.shapes[0][1] = tensor_desc.shapes[2][1]
        tensor_desc.shapes[1][1] = tensor_desc.shapes[0][1]
        tensor_desc.shapes[3][1] = tensor_desc.shapes[2][1]
        return True


class SortOperation(OperationValidation):
    soc_version = get_soc_version()

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        num = op_param['num']
        if num[0] > tensor_desc.shapes[0][-1]:
            return False
        return True

    def in_dtype_check_modify(op_param, in_dtype, soc_version=soc_version):
        if soc_version == 'Ascend310P' and in_dtype in ['bf16','float']:
            return False
        return True


class UnpadWithHiddenStateOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if len(op_param['qSeqLen']) > 32:
            return False
        for seq_len in op_param['qSeqLen']:
            if seq_len <= 0 or not isinstance(seq_len, int):
                return False
        logging.debug("op_param: %s", op_param)
        return True

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        seq_len_list, max_seq_len_imm = op_param['qSeqLen'], op_param['maxSeqLen']
        if in_shape[0] != len(seq_len_list):
            return False
        if in_shape[1] != max_seq_len_imm:
            return False
        if in_shape[2] not in [2048, 4096]:
            return False
        return True


class PadWithHiddenStateOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if len(op_param['qSeqLen']) > 32:
            return False
        for seq_len in op_param['qSeqLen']:
            if seq_len <= 0 or not isinstance(seq_len, int):
                return False
        logging.debug("op_param: %s", op_param)
        return True

    @staticmethod
    def in_shape_check_modify(op_param, in_shape):
        logging.debug("op_param: {}, in_num: {}".format(op_param, in_shape))
        if in_shape[0] != 0:
            return False
        if in_shape[1] not in [2048, 4096]:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        seq_len_list = op_param['qSeqLen']
        tensor_desc.shapes[0][0] = sum(seq_len_list)
        return True


class FastSoftMaxOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if len(op_param['qSeqLen']) > 32:
            return False
        for seq_len in op_param['qSeqLen']:
            if seq_len <= 0 or not isinstance(seq_len, int):
                return False
        logging.debug("op_param: %s", op_param)
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        seq_len_list, head_num_imm = op_param['qSeqLen'], op_param['headNum']
        tensor_desc.shapes[0][0] = sum(x ** 2 for x in seq_len_list) * head_num_imm
        return True


class FastSoftMaxGradOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if len(op_param['qSeqLen']) > 32:
            return False
        for seq_len in op_param['qSeqLen']:
            if seq_len <= 0 or not isinstance(seq_len, int):
                return False
        logging.debug("op_param: %s", op_param)
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        seq_len_list, head_num_imm = op_param['qSeqLen'], op_param['headNum']
        tensor_desc.shapes[0][0] = sum(x ** 2 for x in seq_len_list) * head_num_imm
        tensor_desc.shapes[1][0] = sum(x ** 2 for x in seq_len_list) * head_num_imm
        return True


class GroupedMatmulWithRoutingOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        """
        Args:
            op_param (Dict): '{ParamA: aaa, ParamB: bbb}'.

        Returns:
            Bool: checking results.
        """
        soc_version = get_soc_version()
        if soc_version == "Ascend310P":
            return False
        return True

    @staticmethod
    def in_num_check_modify(op_param, in_num):
        logging.debug("op_param: %s, in_num: %d", op_param, in_num)
        outDataType = op_param['outDataType']
        if outDataType == -1 and in_num != 4:
            return False
        if outDataType == 1 and in_num == 4:
            return False
        if outDataType == 27 and in_num == 4:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        groupedMatmulType, topK, outDataType, transposeB = op_param['groupedMatmulType'], op_param['topK'], op_param[
            'outDataType'], op_param['transposeB']
        num_tokens = random.randint(128, 512)
        num_experts = random.randint(128, 256)
        hiddenSizeIn_Down = hiddenSizeOut_Up = random.randint(1, 8) * 32
        hiddenSizeIn_Up = hiddenSizeOut_Down = random.randint(1, 160) * 32
        if len(tensor_desc.dtypes) == 4:
            tensor_desc.shapes = [[128, 128], [128, 32, 32], [128], [128]]
            tensor_desc.shapes[2] = [num_experts]
            tensor_desc.shapes[3] = [num_tokens * topK]
            tensor_desc.dtypes[1] = tensor_desc.dtypes[0]
            if groupedMatmulType == 0:
                tensor_desc.shapes[0] = [num_tokens, hiddenSizeIn_Up]  # AcTensor
                if transposeB:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeOut_Up, hiddenSizeIn_Up]  # ExpertWeight
                else:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeIn_Up, hiddenSizeOut_Up]
            elif groupedMatmulType == 1:
                tensor_desc.shapes[0] = [num_tokens * topK, hiddenSizeIn_Down]  # AcTensor
                if transposeB:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeOut_Down, hiddenSizeIn_Down]  # ExpertWeight
                else:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeIn_Down, hiddenSizeOut_Down]  # ExpertWeight

            tensor_desc.dtypes[2::] = ['int32', 'int32']
            tensor_desc.formats[::] = ['nd', 'nd', 'nd', 'nd']
        elif len(tensor_desc.dtypes) == 6:
            tensor_desc.dtypes[0::] = ['int8', 'int8', 'int32', 'int32', 'float', 'float']
            tensor_desc.formats[2::] = ['nd', 'nd', 'nd', 'nd']
            tensor_desc.shapes = [[128, 128], [128, 32, 32], [128], [128], [128, 32], [128]]
            tensor_desc.shapes[2] = [num_experts]
            tensor_desc.shapes[3] = [num_tokens * topK]
            if groupedMatmulType == 0:
                tensor_desc.shapes[0] = [num_tokens, hiddenSizeIn_Up]
                tensor_desc.shapes[4] = [num_experts, hiddenSizeOut_Up]
                tensor_desc.shapes[5] = [num_tokens]  # mscale
                if transposeB:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeOut_Up, hiddenSizeIn_Up]  # ExpertWeight
                else:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeIn_Up, hiddenSizeOut_Up]  # ExpertWeight
            elif groupedMatmulType == 1:
                tensor_desc.shapes[0] = [num_tokens * topK, hiddenSizeIn_Down]  # AcTensor
                tensor_desc.shapes[4] = [num_experts, hiddenSizeOut_Down]
                tensor_desc.shapes[5] = [num_tokens * topK]  # mscale
                if transposeB:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeOut_Down, hiddenSizeIn_Down]  # ExpertWeight
                else:
                    tensor_desc.shapes[1] = [num_experts, hiddenSizeIn_Down, hiddenSizeOut_Down]  # ExpertWeight
            if tensor_desc.formats[0] == 'fractal_nz':
                return False
            tensor_desc.data_gen_ranges[4::] = ["-0.1,0.1", "-0.1,0.1"]
        return True


class GroupTopkOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        groupNum, k = op_param['groupNum'], op_param['k']
        groupMultiFlag, n = op_param['groupMultiFlag'], op_param['n']
        if groupMultiFlag == 1 and groupNum > 128:
            return False
        if get_soc_version() == 'Ascend310P':
            return False
        elif groupNum < k:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        groupNum = op_param['groupNum']
        groupMultiFlag, n = op_param['groupMultiFlag'], op_param['n']
        tensor_desc.dtypes[1] = 'int32'
        tensor_desc.shapes[1] = [1024]
        if tensor_desc.dtypes[0] == 'int32':
            return False
        elif len(tensor_desc.shapes[0]) != 2:
            return False
        if tensor_desc.shapes[0][1] % groupNum != 0:
            return False
        if groupMultiFlag == 1:
            if tensor_desc.shapes[0][1] / groupNum < n:
                return False
        return True


class BlockCopyOperation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        # print(tensor_desc)
        if get_soc_version() == 'Ascend310P':
            tensor_desc.dtypes[2] = 'int32'
            tensor_desc.dtypes[3] = 'int32'
            tensor_desc.dtypes[4] = 'int32'
            tensor_desc.formats[0] = random.choice(["nd", "fractal_nz"])
            tensor_desc.formats[1] = random.choice(["nd", "fractal_nz"])
            tensor_desc.formats[2] = "nd"
            tensor_desc.formats[3] = "nd"
            tensor_desc.formats[4] = "nd"
            tensor_desc.dtypes[0] = tensor_desc.dtypes[1]
            tensor_desc.shapes[2][0] = tensor_desc.shapes[4][0]
            tensor_desc.shapes[0][0] = tensor_desc.shapes[1][0]
            if len(tensor_desc.shapes[0]) != 4 or len(tensor_desc.shapes[1]) != 4:
                return False
            tensor_desc.shapes[0][1] = tensor_desc.shapes[1][1]
            tensor_desc.shapes[0][2] = tensor_desc.shapes[1][2]
            tensor_desc.shapes[0][3] = tensor_desc.shapes[1][3]
            tensor_desc.formats[0] = tensor_desc.formats[1]
            if len(tensor_desc.shapes[2]) != 1 or len(tensor_desc.shapes[3]) != 1 or len(tensor_desc.shapes[4]) != 1:
                return False
            if tensor_desc.shapes[0][0] < tensor_desc.shapes[2][0]:
                return False
            if tensor_desc.shapes[0][0] < tensor_desc.shapes[3][0]:
                return False
            if tensor_desc.shapes[2][0] > tensor_desc.shapes[3][0]:
                return False
            if tensor_desc.shapes[2][0] + tensor_desc.shapes[3][0] > tensor_desc.shapes[0][0]:
                return False
            if tensor_desc.dtypes[0] != 'float16' or tensor_desc.dtypes[1] != 'float16':
                return False
            if tensor_desc.formats[0] == "nd" and tensor_desc.formats[1] == "nd":
                if (tensor_desc.shapes[0][1] * tensor_desc.shapes[0][2] * tensor_desc.shapes[0][3]) % 16 != 0:
                    return False
            if tensor_desc.formats[0] == "fractal_nz" and tensor_desc.formats[1] == "fractal_nz":
                tensor_desc.shapes[0][3] = 16
                tensor_desc.shapes[1][3] = 16
                if ((tensor_desc.shapes[0][2] % 16) != 0) or ((tensor_desc.shapes[1][2] % 16) != 0):
                    return False
            tensor_desc.data_gen_types = ["random", "random", "customize", "customize", "customize"]
            return True
        else:
            tensor_desc.dtypes[2] = 'int32'
            tensor_desc.dtypes[3] = 'int32'
            tensor_desc.dtypes[4] = 'int32'
            tensor_desc.formats[0] = "nd"
            tensor_desc.formats[1] = "nd"
            tensor_desc.formats[2] = "nd"
            tensor_desc.formats[3] = "nd"
            tensor_desc.formats[4] = "nd"
            tensor_desc.dtypes[0] = tensor_desc.dtypes[1]
            tensor_desc.shapes[2][0] = tensor_desc.shapes[4][0]
            tensor_desc.shapes[0][0] = tensor_desc.shapes[1][0]
            if len(tensor_desc.shapes[0]) != 4 or len(tensor_desc.shapes[1]) != 4:
                return False
            tensor_desc.shapes[0][1] = tensor_desc.shapes[1][1]
            tensor_desc.shapes[0][2] = tensor_desc.shapes[1][2]
            tensor_desc.shapes[0][3] = tensor_desc.shapes[1][3]
            if len(tensor_desc.shapes[2]) != 1 or len(tensor_desc.shapes[3]) != 1 or len(tensor_desc.shapes[4]) != 1:
                return False
            if tensor_desc.shapes[0][0] < tensor_desc.shapes[2][0]:
                return False
            if tensor_desc.shapes[0][0] < tensor_desc.shapes[3][0]:
                return False
            if tensor_desc.shapes[2][0] > tensor_desc.shapes[3][0]:
                return False
            if tensor_desc.shapes[2][0] + tensor_desc.shapes[3][0] > tensor_desc.shapes[0][0]:
                return False
            tensor_desc.data_gen_types = ["random", "random", "customize", "customize", "customize"]
            return True


class GroupedMatmulInplaceAddOperation(OperationValidation):

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        transposeA, transposeB = op_param['transposeA'], op_param['transposeB']
        tensor_desc.dtypes[0] = tensor_desc.dtypes[1]
        tensor_desc.dtypes[2] = 'int64'
        tensor_desc.dtypes[3] = 'float'
        tensor_desc.dtypes[0] = tensor_desc.dtypes[1]
        m = random.choice([10, 16, 256, 560])
        k = random.randint(3, 655)
        n = random.randint(1, 655)
        numGroup = random.randint(1, 128)
        tensor_desc.shapes = [[1, 2], [1, 2], [1], [1, 2]]
        tensor_desc.data_gen_ranges = ["-5,5", "-5,5", "0,5", "-5,5"]
        if tensor_desc.dtypes[0] == 'int64' or tensor_desc.dtypes[1] == 'float':
            return False
        if transposeA == False:
            tensor_desc.shapes[0] = [m, k]
            tensor_desc.shapes[1] = [k, n]
            tensor_desc.shapes[2] = [numGroup]
            tensor_desc.shapes[3] = [m, numGroup * n]
        if transposeA == True:
            tensor_desc.shapes[0] = [k, m]
            tensor_desc.shapes[1] = [k, n]
            tensor_desc.shapes[2] = [numGroup]
            tensor_desc.shapes[3] = [m, numGroup * n]
        return True


class AllToAllOperation(OperationValidation):
    @staticmethod
    def op_param_check_modify(op_param):
        if op_param["backend"] != "lccl" and op_param["transpose"] is True:
            return False
        if get_soc_version(only_910C=True) != "Ascend910C" and op_param["backend"] != "hccl" and op_param["transpose"] is False:
            return False
        return True

    @staticmethod
    def in_dtype_check_modify(op_param, in_dtype):
        if op_param["transpose"] is not False and in_dtype not in ["bf16", "float16"]:
            return False
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        ranksize = op_param['rankSize']
        from functools import reduce
        import operator
        def multiply_elements(lst):
            return reduce(operator.mul, lst, 1)
        if multiply_elements(tensor_desc.shapes[0]) % int(ranksize) != 0:
            return False
        if multiply_elements(tensor_desc.shapes[0]) > 39845888:
            return False
        if op_param["transpose"] is True:
            if len(tensor_desc.shapes[0]) != 2:
                return False
            if tensor_desc.shapes[0][1] * 2 > 90 * 1024:
                return False
            if tensor_desc.shapes[0][1] % int(ranksize) != 0:
                return False
        return True


class CohereLayerNormOperation(OperationValidation):

    @staticmethod
    def op_param_check_modify(op_param):
        op_param["epsilon"] = float(op_param["epsilon"])
        return True


    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if len(tensor_desc.shapes[0]) == 2:
            return False
        if len(tensor_desc.shapes[1]) != 2:
            return False
        if tensor_desc.shapes[0][-1] % 32 != 0:
            return False
        tensor_desc.dtypes[0] = tensor_desc.dtypes[1]
        tensor_desc.shapes[1][:] = tensor_desc.shapes[0][-2:]
        return True


class SwigluQuantOperation(OperationValidation):

    @staticmethod
    def op_param_check_modify(op_param):
        return True

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if tensor_desc.shapes[0][1] == 1:
            return False
        return True


class FaUpdateOperation(OperationValidation):

    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        faUpdateType, sp = op_param['faUpdateType'], op_param['sp']
        tensor_desc.shapes[0][0] = tensor_desc.shapes[1][0] = sp
        tensor_desc.shapes[0][1] = tensor_desc.shapes[1][1]
        if len(tensor_desc.shapes[0]) != 2 or len(tensor_desc.shapes[1]) != 3:
            return False
        if tensor_desc.shapes[1][2] > 512:
            return False
        return True

class ScatterElementsV2Operation(OperationValidation):
    @staticmethod
    def tensor_desc_check_modify(op_param, tensor_desc: TensorDesc):
        if tensor_desc.dtypes[1] != 'int32' and tensor_desc.dtypes[1] != 'int64':
            tensor_desc.dtypes[1] = 'int32'
 
        # 1：indice_tensor dim需要和input_tensor保持一致。
        if len(tensor_desc.shapes[1]) != len(tensor_desc.shapes[0]):
            return False
 
        # 2：indice_tensor非尾轴和非0轴的shape需要和input_tensor完全相同。
        for i in range(1, len(tensor_desc.shapes[0]) - 1):
            tensor_desc.shapes[1][i] = tensor_desc.shapes[0][i]
 
 
        # 8：indice_tensor 首维和尾维需要小于 input_tensor
        if tensor_desc.shapes[1][0] > tensor_desc.shapes[0][0]:
            tensor_desc.shapes[1][0] = tensor_desc.shapes[0][0]
 
        # 8：indice_tensor 首维和尾维需要小于 input_tensor
        input_tensor_last_dim = len(tensor_desc.shapes[1]) - 1
        if tensor_desc.shapes[1][input_tensor_last_dim] > tensor_desc.shapes[0][input_tensor_last_dim]:
            tensor_desc.shapes[1][input_tensor_last_dim] = tensor_desc.shapes[0][input_tensor_last_dim]
 
        # 4：update_tensor dim和shape需要和indice_tensor完全一致
        if len(tensor_desc.shapes[1]) != len(tensor_desc.shapes[2]):
            tensor_desc.shapes[2] = tensor_desc.shapes[1]
 
        # 4：update_tensor dim和shape需要和indice_tensor完全一致
        for i in range(0, len(tensor_desc.shapes[1])):
            tensor_desc.shapes[2][i] = tensor_desc.shapes[1][i]
 
        # 5:update_tensor 的dtype需要和input_tensor保持一致
        if tensor_desc.dtypes[2] != tensor_desc.dtypes[0]:
            tensor_desc.dtypes[2] = tensor_desc.dtypes[0]
 
        tensor_desc.data_gen_ranges[1] = f"1,{tensor_desc.shapes[0][-1] - 1}"
 
        axis, reduction = op_param['axis'], op_param['reduction']
        if axis != -1:
            op_param['axis'] = -1
 
        if reduction != 0 and reduction != 1:
            return False
        return True