# -*- coding: UTF-8 -*-
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import re
import sys
import stat

HEAD = """/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/log.h"
#include <string>
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/types.h"
#include "fuzz_util.h"
#include "secodeFuzz.h"
#include <gtest/gtest.h>
"""

BASE_TYPE_TABLE = {'uint8_t': 'u8', 'uint16_t': 'u16', 'uint32_t': 'u32', 'uint64_t': 'u64',
                   'int8_t': 's8', 'int16_t': 's16', 'int32_t': 's32', 'int64_t': 's64',
                   'std::string': 'char *', 'bool': 'bool', 'int': 'int', 'float': 'float',
                   'double': 'double', 'size_t': 'size_t'}
FUZZ_TYPE_TABLE = {'u8': 'DT_SetGetU8', 'u16': 'DT_SetGetU16', 'u32': 'DT_SetGetU32', 'u64': 'DT_SetGetU64',
                   's8': 'DT_SetGetS8', 's16': 'DT_SetGetS16', 's32': 'DT_SetGetS32', 's64': 'DT_SetGetS64',
                   'char *': 'DT_SetGetString', 'bool': 'FuzzUtil::GetRandomBool', 'int': 'DT_SetGetS32',
                   'float': 'DT_SetGetFloat',
                   'double': 'DT_SetGetDouble', 'size_t': 'DT_SetGetU64'}


def get_all_param_enums(file_path):
    param_enum = dict()
    file_lines = []
    with open(file_path, 'r') as file:
        file_lines = file.readlines()
    enum_name = ''
    end_flag = True
    value = 0
    struct_end_flag = True
    param_name = ''
    for line in file_lines:
        line = line.split('//')[0]
        line = line.strip()
        if line.startswith('struct') and struct_end_flag and not line.endswith('};'):
            struct_end_flag = False
            param_name = line.split(' ')[1]
        if line.startswith('enum'):
            end_flag = False
            items = line.split(' ')
            enum_name = items[1]
            if param_name != '':
                enum_name = param_name + '::' + enum_name
            if enum_name not in param_enum:
                param_enum[enum_name] = []
                continue
        if not end_flag:
            if '};' == line:
                end_flag = True
                enum_name = ''
                continue
            items = line.split(' ')
            if len(items) > 1:
                value = eval(items[2].replace(',', ''))
            else:
                value += 1
            if value not in param_enum[enum_name]:
                param_enum[enum_name].append(value)
        if end_flag and '};' == line and not struct_end_flag:
            struct_end_flag = True
            param_name = ''
            continue
    return param_enum


def get_temp_struct_param(items, op_name, temp_struct_param, temp_struct_name, param_enums):
    temp_value = ''
    temp_type = items[0]
    temp_param = items[1]
    if len(items) > 2:
        temp_value = items[-1].replace(';', '')
    if op_name + 'Param::' + temp_type in param_enums:
        temp_type = op_name + 'Param::' + temp_type
        temp_value = param_enums[temp_type]
    elif temp_type in param_enums:
        temp_value = param_enums[temp_type]
    temp_struct_param[temp_struct_name].append((temp_param, temp_type, temp_value))


def get_operation_inner_param(items, op_name, temp_struct_param, operation_param, param_enums):
    if len(items) == 2:
        param_type = items[0]
        param = items[1].replace(';', '')
        if param_type in temp_struct_param:
            for inner_param in temp_struct_param[param_type]:
                inner_param_name, inner_param_type, inner_param_value = inner_param
                param_name = param + '.' + inner_param_name
                operation_param[param_name] = (inner_param_type, inner_param_value)
        elif op_name + 'Param::' + param_type in param_enums:
            param_type = op_name + 'Param::' + param_type
            value = param_enums[param_type]
            operation_param[param] = (param_type, value)
        elif param_type in param_enums:
            value = param_enums[param_type]
            operation_param[param] = (param_type, value)
        else:
            operation_param[param] = (param_type, '')
    elif len(items) == 4:
        param_type = items[0]
        param = items[1]
        if op_name + 'Param::' + param_type in param_enums:
            param_type = op_name + 'Param::' + param_type
            value = param_enums[param_type]
        elif param_type in param_enums:
            value = param_enums[param_type]
        else:
            value = items[-1].replace(';', '')
        operation_param[param] = (param_type, value)


def get_all_operation_params(file_path):
    all_operation_params = dict()
    temp_struct_param = dict()
    param_enums = get_all_param_enums(file_path)
    op_name = ''
    temp_struct_name = ''
    file_lines = []
    skip_flag = False
    with open(file_path, 'r') as file:
        file_lines = file.readlines()
    end_flag = True
    end_inner_flag = True
    for line in file_lines:
        line = line.strip()
        if line.startswith('//!'):
            continue
        line = line.split('//')[0]
        items = line.split(' ')

        if '};' == line:
            if skip_flag:
                skip_flag = False
            elif not end_inner_flag:
                end_inner_flag = True
            elif not end_flag:
                end_flag = True
            continue

        if line.startswith('enum'):
            skip_flag = True
            continue

        if skip_flag:
            continue

        if end_flag and line.startswith('struct ') and end_inner_flag:
            op_name = ''.join(re.findall('[A-Z][^A-Z]*', items[1])[:-1])
            operation_param = dict()
            all_operation_params[op_name] = operation_param
            if not line.endswith('};'):
                end_flag = False
            continue
        
        if not end_flag:
            if line.startswith('struct'):
                end_inner_flag = False
                temp_struct_name = items[1]
                temp_struct_param[temp_struct_name] = []
            elif not end_inner_flag:
                get_temp_struct_param(items, op_name, temp_struct_param, temp_struct_name, param_enums)
            elif end_inner_flag:
                get_operation_inner_param(items, op_name, temp_struct_param, operation_param, param_enums)
    
    return all_operation_params


class OperationFuzzTestGenerator:

    def __init__(self, operation_name, operation_params, seed, count, is_reproduce, op_type):
        self.operation_name = operation_name
        self.seed = seed
        self.count = count
        self.is_reproduce = is_reproduce
        self.tab_num = 0
        self.operation_params = operation_params
        self.op_type = op_type


    def __generate_svector_param(self, param_name, param_type, tab_string, real_param_name):
        match = re.search(r'<.*?>', param_type)
        svector_type = match.group(0)[1:-1]
        svector_type = BASE_TYPE_TABLE[svector_type]
        temp_content = f'{tab_string}u32 {real_param_name}SvectorLen = (u32) std::rand();\n'
        temp_content += f'{tab_string}{real_param_name}SvectorLen = {real_param_name}SvectorLen % 10;\n'
        temp_content += f'{tab_string}ATB_LOG(INFO) << "{real_param_name}SvectorLen: " << {real_param_name}SvectorLen;\n'
        temp_content += f'{tab_string}{param_name}.reserve({real_param_name}SvectorLen);\n'
        temp_content += f'{tab_string}{param_name}.resize({real_param_name}SvectorLen);\n'
        temp_content += f'{tab_string}for (size_t i = 0; i < {real_param_name}SvectorLen; i++) {{\n'
        self.tab_num += 1
        temp_content += '\t' * self.tab_num + f'{param_name}[i] = ({svector_type}) (std::rand() % 20) - 10;\n'
        temp_content += '\t' * self.tab_num + f'ATB_LOG(INFO) << "{param_name}[" << i <<"]: " << {param_name}[i];\n'
        self.tab_num -= 1
        temp_content += '\t' * self.tab_num + '}\n'
        return temp_content

    def __generate_inner_param_fuzz(self, param_name, param_type, param_value):
        real_param_name = param_name.split('.')[-1]
        tab_string = '\t' * self.tab_num
        fuzz_param_content = ''
        if param_type in BASE_TYPE_TABLE:
            fuzz_type = BASE_TYPE_TABLE[param_type]
            if fuzz_type == 'char *':
                if '' == param_value:
                    param_value = '""'
                temp_content = f'{tab_string}char {real_param_name}InitValue[] = {param_value};\n'
                temp_content += f'{tab_string}{param_name} = {FUZZ_TYPE_TABLE[fuzz_type]}(&g_Element[fuzzIndex++], {len(param_value)}, 1000, {real_param_name}InitValue);\n'
                temp_content += f'{tab_string}ATB_LOG(INFO) << "{param_name}: " << {param_name};\n'
            elif fuzz_type == 'bool':
                temp_content = f'{tab_string}{param_name} = {FUZZ_TYPE_TABLE[fuzz_type]}(fuzzIndex);\n'
                temp_content += f'{tab_string}ATB_LOG(INFO) << "{param_name}: " << {param_name};\n'
            else:
                if '' == param_value:
                    param_value = 0
                temp_content = f'{tab_string}{param_name} = *({fuzz_type} *) {FUZZ_TYPE_TABLE[fuzz_type]}(&g_Element[fuzzIndex++], {param_value});\n'
                temp_content += f'{tab_string}ATB_LOG(INFO) << "{param_name}: " << {param_name};\n'
        elif 'SVector' in param_type or 'vector' in param_type:
            temp_content = self.__generate_svector_param(param_name, param_type, tab_string, real_param_name)
        elif 'HcclComm' == param_type:
            temp_content = f'{tab_string}{param_name} = {param_value};\n'
        elif 'aclDataType' == param_type:
            real_param_name = param_name.split('.')[1]
            temp_content = f'{tab_string}s32 {real_param_name}Dtype = *(s32 *) DT_SetGetS32(&g_Element[fuzzIndex++], 1);\n'
            temp_content += f'{tab_string}{param_name} = FuzzUtil::GetRandomAclDataType({real_param_name}Dtype);\n'
        elif 'Type' in param_type or 'Mode' in param_type or 'Cfg' or 'Layout' in param_type:
            value_len = len(param_value)
            param_value = str(param_value).replace('[', '{').replace(']', '}')
            real_param_name = param_name.split('.')[1] + param_name.split('.')[-1]
            temp_content = f'{tab_string}int {real_param_name}EnumTable[] = {param_value};\n'
            temp_content += f'{tab_string}{param_name} = static_cast<atb::{self.op_type}::{param_type}>(*(int *) DT_SetGetNumberEnum(&g_Element[fuzzIndex++], 1, {real_param_name}EnumTable, {value_len}));\n'
            temp_content += f'{tab_string}ATB_LOG(INFO) << "{param_name}: " << {param_name};\n'
        elif 'Param' in param_type:
            op_param_name = ''.join(re.findall('[A-Z][^A-Z]*', param_type)[:-1])
            temp_content = ''
            inner_params = ALL_INFER_OPERATION_PARAMS[op_param_name]
            for inner_param_name, inner_param_content in inner_params.items():
                inner_param_type = inner_param_content[0]
                inner_param_value = inner_param_content[1]
                temp_content += self.__generate_inner_param_fuzz(param_name + '.' + inner_param_name, inner_param_type,
                                                                 inner_param_value)
        fuzz_param_content += temp_content
        return fuzz_param_content

    def __generate_param_fuzz(self, operation_param):
        tab_string = '\t' * self.tab_num
        op_name = ''.join(re.findall('[A-Z][^A-Z]*', self.operation_name)[:-1])
        operation_param_name = op_name + 'Param'
        fuzz_param_content = f'{tab_string}{self.op_type}::{operation_param_name} param;\n'
        for param_name, param_cotent in operation_param.items():
            param_name = 'param.' + param_name
            param_type = param_cotent[0]
            param_value = param_cotent[1]
            fuzz_param_content += self.__generate_inner_param_fuzz(param_name, param_type, param_value)
        return fuzz_param_content

    def __generate_operation_fuzz(self):
        tab_string = '\t' * self.tab_num
        operation_name = self.operation_name[0].lower() + self.operation_name[1:]
        fuzz_operation_content = f'{tab_string}Operation *{operation_name} = nullptr;\n'
        fuzz_operation_content += f'{tab_string}uint32_t fuzzIndex = 0;\n'
        fuzz_param_content = self.__generate_param_fuzz(self.operation_params)
        fuzz_operation_content += fuzz_param_content
        fuzz_operation_content += f'{tab_string}CreateOperation(param, &{operation_name});\n'
        return fuzz_operation_content

    def __generate_intensor_des(self):
        operation_name = self.operation_name[0].lower() + self.operation_name[1:]
        tensor_des_content = '\t' * self.tab_num + 'SVector<TensorDesc> inTensorDescs;\n'
        tensor_des_content += '\t' * self.tab_num + f'for (size_t i = 0; i < {operation_name}->GetInputNum(); i++) {{\n'
        self.tab_num += 1
        tensor_des_content += '\t' * self.tab_num + 'TensorDesc desc;\n'
        tensor_des_content += '\t' * self.tab_num + 's32 dtypeRdNum = (s32) (std::rand() % 100) - 50;\n'
        tensor_des_content += '\t' * self.tab_num + 's32 formatRdNum = (s32) (std::rand() % 100) - 50;\n'
        tensor_des_content += '\t' * self.tab_num + 's32 dimNumRdNum = (s32) (std::rand() % 100) - 50;\n'
        tensor_des_content += '\t' * self.tab_num + 'desc.dtype = FuzzUtil::GetRandomAclDataType(dtypeRdNum);\n'
        tensor_des_content += '\t' * self.tab_num + 'ATB_LOG(INFO) << "desc.dtype: " << desc.dtype;\n'
        tensor_des_content += '\t' * self.tab_num + 'desc.format = FuzzUtil::GetRandomAclFormat(formatRdNum);\n'
        tensor_des_content += '\t' * self.tab_num + 'ATB_LOG(INFO) << "desc.format: " << desc.format;\n'
        tensor_des_content += '\t' * self.tab_num + 'for (size_t i = 0; i < MAX_DIM; i++) {\n'
        self.tab_num += 1
        tensor_des_content += '\t' * self.tab_num + 's32 dim = (s32) (std::rand() % 2000) - 1000;\n'
        tensor_des_content += '\t' * self.tab_num + 'desc.shape.dims[i] = dim;\n'
        tensor_des_content += '\t' * self.tab_num + 'ATB_LOG(INFO) << "desc.shape.dims[" << i << "]: " << desc.shape.dims[i];\n'
        self.tab_num -= 1
        tensor_des_content += '\t' * self.tab_num + '}\n'
        tensor_des_content += '\t' * self.tab_num + 'desc.shape.dimNum = FuzzUtil::GetRandomDimNum(dimNumRdNum);\n'
        tensor_des_content += '\t' * self.tab_num + 'ATB_LOG(INFO) << "desc.shape.dimNum: " << desc.shape.dimNum;\n'
        tensor_des_content += '\t' * self.tab_num + f'inTensorDescs.push_back(desc);\n'
        self.tab_num -= 1
        tensor_des_content += '\t' * self.tab_num + '}\n'
        return tensor_des_content

    def __get_fuzz_name(self, fuzz_type):
        return f'\tstd::string fuzzName = "{self.operation_name}DTFuzz{fuzz_type}";\n'

    def __dt_fuzz_start(self, fuzz_type):
        content = self.__get_fuzz_name(fuzz_type)
        content += self.tab_num * '\t' + f'DT_FUZZ_START({self.seed}, {self.count}, const_cast<char*>(fuzzName.c_str()), {self.is_reproduce}) {{\n'
        self.tab_num += 1
        return content

    def __generate_inferShape_test(self):
        operation_name = self.operation_name[0].lower() + self.operation_name[1:]
        fuzz_type = 'InferShape'
        inferShape_test = f'TEST({self.operation_name}DTFuzz, {fuzz_type})\n{{\n'
        self.tab_num += 1
        tab_string = '\t' * self.tab_num
        log_content = f'{tab_string}ATB_LOG(INFO) << "begin====================";\n'
        inferShape_content = inferShape_test
        inferShape_content += f'{tab_string}std::srand(time(NULL));\n'
        inferShape_content += log_content
        inferShape_content += self.__dt_fuzz_start(fuzz_type)
        inferShape_content += self.__generate_operation_fuzz()
        assert_content = '\t' * self.tab_num + f'if({operation_name} == nullptr) {{\n'
        self.tab_num += 1
        assert_content += '\t' * self.tab_num + 'continue;\n'
        self.tab_num -= 1
        assert_content += '\t' * self.tab_num + '}\n'
        inferShape_content += assert_content
        inferShape_content += self.__generate_intensor_des()
        inferShape_content += '\t' * self.tab_num + 'SVector<TensorDesc> outTensorDescs;\n'
        inferShape_content += '\t' * self.tab_num + f'Status s = {operation_name}->InferShape(inTensorDescs, outTensorDescs);\n'
        inferShape_content += '\t' * self.tab_num + 'if (s != 0) {\n'
        self.tab_num += 1
        inferShape_content += '\t' * self.tab_num + 'ATB_LOG(INFO) << "The error type is: " << FuzzUtil::errorType_.at(s);\n'
        self.tab_num -= 1
        inferShape_content += '\t' * self.tab_num + '}\n'
        inferShape_content += '\t' * self.tab_num + f'DestroyOperation({operation_name});\n'
        self.tab_num -= 1
        inferShape_content += '\t' * self.tab_num + '}\n'
        inferShape_content += '\t' * self.tab_num + 'DestroyOperation(nullptr);\n'
        inferShape_content += '\t' * self.tab_num + 'DT_FUZZ_END()\n'
        inferShape_content += '\t' * self.tab_num + 'SUCCEED();\n'
        self.tab_num -= 1
        inferShape_content += '\t' * self.tab_num + '}\n\n'
        return inferShape_content

    def __generate_setUpExecute(self):
        operation_name = self.operation_name[0].lower() + self.operation_name[1:]
        fuzz_type = 'SetUpExecute'
        setUpExecute_content = f'TEST({self.operation_name}DTFuzz, {fuzz_type})\n{{\n'
        self.tab_num += 1
        log_content = '\t' * self.tab_num + 'ATB_LOG(INFO) << "begin====================";\n'
        setUpExecute_content += '\t' * self.tab_num + 'std::srand(time(NULL));\n'
        setUpExecute_content += log_content
        setUpExecute_content += '\t' * self.tab_num + 'DT_Set_TimeOut_Second(30);\n'
        setUpExecute_content += self.__dt_fuzz_start(fuzz_type)
        setUpExecute_content += self.__generate_operation_fuzz()
        assert_content = '\t' * self.tab_num + f'if({operation_name} == nullptr) {{\n'
        self.tab_num += 1
        assert_content += '\t' * self.tab_num + 'continue;\n'
        self.tab_num -= 1
        assert_content += '\t' * self.tab_num + '}\n'
        setUpExecute_content += assert_content
        setUpExecute_content += self.__generate_intensor_des()
        setUpExecute_content += '\t' * self.tab_num + 'SVector<TensorDesc> outTensorDescs;\n'
        setUpExecute_content += '\t' * self.tab_num + f'Status s = FuzzUtil::SetupAndExecute({operation_name}, inTensorDescs, outTensorDescs);\n'
        setUpExecute_content += '\t' * self.tab_num + 'if (s != 0) {\n'
        self.tab_num += 1
        setUpExecute_content += '\t' * self.tab_num + 'ATB_LOG(INFO) << "The error type is: " << FuzzUtil::errorType_.at(s);\n'
        self.tab_num -= 1
        setUpExecute_content += '\t' * self.tab_num + '}\n'
        setUpExecute_content += '\t' * self.tab_num + f'DestroyOperation({operation_name});\n'
        self.tab_num -= 1
        setUpExecute_content += '\t' * self.tab_num + '}\n'
        setUpExecute_content += '\t' * self.tab_num + 'DestroyOperation(nullptr);\n'
        setUpExecute_content += '\t' * self.tab_num + 'DT_FUZZ_END()\n'
        setUpExecute_content += '\t' * self.tab_num + 'SUCCEED();\n'
        self.tab_num -= 1
        setUpExecute_content += '\t' * self.tab_num + '}\n'
        return setUpExecute_content

    def generate_fuzz_test(self, file_path):
        start_content = '\nnamespace atb {\n'
        inferShape_content = self.__generate_inferShape_test()
        setUpExecute_content = self.__generate_setUpExecute()
        end_content = '}'

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(f'{file_path}/test_{self.operation_name}.cpp', flags, modes), 'w') as file:
            file.write(HEAD)
            file.write(start_content)
            file.write(inferShape_content)
            file.write(setUpExecute_content)
            file.write(end_content)


if __name__ == '__main__':
    arguments = sys.argv
    INFER_OP_PARAM_PATH = f'{arguments[1]}/include/atb/infer_op_params.h'
    TRAIN_OP_PARAM_PATH = f'{arguments[1]}/include/atb/train_op_params.h'
    ALL_INFER_OPERATION_PARAMS = get_all_operation_params(INFER_OP_PARAM_PATH)
    ALL_TRAIN_OPERATION_PARAMS = get_all_operation_params(TRAIN_OP_PARAM_PATH)
    file_path = f'{arguments[1]}/tests/apitest/fuzztest'
    for op_name, operation_params in ALL_INFER_OPERATION_PARAMS.items():
        if 'Ffn' in op_name:
            continue
        infer_fuzz_test = OperationFuzzTestGenerator(f'{op_name}Operation', operation_params, int(arguments[2]), int(arguments[3]), 0, 'infer')
        infer_fuzz_test.generate_fuzz_test(file_path)

    for op_name, operation_params in ALL_TRAIN_OPERATION_PARAMS.items():
        train_fuzz_test = OperationFuzzTestGenerator(f'{op_name}Operation', operation_params, int(arguments[2]), int(arguments[3]), 0, 'train')
        train_fuzz_test.generate_fuzz_test(file_path)
