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
import yaml
import random
import logging
import argparse
import json
import pandas as pd
from typing import List
from pydantic import BaseModel
import testdata_check_modify

DEFAULT_DTYPES = ["float", "float16", "int8", "int32", "uint8", "int16", "uint16", "utin32",
    "int64", "uint64", "double", "bool", "string", "complex64", "complex128", "bf16"]
DEFAULT_FORMATS = ["undefined", "nchw", "nhwc", "nd", "nc1hwc0",
                "fractal_z", "nc1hwc0_c04", "hwcn", "ndhwc",
                "fractal_nz", "ncdhw", "ndc1hwc0", "fractal_z_3d"]
DEFAULT_RANGES_VALUES = ["-1, 1", "-7, 7"]
DEFAULT_TYPES_VALUES = ["customize"]
DEFAULT_DIM_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_DIM_WEIGHTS = [0.05, 0.2, 0.2, 0.2, 0.2, 0.01, 0.01, 0.01]
DEFAULT_DIM_VALUE_RANGES = [
    [1],
    [7, 9],
    [15, 17],
    [19, 21],
    [255, 257],
    [131073],
    [2147483648],
    [1, 1024],
    [1025, 10240],
    [10241, 102400],
    [102401, 1024000],
    [1024001, 10240000],
    [10240001, 102400000],
    [102400001, 1024000000],
    [1024000001, 2147483648]
]
DEFAULT_DIM_VALUE_WEIGHTS = [
    0.05,
    0.1,
    0.1,
    0.1,
    0.01,
    0.01,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001
]

class OpParam(BaseModel):
    key: str
    values: List

class Shape(BaseModel):
    dim_numbers: List[int] = DEFAULT_DIM_NUMBERS
    dim_number_weights: List[float] = DEFAULT_DIM_WEIGHTS
    dim_values: List[List[int]] = DEFAULT_DIM_VALUE_RANGES
    dim_value_weights: List[float] = DEFAULT_DIM_VALUE_WEIGHTS

    def generate_one_shape(self, op_name, op_param):
        dim_number = random.choices(self.dim_numbers, weights=self.dim_number_weights)
        logging.debug("dim_number: %s", dim_number)
        dim_value_range_list = random.choices(self.dim_values, weights=self.dim_value_weights, k = dim_number[0])
        logging.debug("dim_value_range_list: %s", dim_value_range_list)
        dim_value_list = []
        for range in dim_value_range_list:
            dim_value_list.append(random.randint(range[0], range[-1]))
        in_shape_check_modify_func = 'testdata_check_modify.' + op_name + '.in_shape_check_modify'
        if not eval(in_shape_check_modify_func)(op_param, dim_value_list):
            return self.generate_one_shape(op_name, op_param)
        else:
            logging.debug("dim_value_list: %s", dim_value_list)
            return dim_value_list.copy()

class CaseConfig(BaseModel):
    op_name: str
    op_param: List[OpParam] = None
    in_num: List[int]
    in_dtype: List[str] = None
    in_format: List[str] = None
    in_shape: Shape = None
    data_gen_range: List[str] = None
    data_gen_type: List[str] = None
    op_param_combinations: List[dict] = []

    def __init__(self, yaml_content):
        super().__init__(**yaml_content)
        self.in_dtype = self.in_dtype if self.in_dtype != None else DEFAULT_DTYPES
        self.in_format = self.in_format if self.in_format != None else DEFAULT_FORMATS
        self.in_shape = self.in_shape if self.in_shape != None else Shape()
        self.data_gen_range = self.data_gen_range if self.data_gen_range != None else DEFAULT_RANGES_VALUES
        self.data_gen_type = self.data_gen_type if self.data_gen_type != None else DEFAULT_TYPES_VALUES
        self.__generate_op_param_combinations()

    def __generate_combinations(self, list, cur_combination, cur_pos, length):
        if (cur_pos == length):
            logging.debug("cur_pos %d, cur_combination: %s", cur_pos, cur_combination)
            op_param_check_modify_func = 'testdata_check_modify.' + self.op_name + '.op_param_check_modify'
            if eval(op_param_check_modify_func)(cur_combination):
                self.op_param_combinations.append(cur_combination.copy())
            return
        for i in range(len(list[cur_pos].values)):
            cur_combination[list[cur_pos].key] = list[cur_pos].values[i]
            self.__generate_combinations(list, cur_combination, cur_pos + 1, length)

    def __generate_op_param_combinations(self):
        if self.op_param == None:
            self.op_param_combinations.append({})
        else:
            self.__generate_combinations(self.op_param, {}, 0, len(self.op_param))
        logging.debug("all op_param combinations: %s", self.op_param_combinations)

class OperationCase:
    def __init__(self, output_file_path, args):
        self.operation_case = pd.DataFrame(columns=['CaseNum', 'CaseName', 'OpName', 'OpParam', 'InNum', 'InDType',
                                                    'InFormat', 'InShape', 'OutNum', 'OutDType', 'OutFormat',
                                                    'OutShape', 'DataGenType', 'DataGenRange', 'InTensorFile',
                                                    'OutTensorFile', 'TestType', 'TestLevel', 'FromModel', 'SocVersion',
                                                    'ExpectedError'])

        self.index = 0
        self.output_file_path = output_file_path
        self.args = args

    def load_data_from_case_config(self, case_config: CaseConfig):
        self.op_name = case_config.op_name
        self.op_param_list = case_config.op_param_combinations
        self.in_num_list = case_config.in_num
        self.in_dtype_list = case_config.in_dtype
        self.in_format_list = case_config.in_format
        self.data_gen_range_list = case_config.data_gen_range
        self.data_gen_type_list = case_config.data_gen_type
        self.in_shape = case_config.in_shape

    def __generate_one_dtype(self, op_param):
        dtype = random.choices(self.in_dtype_list)[0]
        in_dtype_check_modify_func = 'testdata_check_modify.' + self.op_name + '.in_dtype_check_modify'
        if not eval(in_dtype_check_modify_func)(op_param, dtype):
            return self.__generate_one_dtype(op_param)
        else:
            return dtype

    def __generate_one_format(self, op_param):
        format = random.choices(self.in_format_list)[0]
        in_format_check_modify_func = 'testdata_check_modify.' + self.op_name + '.in_format_check_modify'
        if not eval(in_format_check_modify_func)(op_param, format):
            return self.__generate_one_format(op_param)
        else:
            return format

    def __generate_tensor_desc(self, op_param, in_num):
        self.tensor_desc = testdata_check_modify.TensorDesc()
        self.tensor_desc.nums = in_num
        for i in range(in_num):
            self.tensor_desc.dtypes.append(self.__generate_one_dtype(op_param))
            self.tensor_desc.formats.append(self.__generate_one_format(op_param))
            self.tensor_desc.shapes.append(self.in_shape.generate_one_shape(self.op_name, op_param))
            self.tensor_desc.data_gen_ranges.append(random.choices(self.data_gen_range_list)[0])
            self.tensor_desc.data_gen_types.append(random.choices(self.data_gen_type_list)[0])
        tensor_desc_check_modify_func = 'testdata_check_modify.' + self.op_name + '.tensor_desc_check_modify'
        input_data_volumn_check_func = 'testdata_check_modify.' + self.op_name + '.input_data_volumn_check'
        if not (eval(input_data_volumn_check_func)(op_param, self.tensor_desc) and eval(tensor_desc_check_modify_func)(op_param, self.tensor_desc)):
            self.__generate_tensor_desc(op_param, in_num)

    def generate_cases(self):
        while (self.index < self.args.number):
            self.__generate_cases_one_round()
        logging.info(f"{self.op_name} has {self.index} testcases generated")

    def __generate_cases_one_round(self):
        in_num_check_modify_func = 'testdata_check_modify.' + self.op_name + '.in_num_check_modify'
        for op_param in self.op_param_list:
            for in_num in self.in_num_list:
                if not eval(in_num_check_modify_func)(op_param, in_num):
                    continue
                self.index += 1                            
                self.__generate_fixed_data(op_param, in_num)
                if in_num == 0:
                    self.operation_case.loc[self.index, 'InDType'] = ' '
                    self.operation_case.loc[self.index, 'InFormat'] = ' '
                    self.operation_case.loc[self.index, 'InShape'] = ' '
                    self.operation_case.loc[self.index, 'DataGenRange'] = ' '
                    self.operation_case.loc[self.index, 'DataGenType'] = ' '
                else:
                    self.__generate_tensor_desc(op_param, in_num)
                    self.operation_case.loc[self.index, 'InDType'] = ';'.join(self.tensor_desc.dtypes)
                    self.operation_case.loc[self.index, 'InFormat'] = ';'.join(self.tensor_desc.formats)
                    self.operation_case.loc[self.index, 'InShape'] = ';'.join(','.join(str(d) for d in shape) for shape in self.tensor_desc.shapes)
                    self.operation_case.loc[self.index, 'DataGenRange'] = ';'.join(self.tensor_desc.data_gen_ranges)
                    self.operation_case.loc[self.index, 'DataGenType'] = ';'.join(self.tensor_desc.data_gen_types)

    def __generate_fixed_data(self, op_param, in_num):
        self.operation_case.loc[self.index, 'CaseNum'] = self.index
        self.operation_case.loc[self.index, 'CaseName'] = self.op_name + str(self.index)
        self.operation_case.loc[self.index, 'OpName'] = self.op_name
        self.operation_case.loc[self.index, 'OpParam'] = json.dumps(op_param)
        self.operation_case.loc[self.index, 'InNum'] = in_num
        self.operation_case.loc[self.index, 'OutNum'] = 'reserved'
        self.operation_case.loc[self.index, 'OutDType'] = 'reserved'
        self.operation_case.loc[self.index, 'OutFormat'] = 'reserved'
        self.operation_case.loc[self.index, 'OutShape'] = 'reserved'
        self.operation_case.loc[self.index, 'InTensorFile'] = ' '
        self.operation_case.loc[self.index, 'OutTensorFile'] = ' '
        self.operation_case.loc[self.index, 'TestType'] = 'Generalization'
        self.operation_case.loc[self.index, 'TestLevel'] = ' '
        self.operation_case.loc[self.index, 'FromModel'] = ' '
        self.operation_case.loc[self.index, 'SocVersion'] = self.args.soc_version
        self.operation_case.loc[self.index, 'ExpectedError'] = 'NO_ERROR'

    def save_cases_to_csv(self):
        max_widths = [max([len(str(row[i])) for row in self.operation_case.values] + [len(str(self.operation_case.columns[i]))]) for i in range(len(self.operation_case.columns))]
        if not os.path.exists(self.output_file_path):
            os.makedirs(self.output_file_path)
        output_file = self.output_file_path + "/" + self.op_name + "_Generalization_TestCase.csv"
        with open(output_file, 'w') as f:
            header = [str(col).ljust(max_widths[i]) for i, col in enumerate(self.operation_case.columns)]
            f.write('|'.join(header) + '\n')
            for row in self.operation_case.values:
                line = [str(col).ljust(max_widths[i]) for i, col in enumerate(row)]
                f.write('|'.join(line) + '\n')
        logging.info(self.op_name + " generalization csvopstest file is saved to " + output_file)

class YamlConfig:
    def __init__(self, yaml_file, output_file_path, operation_name, args):
        f = open(yaml_file, "r")
        self.yaml_contents = yaml.safe_load_all(f)
        self.output_file_path = output_file_path
        self.operation_name = operation_name
        self.args = args

    def generate_all_operation_cases(self):
        for i, yaml_content in enumerate(self.yaml_contents):
            logging.debug("yaml config number: %s", str(i + 1))
            logging.debug("yaml config content: %s", yaml_content)
            if self.operation_name != '' and yaml_content['op_name'] != self.operation_name:
                logging.debug("Operation case {} will not be generated because selected operation case is {}".format(yaml_content['op_name'], self.operation_name))
                continue
            case_config = CaseConfig(yaml_content)
            operation_case = OperationCase(self.output_file_path, self.args)
            operation_case.load_data_from_case_config(case_config)
            operation_case.generate_cases()
            operation_case.save_cases_to_csv()

def set_debug_level(level_param):
    if (level_param == 'info'):
        logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    elif (level_param == 'debug'):
        logging.basicConfig(level=logging.DEBUG,format='[%(asctime)s] [%(levelname)s] [:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ascend-transformer-boost generalization csvopstest case generator."
                                     "if you encounter 'ModuleNotFoundError', run 'pip3 install -r requirements.txt'.")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-i', '--input', default='./operation_testdata_config.yaml', help='the input yamlfile to be processed')
    parser.add_argument('-o', '--output', default='./output', help='the output path containing generated csvopstest files.')
    parser.add_argument('-l', '--level', default='info', choices=['info', 'debug'], help='log level')
    parser.add_argument('-op', '--operation_name', default='', help='the csv of which operation that will be generated')
    parser.add_argument('-s', '--soc_version', default='', choices=['Ascend310P', 'Ascend910B'], help='generate testcases of specific ascend soc platform')
    parser.add_argument('-n', '--number', default=700, type=int, help='the total number of generated cases for one operation')
    args = parser.parse_args()
    yaml_file = args.input
    output_file_path = args.output
    operation_name = args.operation_name
    set_debug_level(args.level)

    # main process
    logging.info("-------------------------------------AtbCsvopstest Case Generator Begins-------------------------------------")
    logging.info("Case is running in path: " + os.getcwd())
    yaml_config = YamlConfig(yaml_file, output_file_path, operation_name, args)
    yaml_config.generate_all_operation_cases()
    logging.info("--------------------------------------AtbCsvopstest Case Generator Ends--------------------------------------")
