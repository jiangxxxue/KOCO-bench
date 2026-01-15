#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import logging
import stat
import argparse

CANN_COPYRIGHT = '''/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
'''

TYPE_MAP = {
    'SVector': 'AsdOps::SVECTOR_TYPE',
    'vector': 'AsdOps::VECTOR_TYPE'
}


def __parse_param_file(operation_param_file: str):
    struct_name = ''
    mem_list = []
    with open(operation_param_file) as fd:
        lines = fd.readlines()
        process_line = False
        end_line = False
        for org_line in lines:
            line = org_line.strip()
            line = line.strip(' ')
            fields = line.split(' ')
            if line.startswith('//'):
                continue
            if line.startswith('bool operator=='):
                end_line = True
            if end_line:
                break
            if line.startswith('struct'):
                process_line = True
            if not process_line:
                continue
            if line.startswith('enum'):
                continue
            if len(fields) == 3 and fields[0] == 'struct':
                struct_name = fields[1]
            elif line.find(';') != -1 and len(fields) >= 2:
                mem_list.append([fields[0], fields[1].strip(';').strip('{0,')])
            else:
                pass
    return struct_name, mem_list


def __parse_typename(typename: str):
    raw_type = typename.split('<')[0].rsplit('::')[-1]
    type_type = TYPE_MAP.get(raw_type, 'AsdOps::BASIC_TYPE')
    is_vector = (raw_type in TYPE_MAP.keys())
    if is_vector:
        decayed_type = typename.split('<')[-1].rsplit('>')[0]
    else:
        decayed_type = typename
    is_basic_type = not (decayed_type[0].isalpha() and decayed_type[0].isupper())
    is_complete_type = decayed_type.startswith('Mki::') or is_basic_type
    return (type_type, decayed_type, is_complete_type)


def __get_filtered_file_list(origin_file_list):
    filtered_file_list = []
    for file in origin_file_list:
        if file == 'params.h' or file == 'common.h' or not file.endswith('.h'):
            continue
        else:
            filtered_file_list.append(file)
    return filtered_file_list


def __generate_utils_cpp(op_params_dir: str, namespace: str, dest_file_path: str):
    op_func_list = []
    for path, _, file_list in os.walk(op_params_dir):
        filtered_files = __get_filtered_file_list(file_list)
        if not filtered_files:
            logging.error("cannot find param header file in directory: %s", os.path.realpath(op_params_dir))
            exit(1)
        for file in filtered_files:
            op_func_list.append(__parse_param_file(os.path.join(path, file)))
        
    with os.fdopen(os.open(dest_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                   'w') as fd:
        fd.write(CANN_COPYRIGHT)
        fd.write('#include <cstddef>\n')
        fd.write(f'#include "{namespace.lower()}/params/params.h"\n')
        fd.write(f'#include "sink_common.h"\n')
        for struct_name, mem_list in op_func_list:
            fd.write(f'\ntemplate<> size_t AsdOps::GetOffset<{namespace}::OpParam::{struct_name}>'
                     '(size_t i, uint64_t &type)\n')
            fd.write('{\n')
            if len(mem_list) == 0:
                fd.write('    (void)i;\n')
                fd.write('    type = AsdOps::UNDEFINED_TYPE;\n')
                fd.write('    return 0;\n')
            else:
                fd.write(f'    static const size_t offsets[{len(mem_list)}] = {{\n')
                for i, member in enumerate(mem_list):
                    domain = member[1].split('[', 1)[0]
                    fd.write(f'        offsetof({namespace}::OpParam::{struct_name}, {domain}),  // {i}\n')
                fd.write('    };\n')
                fd.write(f'    static const uint64_t types[{len(mem_list)}] = {{\n')
                for member in mem_list:
                    type_type, decayed_type, is_complete_type = __parse_typename(member[0])
                    type_name = decayed_type if is_complete_type else   \
                                f'{namespace}::OpParam::{struct_name}::{decayed_type}'
                    fd.write(f'        ({type_type} | (sizeof({type_name}) << AsdOps::SHIFT_BITS)),\n')
                fd.write('    };\n')
                fd.write('    type = types[i];\n')
                fd.write('    return offsets[i];\n')
            fd.write('}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, required=True)
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--dest_file_path', type=str, required=True)
    input_args = parser.parse_args()
    current_dir = os.path.split(os.path.realpath(__file__))[0]
    __generate_utils_cpp(input_args.params_path, input_args.namespace, input_args.dest_file_path)
