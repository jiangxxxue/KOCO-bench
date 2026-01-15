# -*- coding: utf-8 -*-
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import argparse
import configparser
import json
import logging
import os
import stat
from collections import namedtuple

from build_util import get_build_target_list

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

JsonSpecification = namedtuple(
    "JsonSpecification", ["mode", "inputs", "outputs", "attrs", "dir", "deterministic"])

TacticDef = namedtuple("TacticDef", [
    "ops_name", "operation", "input_num", "output_num", "dtypes_in",
    "dtypes_out", "formats_in", "formats_out", "mode", "attrs", "soc_support", "deterministic"])



def get_code_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def get_tbe_kernel_path():
    result = True
    tbe_kernel_path = os.getenv("ASCEND_KERNEL_PATH")
    if not os.path.exists(tbe_kernel_path):
        result = False
    return tbe_kernel_path, result


def get_build_cache_path():
    result = True
    build_cache_dir = os.getenv("CACHE_DIR")
    if not os.path.exists(build_cache_dir):
        result = False
    return build_cache_dir, result


def read_tbe_config_file(input_args):
    tbe_config_ini, result = None, True
    if not os.path.exists(input_args.src_ini_path):
        result = False
        logging.error("ini file: %s not exist!", input_args.src_ini_path)
        return tbe_config_ini, result
    tbe_config_ini = configparser.RawConfigParser()
    tbe_config_ini.optionxform = lambda option: option
    try:
        tbe_config_ini.read(input_args.src_ini_path)
    except configparser.MissingSectionHeaderError:
        result = False
        logging.error("ini file: %s format error!", input_args.src_ini_path)
    except configparser.ParsingError:
        result = False
        logging.error("ini file: %s format error!", input_args.src_ini_path)
    return tbe_config_ini, result


def read_tbe_json_file(json_file_path):
    result = True
    ops_specification_list = []
    try:
        json_list = os.listdir(json_file_path)
        for file_name in json_list:
            if not file_name.endswith(".json") or file_name.endswith("failed.json"):
                continue
            json_file = os.path.join(json_file_path, file_name)
            with open(json_file) as f:
                text = json.load(f)
                item = text["supportInfo"]
                inputs = item["inputs"]
                outputs = item["outputs"]
                mode = item["implMode"] if "implMode" in item else None
                attrs = item["attrs"] if "attrs" in item else None
                deterministic = item["deterministic"] if "deterministic" in item else None
                json_info = JsonSpecification(
                    mode=mode, inputs=inputs, outputs=outputs, attrs=attrs, dir=file_name, deterministic=deterministic)
                ops_specification_list.append(json_info)
    except FileNotFoundError:
        logging.error("file %s is not found!", json_file)
        result = False
    except json.decoder.JSONDecodeError:
        logging.error("file %s is not json file!", json_file)
        result = False
    except KeyError:
        logging.error("keyerror in file %s!", json_file)
        result = False
    return ops_specification_list, result


def impl_mode_matched_or_not(json_mode, tactic_mode):
    if not json_mode:
        return True
    if not tactic_mode and "high_precision" not in json_mode:
        return False
    if tactic_mode and tactic_mode not in json_mode:
        return False
    return True


def check_dtype_matched_or_not(data_defs, data_dtypes):
    try:
        for i, dtype in enumerate(data_dtypes):
            if dtype == "" and data_defs[i] is None:
                continue
            if data_defs[i] is None:
                return False
            dtypes = dtype.split("/")
            tensor_dtype_not_match = isinstance(data_defs[i], dict) and data_defs[i]["dtype"] not in dtypes
            tensorlist_dtype_not_match = isinstance(data_defs[i], list) and data_defs[i][0]["dtype"] not in dtypes
            if tensor_dtype_not_match or tensorlist_dtype_not_match:
                return False
    except IndexError:
        return False
    except TypeError:
        return False
    return True


def check_format_matched_or_not(data_defs, data_formats):
    try:
        if not data_formats:
            for data_def in data_defs:
                if data_def is None:
                    continue
                tensor_format_not_nd = isinstance(data_def, dict) and data_def["format"] != "ND"
                tensorlist_format_not_nd = isinstance(data_def, list) and data_def[0]["format"] != "ND"
                if tensor_format_not_nd or tensorlist_format_not_nd:
                    return False
        else:
            for i, dformat in enumerate(data_formats):
                if data_defs[i] is None:
                    return False
                tensor_format_not_match = isinstance(data_defs[i], dict) and data_defs[i]["format"] != dformat
                tensorlist_format_not_match = isinstance(data_defs[i], list) and data_defs[i][0]["format"] != dformat
                if tensor_format_not_match or tensorlist_format_not_match:
                    return False
    except IndexError:
        return False
    except TypeError:
        return False
    return True


def inputs_outputs_matched_or_not(data_defs, data_num, data_dtypes, data_formats):
    if len(data_defs) == 0 and data_num == 0:
        return True
    if len(data_defs) == 1 and "name" not in data_defs[0]:
        # input paramType is dynamic
        data_defs = data_defs[0]

    if len(data_defs) != data_num:
        return False

    return check_dtype_matched_or_not(data_defs, data_dtypes) and check_format_matched_or_not(data_defs, data_formats)


def attrs_matched_or_not(json_attrs, tactic_attrs):
    if not tactic_attrs:
        return True
    for i, attr in enumerate(tactic_attrs):
        try:
            if attr != str(json_attrs[i]["value"]):
                return False
        except IndexError:
            return False
        except TypeError:
            return False
    return True


def deterministic_matched_or_not(json_deterministic, tactic_deterministic):
    if json_deterministic == "ignore":
        return True
    if json_deterministic.lower() == tactic_deterministic.lower():
        return True
    else:
        return False


def get_match_json(json_info_dir, tactic_info):
    result = False
    match_json_dir = ""
    ops_specification_list, ret = read_tbe_json_file(json_info_dir)
    if not ret:
        return match_json_dir, result
    count_check = 0
    for json_info in ops_specification_list:
        matched = impl_mode_matched_or_not(json_info.mode, tactic_info.mode) \
                  and inputs_outputs_matched_or_not(
            json_info.inputs, tactic_info.input_num, tactic_info.dtypes_in, tactic_info.formats_in) \
                  and inputs_outputs_matched_or_not(
            json_info.outputs, tactic_info.output_num, tactic_info.dtypes_out, tactic_info.formats_out) \
                  and attrs_matched_or_not(json_info.attrs, tactic_info.attrs) \
                  and deterministic_matched_or_not(json_info.deterministic, tactic_info.deterministic)

        if matched:
            match_json_dir, result = json_info.dir, True
            count_check += 1

    if count_check != 1:
        logging.error(
            f"{json_info_dir}: matched json file number is {count_check}, which should be 1")
        result = False
    return match_json_dir, result


def get_tbe_tactic_json(input_args, tbe_config_ini):
    result = True
    json_paths_info = configparser.ConfigParser()
    json_paths_info.optionxform = lambda option: option

    try:
        json_paths_info.read(input_args.dst_ini_path)
    except configparser.MissingSectionHeaderError:
        result = False
        logging.error("ini file: %s format error!", input_args.dst_ini_path)
    except configparser.ParsingError:
        result = False
        logging.error("ini file: %s format error!", input_args.dst_ini_path)

    tbe_kernel_path, ret = get_tbe_kernel_path()
    if not ret:
        result = False
        logging.error("get tbe kernel path failed")
        return json_paths_info, result
    build_cache_dir, ret = get_build_cache_path()
    if not ret:
        result = False
        logging.error("get build cache dir failed")
        return json_paths_info, result
    build_cache_obj_dir = os.path.join(build_cache_dir, "obj")
    target_version_list = get_build_target_list()
    logging.info("target version list: %s", target_version_list)
    for target_version in target_version_list:
        try:
            for tactic_name in tbe_config_ini.sections():
                try:
                    ops = tbe_config_ini.get(tactic_name, "ops")
                    operation_name = tbe_config_ini.get(
                        tactic_name, "operationName")
                    input_num = int(tbe_config_ini.get(
                        tactic_name, "inputCount"))
                    output_num = int(tbe_config_ini.get(
                        tactic_name, "outputCount"))
                    input_dtypes = tbe_config_ini.get(tactic_name, "dtypeIn")
                    output_dtypes = tbe_config_ini.get(tactic_name, "dtypeOut")
                    input_formats = tbe_config_ini.get(
                        tactic_name, "formatIn", fallback=None)
                    output_formats = tbe_config_ini.get(
                        tactic_name, "formatOut", fallback=None)
                    mode = tbe_config_ini.get(
                        tactic_name, "mode", fallback=None)
                    attrs = tbe_config_ini.get(
                        tactic_name, "attrs", fallback=None)
                    soc_support = tbe_config_ini.get(tactic_name, "socSupport", fallback=None)
                    deterministic = tbe_config_ini.get(
                        tactic_name, "deterministic", fallback='ignore')
                except configparser.NoOptionError:
                    logging.error("configparser option is not found: %s", tactic_name)
                    continue
                except ValueError:
                    logging.error("string-to-int failed!")
                    continue
                except configparser.InterpolationError:
                    result = False
                    logging.error("invalid interpolation syntax!")
                    break
                except configparser.Error as e:
                    result = False
                    logging.error("Error: %s", e)
                    break

                input_dtype_arr = input_dtypes.split(",")
                output_dtype_arr = output_dtypes.split(",")
                input_format_arr = input_formats.split(
                    ",") if input_formats else None
                output_format_arr = output_formats.split(
                    ",") if output_formats else None
                attr_arr = attrs.split(',') if attrs else None

                tactic_info = TacticDef(ops_name=ops, operation=operation_name,
                                        input_num=input_num, output_num=output_num,
                                        dtypes_in=input_dtype_arr, dtypes_out=output_dtype_arr,
                                        formats_in=input_format_arr, formats_out=output_format_arr,
                                        mode=mode, attrs=attr_arr, soc_support=soc_support,
                                        deterministic=deterministic)
                if tactic_info.soc_support and target_version not in tactic_info.soc_support.split(","):
                    continue
                json_info_dir = os.path.join(
                    tbe_kernel_path, target_version, ops)
                match_json_dir, ret = get_match_json(
                    json_info_dir, tactic_info)
                if not ret:
                    logging.error(
                        f"[{target_version}] get tactic failed: {tactic_name}")
                    exit(1)
                if not json_paths_info.has_section(operation_name):
                    json_paths_info.add_section(operation_name)
                json_paths_info.set(
                    operation_name, tactic_name + "." + target_version,
                    os.path.join(ops, match_json_dir))
                print(os.path.join(build_cache_obj_dir, target_version, operation_name,
                                   match_json_dir)[:-5] + '_' + tactic_name.lower() + '.cpp')

        except configparser.NoSectionError:
            result = False
            logging.error("configparser section is not found")
        except configparser.Error as e:
            result = False
            logging.error("Error: %s", e)
    return json_paths_info, result


def write_tbe_tactic_json(input_args, json_paths_info):
    fd = os.open(input_args.dst_ini_path, os.O_WRONLY | os.O_CREAT |
                 os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR)
    with os.fdopen(fd, 'w+') as f:
        try:
            json_paths_info.write(f, space_around_delimiters=False)
        except configparser.Error as e:
            logging.error("Error: %s", e)
            return
    logging.info(f"write {input_args.dst_ini_path} success")


def main():
    code_root_dir = get_code_root()
    tactic_info_path = os.path.join(code_root_dir, "src/kernels/configs/ops/tbe_tactic_info.ini")
    
    build_cache_dir, _ = get_build_cache_path()
    tactic_json_path = os.path.join(build_cache_dir, "tbe_tactic_json.ini")

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ini_path', type=str, required=False,
                        default=tactic_info_path)
    parser.add_argument('--dst_ini_path', type=str, required=False,
                        default=tactic_json_path)
    input_args = parser.parse_args()

    tbe_config_ini, ret = read_tbe_config_file(input_args)
    if not ret:
        logging.error("get tbe tactic info failed!")
        exit(1)
    json_paths_info, ret = get_tbe_tactic_json(input_args, tbe_config_ini)
    if not ret:
        logging.error("get tbe tactic json failed!")
        exit(1)
    write_tbe_tactic_json(input_args, json_paths_info)


if __name__ == "__main__":
    main()
