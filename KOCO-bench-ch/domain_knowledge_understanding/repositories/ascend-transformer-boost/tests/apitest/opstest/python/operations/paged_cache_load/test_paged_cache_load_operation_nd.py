import json
import math
import os
import random
import sys
import unittest
import logging
import numpy as np
import torch
import torch_npu
random.seed(3)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

MAX_SEQ_LEN = 1024

shape_dtype_list = [
    # batch|seq_len|num_heads|head_size_k|head_size_v|block_size|num_blocks|dtype
    [16,1,16,144,128,128,128,"float16"],
    [16,1,16,144,128,128,128,"bfloat16"],
    [16,1,16,144,128,128,128,"int8"],
    
    [32,1,16,192,128,64,256,"float16"],
    [32,1,16,192,128,64,256,"bfloat16"],
    [32,1,16,192,128,64,256,"int8"],
]

# ND:type=0
configs = [
    {"kvCacheCfg": 1, "isSeqLensCumsumMode": True, "hasSeqStarts": True},
    {"kvCacheCfg": 1, "isSeqLensCumsumMode": False, "hasSeqStarts": True},
    {"kvCacheCfg": 1, "isSeqLensCumsumMode": False, "hasSeqStarts": False},
    {"kvCacheCfg": 1, "isSeqLensCumsumMode": True, "hasSeqStarts": False},
]

type_list_numpy = {"float16":np.float16 , "bfloat16": np.float32, "int8": np.int8, "int32": np.int32}
type_list_torch = {"float16":torch.float16 , "bfloat16": torch.bfloat16, "int8": np.int8, "int32": torch.int32}

format_dict = {"undefined": -1, "nchw": 0, "nhwc": 1, "nd": 2, "nc1hwc0": 3,
                "fractal_z": 4, "nc1hwc0_c04": 12, "hwcn": 16, "ndhwc": 27,
                "fractal_nz": 29, "ncdhw": 30, "ndc1hwc0": 32, "fractal_z_3d": 33}

def generate_data(
        batch,
        seq_len,
        num_heads,
        head_size_k,
        head_size_v,
        block_size,
        num_blocks,
        dtype,
        param_dict
):
    soc_version = operation_test.get_soc_version()
    isSeqLensCumsumMode = param_dict.get('isSeqLensCumsumMode')
    hasSeqStarts = param_dict.get('hasSeqStarts')
    num_tokens = batch * seq_len
    key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(type_list_numpy[dtype])
    value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(type_list_numpy[dtype])
    if isSeqLensCumsumMode:
        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
        max_context_len = max(context_lens)
        cu_context_lens = [0]
        for elem in context_lens:
            cu_context_lens.append(cu_context_lens[-1] + elem)

        max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size + 4
        block_tables = []  # [num_tokens, max_num_blocks_per_seq]
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
            ]
            block_tables.append(block_table)
        #--------------seq_starts-------------------
        seq_starts = []
        if hasSeqStarts:
            seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]
        else:
            seq_starts = [0 for _ in range(num_tokens)]
        seq_starts = np.array(seq_starts).astype(np.int32)
        #-------------------------------------------
        block_tables = np.array(block_tables).astype(np.int32)
        context_lens = np.array(cu_context_lens).astype(np.int32)
        
        sum_context_lens = context_lens[-1]
        
        key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(type_list_numpy[dtype])
        value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(type_list_numpy[dtype])
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(type_list_numpy[dtype])
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(type_list_numpy[dtype])
        print(f"key_expect: {key_expect.shape}")
        kv_rslt_id = 0
        context_start = 0
        for i in range(num_tokens):
            block_table = block_tables[i]
            context_end = int(context_lens[i+1])
            context_len = context_end - context_start
            context_start = context_end
            #--------------seq_starts-------------------
            block_table_offset = 0
            if hasSeqStarts:
                block_table_offset = seq_starts[i] // block_size
            #-------------------------------------------
            for j in range(context_len):
                block_id = int(block_table[block_table_offset + j // block_size])
                block_offset = j % block_size

                if block_id < 0:
                    continue

                temp_k = key_cache[block_id][block_offset]
                temp_v = value_cache[block_id][block_offset]

                key_expect[kv_rslt_id] = temp_k
                value_expect[kv_rslt_id] = temp_v
                kv_rslt_id += 1
                
        ret_data = key_cache, value_cache, block_tables, context_lens, key, value, seq_starts, key_expect, value_expect
        return ret_data
    else:
        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
        max_context_len = max(context_lens)

        max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size + 4
        block_tables = []  # [num_tokens, max_num_blocks_per_seq]
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
            ]
            block_tables.append(block_table)
        #--------------seq_starts-------------------
        seq_starts = []
        if hasSeqStarts:
            seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]
        else:
            seq_starts = [0 for _ in range(num_tokens)]
        seq_starts = np.array(seq_starts).astype(np.int32)
        #-------------------------------------------
        block_tables = np.array(block_tables).astype(np.int32)
        context_lens = np.array(context_lens).astype(np.int32)
        
        sum_context_lens = sum(context_lens)
        
        key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(type_list_numpy[dtype])
        value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(type_list_numpy[dtype])
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(type_list_numpy[dtype])
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(type_list_numpy[dtype])
        print(f"key_expect: {key_expect.shape}")
        kv_rslt_id = 0
        for i in range(num_tokens):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            #--------------seq_starts-------------------
            block_table_offset = 0
            if hasSeqStarts:
                block_table_offset = seq_starts[i] // block_size
            #-------------------------------------------
            for j in range(context_len):
                block_id = int(block_table[block_table_offset + j // block_size])
                block_offset = j % block_size

                if block_id < 0:
                    continue

                temp_k = key_cache[block_id][block_offset]
                temp_v = value_cache[block_id][block_offset]

                key_expect[kv_rslt_id] = temp_k
                value_expect[kv_rslt_id] = temp_v
                kv_rslt_id += 1
    
        ret_data = key_cache, value_cache, block_tables, context_lens, key, value, seq_starts, key_expect, value_expect
        return ret_data
    
    
class PagedCacheLoadOperation(operation_test.OperationTest):

    def golden_calc(self, input_tensors):
        return [self.in_tensors[7], self.in_tensors[8]]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        result = []
        for i in range(len(out_tensor)):
            actual_output = out_tensor[i]
            golden_output = golden_out_tensor[i]
            result.append(torch.equal(actual_output, golden_output))
        logging.info(f"result is {all(result)}")
        return all(result)
    
    def _run_test(self, 
                    shape_dtype_idx: int, 
                    param_idx: int,
                    format_type: str = None, 
                    print_param: bool = False):
            """
            Args:
                shape_dtype_idx: shape_dtype_list的索引
                param_idx: JSON序列化的参数配置（新增参数）
                dtype_conversion: 需要转换的dtype配置 {index: dtype}
                format_type: 指定格式类型（None表示根据参数自动判断）
                print_param: 是否打印参数信息
            """
            not_support_device = ['Ascend910A','Ascend310B','Ascend310P']
            # 获取Soc型号
            if operation_test.get_soc_version() in not_support_device:
                print("These test cases only support A2/A3")
                return True
            # OP NAME
            OP_NAME = "PagedCacheLoadOperation"
            param = json.dumps(configs[param_idx])
            # bf16 类型需要做转换
            indices_to_convert = [0, 1, 4, 5]
            # 参数有效性校验
            try:
                param_dict = json.loads(param)
                if param_dict.get('kvCacheCfg') not in [0, 1]:
                    raise ValueError("Invalid type parameter in PARAM")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for PARAM")

            # 参数打印
            if print_param:
                print(f"PARAM: {param_dict}")
        
            # 数据生成与类型转换
            data = generate_data(*shape_dtype_list[shape_dtype_idx], param_dict)
            in_tensors = [torch.from_numpy(tensor) for tensor in data]

            # 类型转换
            data_type = shape_dtype_list[shape_dtype_idx][-1]
            if indices_to_convert and data_type == "bfloat16":
                in_tensors = [tensor.to(torch.bfloat16) if i in indices_to_convert else tensor for i, tensor in enumerate(in_tensors)]
            # 设备转换
            self.in_tensors = [tensor.npu() for tensor in in_tensors]

            # 格式转换
            format = format_type if format_type else ("fractal_nz" if param_dict.get('type') == 0 else "nd")
            if format in format_dict:
                self.in_tensors[0] = torch_npu.npu_format_cast(self.in_tensors[0], format_dict[format])
                self.in_tensors[1] = torch_npu.npu_format_cast(self.in_tensors[1], format_dict[format])

            # 调试输出
            _ = [print(f"Tensor {i}: {tensor.dtype}, {tensor.device}, {torch.npu.synchronize()}") 
                for i, tensor in enumerate(self.in_tensors)]

            # 执行测试
            self.execute_out(
                OP_NAME,
                param,  # 使用传入的PARAM参数
                self.in_tensors[:7],
                self.in_tensors[4:6]
            )
#================================================Test=====================================================
    def test_fp16(self):
        self._run_test(shape_dtype_idx=0, param_idx=0, print_param=True)
    def test_bf16(self):
        self._run_test(shape_dtype_idx=1, param_idx=0, print_param=True)
    def test_int8(self):
        self._run_test(shape_dtype_idx=2, param_idx=0, print_param=True)
    def test_fp16_1(self):
        self._run_test(shape_dtype_idx=3, param_idx=0, print_param=True)
    def test_bf16_1(self):
        self._run_test(shape_dtype_idx=4, param_idx=0, print_param=True)
    def test_int8_1(self):
        self._run_test(shape_dtype_idx=5, param_idx=0, print_param=True)
#=======================================isSeqLensCumsumMode False============================================
    def test_fp16_2(self):
        self._run_test(shape_dtype_idx=0, param_idx=1, print_param=True)
    def test_bf16_2(self):
        self._run_test(shape_dtype_idx=1, param_idx=1, print_param=True)
    def test_int8_2(self):
        self._run_test(shape_dtype_idx=2, param_idx=1, print_param=True)
#=======================================hasSeqStarts False===================================================
    def test_fp16_3(self):
        self._run_test(shape_dtype_idx=0, param_idx=3, print_param=True)
    def test_bf16_3(self):
        self._run_test(shape_dtype_idx=1, param_idx=3, print_param=True)
    def test_int8_3(self):
        self._run_test(shape_dtype_idx=2, param_idx=3, print_param=True)
#=======================================hasSeqStarts False && isSeqLensCumsumMode=False======================
    def test_fp16_4(self):
        self._run_test(shape_dtype_idx=0, param_idx=2, print_param=True)
    def test_bf16_4(self):
        self._run_test(shape_dtype_idx=1, param_idx=2, print_param=True)
    def test_int8_4(self):
        self._run_test(shape_dtype_idx=2, param_idx=2, print_param=True)

if __name__ == '__main__':
    unittest.main()