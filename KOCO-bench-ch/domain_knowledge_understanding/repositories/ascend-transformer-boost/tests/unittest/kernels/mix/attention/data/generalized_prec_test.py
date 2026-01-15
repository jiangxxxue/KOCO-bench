# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import logging
import sys
import random
import numpy as np
import math
import csv
import os
import openpyxl

def TestFAKVbatchwisePtrCase1():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 16 1024 12 1 12 128 2048 0 0 1 1 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask1_fp16"
    os.system(run_unittest_cmd)

def TestFAKVbatchwisePtrCase2():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 28 88 32 1 32 128 2048 0 0 1 2 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask1_bf16"
    os.system(run_unittest_cmd)

def TestFAKVbatchwisePtrCase3():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 2 513 8 1 32 128 2048 0 0 2 1 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask2_fp16"
    os.system(run_unittest_cmd)

def TestFAKVbatchwisePtrCase4():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 30 1025 8 1 8 128 2048 0 0 2 2 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask2_bf16"
    os.system(run_unittest_cmd)

def TestFAKVbatchwisePtrCase5():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 128 1367 16 1 16 128 2048 0 0 3 1 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask3_fp16"
    os.system(run_unittest_cmd)

def TestFAKVbatchwisePtrCase6():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 128 1634 8 1 8 128 2048 0 0 4 1 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask4_fp16"
    os.system(run_unittest_cmd)

def TestFAKVbatchwisePtrCase7():
    gen_data_cmd = "python3 unpad_FA_data_gen_renew.py 16 114 32 1 32 128 256 1 1 5 1 0"
    os.system(gen_data_cmd)
    run_unittest_cmd = "mki_ops_unittest --gtest_filter=TestFlashAttentionNd.FlashAttentionTestKVPtr_mask5_fp16"
    os.system(run_unittest_cmd)

if __name__ == "__main__":
    folder_for_kvbatches = "kvBatches"
    folder_path = os.path.join(os.getcwd(), folder_for_kvbatches)
    os.makedirs(folder_path, exist_ok = True)
    TestFAKVbatchwisePtrCase1()
    TestFAKVbatchwisePtrCase2()
    TestFAKVbatchwisePtrCase3()
    TestFAKVbatchwisePtrCase4()
    TestFAKVbatchwisePtrCase5()
    TestFAKVbatchwisePtrCase6()
    TestFAKVbatchwisePtrCase7()