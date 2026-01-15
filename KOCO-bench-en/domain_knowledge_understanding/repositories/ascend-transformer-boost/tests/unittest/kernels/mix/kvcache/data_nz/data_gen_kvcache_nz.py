#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 2023-2023 Huawei Technologies Co., Ltd
import logging
import sys
import subprocess
import math
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
_ut_config = {
    "case_num": 10,
    "random_seed": 0,
    # Every tuple features an input argument in order, (min, max, round_factor)
    "inputs": [(1, 32, 1), (64, 384, 64)],
    "default_case": (16, 384),
}

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3])
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)

def gen_data(batch, max_seq):

    hiddensize = 2048
    layer = 1
    layer_id = 0
    seqlen = np.random.randint(64, 65, size=batch, dtype=np.int32)
    tokenoffset = np.zeros(shape=(batch)).astype(np.int32)
    for i in range(batch):
        tokenoffset[i] = seqlen[i] + 4
    ntokens = np.sum(seqlen)
    newkv = np.random.uniform(-5, 5, size=(ntokens, hiddensize)).astype(np.float16)
    ntokens_ceil = (ntokens + 15) // 16 * 16
    newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hiddensize)).astype(np.float16)
    for i in range(ntokens, ntokens_ceil):
        for j in range(hiddensize):
            newkv[i][j] = 0
    cache_in = np.zeros(shape=(layer, batch, max_seq, hiddensize)).astype(np.float16)
    cache_out = np.zeros(shape=(layer, batch, max_seq, hiddensize)).astype(np.float16)

    prefix_ntokens = 0
    for i in range(batch):
        for j in range(seqlen[i]):
            cache_out[layer_id][i][tokenoffset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
        prefix_ntokens += seqlen[i]
    layerid = np.array([layer_id], dtype=np.int32)

    newkv = convert_nd_to_nz(newkv)
    newkv = newkv.reshape([-1, 16, 16])
    newkv.tofile("./newKV.bin")
    layerid.tofile("./layerid.bin")
    cache_in = convert_nd_to_nz(cache_in)
    cache_in = cache_in.reshape([-1, 16, 16])
    cache_in.tofile("./cacheIn.bin")
    tokenoffset.tofile("./tokenoffset.bin")
    seqlen.tofile("./seqlen.bin")
    cache_out = convert_nd_to_nz(cache_out)
    cache_out = cache_out.reshape([-1, 16, 16])
    cache_out.tofile("./cache_Out.bin")
# ---------------- Do not edit following scripts ----------------

def generate_ut_cases(ut_config):
    case_num = ut_config["case_num"]
    input_dim_series = []
    for t in ut_config["inputs"]:
        input_dim_series.append(
            (np.random.randint(t[0], t[1] + 1, case_num) // t[2] * t[2]).tolist()
        )
    return [tuple(x) for x in zip(*input_dim_series)]


def process(argv):
    np.random.seed(_ut_config["random_seed"])
    if len(sys.argv) == 1:
        gen_data(*_ut_config["default_case"])
    else:
        cases = generate_ut_cases(_ut_config)
        run_path = "/".join(sys.path[0].split('/')[:-2])
        for idx, case in enumerate(cases):
            gen_data(*case)
            ret = subprocess.call(f"cd {run_path} && bash run.sh -n", shell=False)
            if ret == 0:
                logging.info(f"Unit test case {idx}: {case} succeeded.")
            else:
                logging.info(f"Unit test case {idx}: {case} failed, stop testing now.")
                return


if __name__ == "__main__":
    process(sys.argv)
