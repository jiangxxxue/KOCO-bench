# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 

import logging
import sys
import subprocess
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


def gen_data(batch, max_seq):

    hiddensize = 1024
    layer = 28
    seqlen = np.random.randint(1, max_seq//2, size=batch, dtype=np.int32)
    # tokenoffset >= seqlen
    tokenoffset = seqlen

    ntokens = np.sum(seqlen)
    logging.info(f"ntokens: {ntokens}")

    newkv = np.random.uniform(-5, 5, size=(ntokens,
                              hiddensize)).astype(np.float16)
    logging.info(f"newKV shape: {newkv.shape}")

    cache_in = np.zeros(shape=(layer, batch, max_seq,
                               hiddensize)).astype(np.float16)

    # GT
    cache_out = np.zeros(shape=(layer, batch, max_seq,
                         hiddensize)).astype(np.float16)

    prefix_ntokens = 0
    for i in range(batch):
        for j in range(seqlen[i]):
            cache_out[0][i][tokenoffset[i]-seqlen[i]+j][:] = newkv[prefix_ntokens+j][:]
        prefix_ntokens += seqlen[i]

    logging.info(f"newkv[0]: {newkv[0]}")
    logging.info(f"cache_out[0][0]: {cache_out[0][0]}")

    layerid = np.array([0], dtype=np.int32)
    logging.info(f"layerid: {layerid}")

    newkv.tofile("./newKV.bin")
    layerid.tofile("./layerid.bin")
    cache_in.tofile("./cacheIn.bin")
    tokenoffset.tofile("./tokenoffset.bin")
    seqlen.tofile("./seqlen.bin")

    cache_out.tofile("./cache_Out.bin")


# ---------------- Do not edit following scripts ----------------


def generate_ut_cases(ut_config):
    case_num = ut_config["case_num"]
    input_dim_series = []
    for t in ut_config["inputs"]:
        input_dim_series.append(
            (np.random.randint(t[0], t[1] + 1,
             case_num) // t[2] * t[2]).tolist()
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
            ret = subprocess.call(
                f"cd {run_path} && bash run.sh -n", shell=False)
            if ret == 0:
                logging.info(f"Unit test case {idx}: {case} succeeded.")
            else:
                logging.info(
                    f"Unit test case {idx}: {case} failed, stop testing now.")
                return


if __name__ == "__main__":
    process(sys.argv)
