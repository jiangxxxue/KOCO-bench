# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import sys
import subprocess
import numpy as np

_ut_config = {
    "case_num": 10,
    "random_seed": 0,
    # Every tuple features an input argument in order, (min, max, round_factor)
    "inputs": [(1, 32, 1), (64, 384, 64)],
    "default_case": (32, 2),
}


def gen_data(batch, max_seq):
    # tensor(batch)
    rotaryCoeff = 4
    headDim = 128 
    # hiddensize = headDim * 32
    hiddensize = 4096
    hiddensizeQ = 4096
    hiddensizeK = 4096    
    
    tmp = np.random.randint(1, 2,size=1,dtype=np.int32)
    tmp[0] = max_seq
    tmp.tofile("./max_seq.bin")
    tmp[0] = batch
    tmp.tofile("./batch.bin")

    tmp[0] = rotaryCoeff

    tmp.tofile("./rotaryCoeff.bin")

    tmp[0] = headDim
    tmp.tofile("./headDim.bin")
    tmp[0] = hiddensize
    tmp.tofile("./hiddensize.bin")

    tmp[0] = hiddensizeQ
    tmp.tofile("./hiddensizeQ.bin")

    tmp[0] = hiddensizeK
    tmp.tofile("./hiddensizeK.bin")


    headNum = hiddensize // headDim
    headNumQ = hiddensizeQ // headDim
    headNumK = hiddensizeK // headDim

    seqlen = np.random.randint(1, max_seq, size=batch, dtype=np.int32)
    np.save("./seqlen.npy", seqlen)
    seqlen.tofile("./seqlen.bin")

    ntokens = np.sum(seqlen)
    logging.info(f"ntokens is {ntokens}")
    ntokens.tofile("./ntokens.bin")

    tmp[0] = ntokens
    tmp.tofile("./ntokens.bin")
    logging.info(ntokens,"----------")

    # q不等 第二个头精度问题 
    q = np.random.uniform(-1, 1, size=(ntokens, hiddensizeQ)).astype(np.float16)
    # for i in range(ntokens):
    #     for j in range(headNum):
    #         src = j * headDim
    #         dst = src + headDim
    #         q[i][src:dst] = q[i][0:headDim]

    q.tofile("./q.bin")
    kk = np.random.uniform(-1, 1, size=(ntokens, hiddensizeK)).astype(np.float16)
    kk.tofile("./k.bin")
    cos = np.random.uniform(-1, 1, size=(ntokens, headDim)).astype(np.float16)
    # 0 ok
    sin = np.random.uniform(-1, 1, size=(ntokens, headDim)).astype(np.float16)
    for i in range(ntokens):
       cos[i][headDim//2:headDim] =  cos[i][0:headDim//2]
       sin[i][headDim//2:headDim] =  sin[i][0:headDim//2]
    cos.tofile("./cos.bin")
    sin.tofile("./sin.bin")
    
    # GT
    rope_q = np.zeros(shape=(ntokens, hiddensizeQ)).astype(np.float16)
    rope_k = np.zeros(shape=(ntokens, hiddensizeK)).astype(np.float16)
    prefix_Ntokens = 0
    cosTable = np.zeros(shape=(ntokens, hiddensize)).astype(np.float16)
    for i in range(ntokens):
        for j in range(headNum):
            # tmp = cosTable[1][i*headDim:(i+1)*headDim]
            # tmp = cosTable[1:3][i*headDim:(i+1)*headDim]
            # print(tmp.shape,cosTable.shape)
            cosTable[i][j*headDim:(j+1)*headDim] = cos[i][:]

    # cos = cos * (-1)

    for i in range(batch):
        curr_seqLen = seqlen[i]


        # q1 = q[prefix_Ntokens:prefix_Ntokens + curr_seqLen] * cosTable[prefix_Ntokens:prefix_Ntokens + curr_seqLen][:hiddensizeQ]
        # k1 = kk[prefix_Ntokens:prefix_Ntokens + curr_seqLen] * cosTable[prefix_Ntokens:prefix_Ntokens + curr_seqLen][:hiddensizeK]
        
        q1 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float16)
        k1 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float16)

        for i in range(prefix_Ntokens, prefix_Ntokens + curr_seqLen):
            q1[i-prefix_Ntokens] = q[i] * cosTable[i][:hiddensizeQ]
            k1[i-prefix_Ntokens] = kk[i] * cosTable[i][:hiddensizeK] 


        
        logging.info(f"q1,k1,{q1.shape},{k1.shape}")

        q2 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float16)
        k2 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float16)        

        for k in range(headNum):
            src_ = k * headDim
            dst_ = (k + 1) * headDim
            strdie = headDim // 2
            rotaryStrdie = headDim // rotaryCoeff
            rotaryTimesPerHead = rotaryCoeff / 2
            
            for cycle in range(int(rotaryTimesPerHead)):
                src =  src_ + cycle * rotaryStrdie * 2
                dst = src + rotaryStrdie * 2

                for curr_seqLeni in range(curr_seqLen):
                    # print("curr_seqLeni ", curr_seqLeni)
                    # print("rotaryStrdie ", rotaryStrdie)
                    # print("prefix_Ntokens ", prefix_Ntokens)
                    # print("rotaryStrdie ", rotaryStrdie)                    
                    #
                    if k < headNumQ:
                        q2[curr_seqLeni][src:src + rotaryStrdie] = q[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                        q2[curr_seqLeni][src + rotaryStrdie:dst] = q[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                        q2[curr_seqLeni][src:dst] = q2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
                    # q2[curr_seqLeni][src + strdie:src + (strdie)+strdie//2] = q[prefix_Ntokens + curr_seqLeni][src+strdie + strdie // 2:dst] * (-1)
                    if k < headNumK:
                        k2[curr_seqLeni][src:src + rotaryStrdie] = kk[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                        k2[curr_seqLeni][src + rotaryStrdie:dst] = kk[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                        k2[curr_seqLeni][src:dst] = k2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
 

 
        rope_q[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += q1 + q2
        rope_k[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += k1 + k2      
        prefix_Ntokens += curr_seqLen
    logging.info("-----------rope q1")
    logging.info(rope_q[0][0:headDim//2])
    logging.info("----------- q")
    logging.info(q[0][headDim//2:headDim])

    logging.info("-----------rope q2")
    logging.info(rope_q[0][headDim//2:headDim])
    logging.info("-----------rope q2")
    logging.info(q[0][0:headDim//2])

    logging.info("-----------rope q11")
    logging.info(rope_q[0][headDim:headDim+headDim//2])
    logging.info("----------- q11")
    logging.info(q[0][headDim+headDim//2:2*headDim])

    logging.info("-----------rope q22")
    logging.info(rope_q[0][headDim//2+headDim:headDim*2])
    logging.info("----------- q22")
    logging.info(q[0][headDim:headDim+headDim//2])
    logging.info("-----------")


    logging.info(rope_k[0])

    logging.info("-----------q")
    
    logging.info(q[0][headDim//2:headDim//2 + 16])
    logging.info(q[0][headDim-16:headDim])
    # logging.info(rope_k[0])
    logging.info("sinnn")
    logging.info(sin[0][:16])
    logging.info(sin[0][headDim//2:headDim//2 + 16])
    logging.info(sin[0][headDim-16:headDim])

    logging.info("sinnn")
    logging.info(cos[0][:16])
    logging.info(cos[0][headDim-16:headDim])


    rope_q.tofile("./rope_q.bin")
    rope_k.tofile("./rope_k.bin")
    # Save 
    




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
            ret = subprocess.call(f"cd {run_path} && bash run.sh -n", shell=True)
            if ret == 0:
                logging.info(f"Unit test case {idx}: {case} succeeded.")
            else:
                logging.info(f"Unit test case {idx}: {case} failed, stop testing now.")
                return


if __name__ == "__main__":
    process(sys.argv)
