#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
import unittest
import torch
import sys
import logging
sys.path.append('../')
import op_test
OP_NAME = "FaUpdateOperation"
faUpdateType = 0

maxUbSize = 188 * 1024
minBlockLength = 3
coreNum = 40
bufferNum = 2

int32_size = 4
float_size = 4
spAligned = 8
hDim = 128

def getMaxTileLength():
    return int((maxUbSize - 8 * int32_size) / (float_size * spAligned * bufferNum * (2 * (1 + hDim) + hDim / spAligned) + hDim * spAligned * float_size))

def getTileLength(total_tileLength):
    blockDim = min((total_tileLength + minBlockLength - 1) / minBlockLength, coreNum)
    formerNum = total_tileLength % blockDim
    tailLength = int(total_tileLength // blockDim)
    formerLength = tailLength + 1 if formerNum else 0
    tileLength = min(getMaxTileLength(), max(formerLength, tailLength))
    return tileLength

class TestDecodeUpdate(op_test.OpTest):
    def golden_calc(self, in_tensors):
        return self.golden_calc_zhV2(in_tensors)

    #老gl gm 留作参考
    def golden_calc_gl_gm(self, in_tensors):
        all_rowmax, all_rowsum, all_out = in_tensors                          
        # (b*s*hc, sp) | (b*s*hc, sp) | (b*s*hc, sp*hd)
        global_max, _ = torch.max(all_rowmax, dim=-1, keepdim=True)               # (b * s * hc, 1)
        all_max_scale = torch.exp(all_rowmax - global_max)                        # (b * s * hc, sp)

        # 更新sum
        all_sum_scaled = all_rowsum * all_max_scale                               # (b * s * hc, sp)
        global_sum = torch.sum(all_sum_scaled, dim=-1, keepdim=True)              # (b * s * hc, 1)

        # 更新out
        all_sum_scale = all_sum_scaled / global_sum

        sp = all_rowmax.shape[-1]
        hd = int(all_out.shape[-1] / sp)

        all_sum_scale = all_sum_scale.unsqueeze(1)
        all_sum_scale = all_sum_scale.repeat(1, hd, 1)
        all_out = all_out.reshape(-1, hd, sp)
        attn_out = torch.sum(all_out * all_sum_scale, dim=-1)
        
        return [attn_out]

    # o = sum(lse_exp/sum(lse_exp)*oi)
    def golden_calc_zhV2(self, in_tensors):
        # old input: (b*s*hc, sp) | (b*s*hc, sp*hd) 
        #  input is: (sp, b*s*hc) | (sp, b*s*hc, hd)
        all_lse, all_out = in_tensors
        sp = all_out.shape[0]
        hd = all_out.shape[-1]

        all_lse = all_lse.transpose(0, 1)
        all_out = all_out.permute(1,2,0).reshape(-1, sp*hd)

        sp_after = all_lse.shape[-1]
        hd_after = int(all_out.shape[-1]/all_lse.shape[-1])    
        
        # (b * s * hc, sp)
        lse_exp = torch.exp(all_lse)
        # (b * s * hc, 1)
        sum_lse_exp = torch.sum(lse_exp, dim=-1, keepdim=True) 
        # (b * s * hc, sp)
        sum_lse_exp = sum_lse_exp.repeat(1, sp)
        lse_exp = lse_exp/sum_lse_exp

        # oi = lse_exp*oi (b * s * hc, hd, sp)*(b*s*hc, hd, sp)
        lse_exp = lse_exp.unsqueeze(1)
        lse_exp = lse_exp.repeat(1, hd, 1)
        all_out = all_out.reshape(-1, hd, sp)
        all_out = all_out * lse_exp

        # o = sum(oi) (b*s*hc, hd)
        all_out = torch.sum(all_out, dim=-1, keepdim=True)
        
        return [all_out]

    # CUDA实现 保精度
    # LSE_final = log(sum(exp(LSE_i)))
    # O_final = sum(exp(LSE_i - LSE_final) * O_i)
    def golden_calc_CUDA(self, in_tensors):
        # (b*s*hc, sp) | (b*s*hc, sp*hd)
        all_lse, all_out = in_tensors
        sp = all_lse.shape[-1]
        hd = int(all_out.shape[-1]/all_lse.shape[-1])                      
        
        # (b * s * hc, sp)
        lse_exp = torch.exp(all_lse)
        # (b * s * hc, 1)
        sum_lse_exp = torch.sum(lse_exp, dim=-1, keepdim=True)
        # (b * s * hc, 1)         
        lse_final = torch.log(sum_lse_exp)

        # (b * s * hc, sp)
        lse = lse - lse_final.repeat(1, sp)

        # (b * s * hc, sp)
        lse = torch.exp(lse)

        # oi = lse_exp*oi (b * s * hc, hd, sp)*(b*s*hc, hd, sp)
        lse = lse.unsqueeze(1)
        lse = lse.repeat(1, hd, 1)
        all_out = all_out.reshape(-1, hd, sp)
        all_out = all_out * lse_exp

        # o = sum(oi) (b*s*hc, hd)
        all_out = torch.sum(all_out, dim=-1, keepdim=True)
        
        return [all_out]

    def golden_compare(self, out_tensors, golden_out_tensors):
        rtol = 1e-5
        atol = 1e-5
        etol = 1e-3
        output = out_tensors[0].to(torch.float32).reshape((-1))
        golden = golden_out_tensors[0].to(torch.float32).reshape((-1))
        if output.dtype == torch.bfloat16:
            rtol = 1e-3
            atol = 1e-3
        elif golden.dtype == torch.float32:
            rtol = 1e-6
            atol = 1e-6
        different_element_results = torch.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
        different_element_indexes = torch.nonzero(different_element_results == False)

        error_ratio = float(different_element_indexes.nelement()) / golden.nelement()
        logging.info("error ratio: %.4f, tolrence: %.4f \n\n\n\n\n" % (error_ratio, etol))
        return error_ratio <= etol

    #   模型常见场景
    def test_fp32_case0_sp2(self):
        logging.info("TEST case0 sp = 2 starting")
        self.set_support_910b_only()
        b = 20
        s = 1
        sp = 2
        hc = int(64 / sp)
        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)
    
        #input0: lse   input2: in
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)
    
        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])

    def test_fp32_case1_sp4(self):
        logging.info("TEST case1 sp = 4 starting")
        self.set_support_910b_only()
        hDim = 512
        b = 32
        s = 1
        sp = 4
        hc = int(128 / sp) #32

        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)

        #input0: lse   input2: in  
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)

        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])

    def test_fp32_case2_sp8(self):
        logging.info("TEST case2 sp = 8 starting")
        self.set_support_910b_only()
        b = 20
        s = 1
        sp = 8
        hc = int(64 / sp)
    
        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)
    
        #input0: lse   input2: in
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)
    
        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])
    
    def test_fp32_case3_largeBs(self):
        logging.info("TEST case3 large BS starting")
        self.set_support_910b_only()
        b = 1000
        s = 1
        sp = 2
        hc = int(64 / sp)
    
        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)
    
        #input0: lse   input2: in
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)
    
        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])
    
    def test_fp32_case4_smallBs(self):
        logging.info("TEST case4 small BS starting")
        self.set_support_910b_only()
        b = 2
        s = 1
        sp = 2
        hc = int(64 / sp)
    
        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)
    
        #input0: lse   input2: in
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)
    
        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])
    
    def test_fp32_case5_Prefill(self):
        logging.info("TEST case5 Prefill starting")
        self.set_support_910b_only()
        b = 2
        s = 4000
        sp = 2
        hc = int(64 / sp)
    
        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)
    
        #input0: lse   input2: in
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)
    
        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])
    
    def test_fp32_case1_sp4(self):
        hDim = 16
        logging.info("TEST case1 sp = 4 starting")
        self.set_support_910b_only()
        b = 2
        s = 150
        sp = 2
        hc = int(64 / sp)
    
        shape0 = (sp, b*s*hc) # sp, b*s*hc   320
        shape2 = (sp, b*s*hc, hDim) # sp, b*s*hc,hdim
        shape4 = (b*s*hc, hDim)
    
        #input0: lse   input2: in
        input0 = torch.rand(shape0, dtype=torch.float32)
        input2 = torch.rand(shape2, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(shape4)
    
        OP_PARAM = {"faUpdateType": faUpdateType, "hDim": hDim, "sp": sp}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input2],
                     [output0])

if __name__ == '__main__':
    unittest.main()
