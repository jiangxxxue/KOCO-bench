import logging
import unittest
import numpy as np
import torch
import torch_npu
import op_test
from tensor_file import read_tensor
# torch.set_printoptions(threshold=torch.inf)

OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 2, "inBeta": True, "inRes": True, "inGamma": True, "outRes": False, "epsilon": 1e-8,"precisionMode":0}
OP_PARAM1 = {"normType": 2, "inBeta": True, "inRes": True, "inGamma": True, "outRes": False, "epsilon": 1e-8,"precisionMode":1}

class Testpostnorm(op_test.OpTest):
    def golden_calc(self, intensors):
        data_type = intensors[0].dtype
        epsilon = self.op_desc["specificParam"]["epsilon"]
        precisionMode = self.op_desc["specificParam"]["precisionMode"]
        #x, bias, res_in, g
        input0 = torch.tensor(intensors[0]).float()
        input1 = torch.tensor(intensors[1]).float()
        input2 = torch.tensor(intensors[2]).float()
        input3 = torch.tensor(intensors[3]).float()
        if data_type != torch.bfloat16:
            if torch.numel(input1) != 0:
            # inputSum = input0(fp16) + input1(fp32) + input2(fp16)
                inputSum_half = input0 + input1 + input2
            else:
                input_0_half = input0 + input2
                inputSum_half = input_0_half

            inputSum = inputSum_half.float()
            square_sum = torch.sum(torch.square(inputSum), axis=-1, keepdims=True)
            factor = 1.0 / torch.sqrt(square_sum / inputSum.shape[-1] + epsilon)
            output0 = inputSum * factor
            if precisionMode == 1:
                output0_half = output0.half() * intensors[3]
            else:
                output0_half = output0 * intensors[3].float()
            return [torch.tensor(output0_half).to(data_type)]
        else:
            if torch.numel(input1) != 0:
                inputSum = input0 + input1 + input2
            else:
                inputSum = input0 + input2
            
            square_sum = torch.sum(torch.square(inputSum), axis=-1, keepdims=True)
            factor = 1.0 / torch.sqrt(square_sum / inputSum.shape[-1] + epsilon)
            output0 = inputSum * factor * input3
            return [torch.tensor(output0).to(data_type)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        absolute_error = torch.abs(golden_out_tensors[0].float() - out_tensors[0].float())
        relative_error = absolute_error / golden_out_tensors[0].float()
        cnt = torch.sum(torch.gt(relative_error, 0.001)).item()
        return ((cnt / torch.prod(torch.tensor(golden_out_tensors[0].size()))).item() < 0.001)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_f16_case000(self):
        '''
            基础场景
            shape: (100, 6000) / (100, 6000) 
            tilingKey: 000 (short seq, full bias, no bf16)
        '''
        shape0 = (20, 5, 6000)
        shape1 = (1, 1, 6000)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half(), torch.tensor(input2).half(),
                      torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_f16_casesmallest(self):
        '''
            基础场景(极小数据测试)
            shape: (10, 32) / (1, 32) 
            tilingKey: 000 (short seq, full bias, no bf16)
        '''
        shape0 = (2, 5, 32)
        shape1 = (1, 1, 32)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half(), torch.tensor(input2).half(),
                      torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_f16_case010(self):
        '''
            基础场景
            shape: (100, 10240) / (1, 10240) 
            tilingKey: 010 (short seq, empty bias, no bf16)
        '''
        shape0 = (20, 5, 10240)
        shape1 = (1, 1, 10240)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.Tensor().half(), torch.tensor(input2).half(),
                      torch.tensor(input3).half()], [torch.zeros(shape0).half()])
    
    @op_test.only_910b
    def test_postnorm_bf16_case011(self):
        '''
            基础场景
            shape: (100, 12288) / (1, 12288) 
            tilingKey: 011 (short seq, empty bias, bf16)
        '''
        shape0 = (20, 5, 12288)
        shape1 = (1, 1, 12288)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.Tensor().bfloat16(), torch.tensor(input2).bfloat16(),
                      torch.tensor(input3).bfloat16()], [torch.zeros(shape0).bfloat16()])
    
    @op_test.only_910b
    def test_postnorm_bf16_case001(self):
        '''
            基础场景
            shape: (100, 8192) / (1, 8192) 
            tilingKey: 001 (short seq, bias, bf16)
        '''
        shape0 = (20, 5, 8192)
        shape1 = (1, 1, 8192)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16(), torch.tensor(input2).bfloat16(),
                      torch.tensor(input3).bfloat16()], [torch.zeros(shape0).bfloat16()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_f16_larger_case110(self):
        '''
            基础场景
            shape: (100, 15360) / (1, 15360) 
            tilingKey: 110 (long seq, empty bias, no bf16)
        '''
        shape0 = (20, 5, 15360)
        shape1 = (1, 1, 15360)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0,16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0,0)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0,16)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0,16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.Tensor().half(), torch.tensor(input2).half(),
                      torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_f16_larger_case100(self):
        '''
            基础场景
            shape: (100, 15360) / (1, 15360) 
            tilingKey: 100 (long seq, with bias, no bf16)
        '''
        shape0 = (20, 5, 10240)
        shape1 = (1, 1, 10240)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0,16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0,16)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0,16)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0,1)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half(), torch.tensor(input2).half(),
                      torch.tensor(input3).half()], [torch.zeros(shape0).half()])
    
    @op_test.only_910b
    def test_postnorm_bf16_case101(self):
        '''
            基础场景
            shape: (100, 20480) / (1, 20480) 
            tilingKey: 101 (long seq, with bias, bf16)
        '''
        shape0 = (20, 5, 20480)
        shape1 = (1, 1, 20480)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16(), torch.tensor(input2).bfloat16(),
                      torch.tensor(input3).bfloat16()], [torch.zeros(shape0).bfloat16()])
    
    @op_test.only_910b
    def test_postnorm_bf16_case111(self):
        '''
            基础场景
            shape: (100, 15360) / (1, 15360) 
            tilingKey: 111 (long seq, empty bias, bf16)
        '''
        shape0 = (100, 15360)
        shape1 = (1, 15360)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16(), torch.tensor(input2).bfloat16(),
                    torch.tensor(input3).bfloat16()], [torch.zeros(shape0).bfloat16()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_f16_case1(self):

        '''
            基础场景
            shape: (10, 16000) / (1, 16000) 
            tilingKey: 100 (long seq, with bias, fp16)
        '''
        shape0 = (2, 5, 16000)
        shape1 = (1, 1, 16000)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 256)
        input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 256)
        
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half(), torch.tensor(input2).half(),
                      torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.only_910b
    def test_postnorm_b16_caserand_with_bias(self):
        for i in range(5):
            logging.debug("bias and bf16 idx shape is %d", i)
            Shape0 = torch.randint(1,512,(1,))
            randVal = torch.randint(16,20480,(1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            logging.debug("x shape is %s", input0.shape)
            logging.debug("res_in shape is %s", input1.shape)
            logging.debug("bias shape is %s", input2.shape)
            logging.debug("gamma shape is %s", input3.shape)

            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16(), torch.tensor(input2).bfloat16(),
                          torch.tensor(input3).bfloat16()], [torch.zeros(shape0).bfloat16()])
    @op_test.only_910b
    def test_postnorm_b16_caserand_no_bias(self):
        for i in range(5):
            logging.debug(" no bias and bf16 idx shape is %d", i)
            Shape0 = torch.randint(1,512,(1,))
            randVal = torch.randint(16,20480,(1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            logging.debug("x shape is %s", input0.shape)
            logging.debug("res_in shape is %s", input1.shape)
            logging.debug("bias shape is %s", input2.shape)
            logging.debug("gamma shape is %s", input3.shape)

            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([torch.tensor(input0).bfloat16(), torch.Tensor().bfloat16(), torch.tensor(input2).bfloat16(),
                          torch.tensor(input3).bfloat16()], [torch.zeros(shape0).bfloat16()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_fp16__no_bias(self):
        for i in range(5):
            logging.debug(" no bias and fp16 idx shape is %d", i)
            Shape0 = torch.randint(1,512,(1,))
            randVal = torch.randint(16,20480,(1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            logging.debug("x shape is %s", input0.shape)
            logging.debug("res_in shape is %s", input1.shape)
            logging.debug("bias shape is %s", input2.shape)
            logging.debug("gamma shape is %s", input3.shape)

            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([torch.tensor(input0).half(), torch.Tensor().half(), torch.tensor(input2).half(),
                          torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_fp16__no_bias_high_perf(self):
        for i in range(5):
            logging.debug(" no bias and fp16 idx shape is %d", i)
            Shape0 = torch.randint(1,512,(1,))
            randVal = torch.randint(16,20480,(1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            logging.debug("x shape is %s", input0.shape)
            logging.debug("res_in shape is %s", input1.shape)
            logging.debug("bias shape is %s", input2.shape)
            logging.debug("gamma shape is %s", input3.shape)

            self.set_param(OP_NAME, OP_PARAM1)
            self.execute([torch.tensor(input0).half(), torch.Tensor().half(), torch.tensor(input2).half(),
                          torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_fp16_caserand_bias(self):
        for i in range(5):
            logging.debug("bias and fp16 idx shape is %d", i)
            Shape0 = torch.randint(1,512,(1,))
            randVal = torch.randint(16,20480,(1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            logging.debug("x shape is %s", input0.shape)
            logging.debug("res_in shape is %s", input1.shape)
            logging.debug("bias shape is %s", input2.shape)
            logging.debug("gamma shape is %s", input3.shape)

            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([torch.tensor(input0).half(), torch.tensor(input1).half(), torch.tensor(input2).half(),
                          torch.tensor(input3).half()], [torch.zeros(shape0).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_postnorm_fp16_caserand_bias_high_perf(self):
        for i in range(5):
            logging.debug("bias and fp16 idx shape is %d", i)
            Shape0 = torch.randint(1,512,(1,))
            randVal = torch.randint(16,20480,(1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            input2 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
            input3 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
            logging.debug("x shape is %s", input0.shape)
            logging.debug("res_in shape is %s", input1.shape)
            logging.debug("bias shape is %s", input2.shape)
            logging.debug("gamma shape is %s", input3.shape)

            self.set_param(OP_NAME, OP_PARAM1)
            self.execute([torch.tensor(input0).half(), torch.tensor(input1).half(), torch.tensor(input2).half(),
                          torch.tensor(input3).half()], [torch.zeros(shape0).half()])

if __name__ == '__main__':
    unittest.main()