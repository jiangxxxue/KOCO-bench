import unittest
import torch
import op_test

OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 2, "inGamma": True, "inBeta": True, "isDynamicQuant": True, "epsilon": 1e-8}

HALF_FLOAT_MIN = -2.0
HALF_FLOAT_MAX = 2.0
QUANTMAX = 127
QUANTMIN = -128

class TestRmsNormDynamicQuant(op_test.OpTest):
    def golden_calc(self, intensors):
        out_shape = intensors[0].shape
        epsilon = self.op_desc["specificParam"]["epsilon"]
        input0 = torch.tensor(intensors[0]).float()
        input1 = torch.tensor(intensors[1]).float()
        input2 = torch.tensor(intensors[2]).float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + epsilon)
        output = (input0 * factor * input1)
        if torch.numel(input2) != 0:
            output = output + input2
        output = output.half()

        scale, _ = torch.max(torch.abs(output), dim=-1, keepdim=True)
        scale = torch.div(torch.tensor([127], dtype=torch.float32), scale)
        output = torch.mul(output, scale)
        out_scale = torch.div(torch.tensor([1], dtype=torch.float32), scale)
        output = torch.clamp(torch.round(output.float()), min=QUANTMIN, max=QUANTMAX)
        return [torch.tensor(output).to(torch.int8), torch.tensor(out_scale).squeeze(-1).to(torch.float32)]


    def golden_compare(self, out_tensors, golden_out_tensors):
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        if torch.any(torch.greater(diff, 1)):
            print("[new standards] output0 accuracy failed")
            return False

        diff = torch.abs(torch.subtract(out_tensors[1], golden_out_tensors[1]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype),
                                   torch.abs(golden_out_tensors[1]))

        if torch.any(torch.greater(diff, 2**(-8) * tensor_max)):
            print("[new standards] output1 accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), 2**(-10))):
            print("[new standards] output1 eb failed")
            return False

        return True

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rmsnormdynamicquant_f16_symmetric_300_32(self):
        '''
            基础场景
            shape: (300, 32) / (300,)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (300, 32)
        shape1 = (32,)
        shape2 = (300)
        input0 = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        OP_PARAM0["isSymmetric"] = True
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape2, dtype=torch.float32),
                      torch.zeros(shape2, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rmsnormdynamicquant_f16_symmetric_401_12288(self):
        '''
            基础场景
            shape: (401, 12288) / (12288,)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (401, 12288)
        shape1 = (12288,)
        shape2 = (401)
        input0 = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        OP_PARAM0["isSymmetric"] = True
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape2, dtype=torch.float32),
                      torch.zeros(shape2, dtype=torch.float32)])

if __name__ == '__main__':
    unittest.main()