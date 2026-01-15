import logging
import unittest
import numpy as np
import torch
import op_test
from tensor_file import read_tensor


OP_NAME = "OnehotOperation"

def infershape(axis1, depth, shape):
    indices_index = 0
    new_shape = []
    if axis1 >= 0 :
        for i in range(len(shape) + 1):
            if i == axis1 :  
                new_shape.push_back(depth[0])
            else :
                new_shape.append(shape[indices_index])
                indices_index = indices_index + 1  
    else:
        for i in range(len(shape) + 1):   
            if (i == len(shape) + 1 + axis1) :
                new_shape.append(depth[0])
            else:
                new_shape.append(shape[indices_index]) 
                indices_index = indices_index + 1  
    return new_shape

class Testonehot(op_test.OpTest):
    def golden_calc(self, intensors):

        data_type = intensors[0].dtype
        depth = self.op_desc["specificParam"]["depth"]

        input0 = torch.tensor(intensors[0]).numpy()

        res = np.eye(depth[0])[input0]
        res = torch.from_numpy(res).to(data_type)
        return [res]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001)

    @op_test.skip_310b
    def test_onehot_int64_case0(self):
        depth = 20
        OP_PARAM0 = {"axis": -1, "depth": [depth]}
        shape0 = (16,)
        shape1 = (1,)
        new_shape = infershape(OP_PARAM0["axis"], OP_PARAM0["depth"], shape0)
        input0 = np.random.randint(depth, size=shape0).astype(np.int64)
        input1 = np.ones(shape=shape1, dtype=np.int64)
        input2 = np.zeros(shape=shape1, dtype=np.int64)
        
        input0_torch = torch.from_numpy(input0)
        input1_torch = torch.from_numpy(input1)
        input2_torch = torch.from_numpy(input2)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0_torch, input1_torch, input2_torch],
                    [torch.zeros(new_shape).long()])

    @op_test.skip_310b
    def test_onehot_int32_case0(self):
        depth = 10
        OP_PARAM0 = {"axis": -1, "depth": [depth]}
        shape0 = (16,)
        shape1 = (1,)
        new_shape = infershape(OP_PARAM0["axis"], OP_PARAM0["depth"], shape0)
        input0 = np.random.randint(depth, size=shape0).astype(np.int32)
        input1 = np.ones(shape1, dtype=np.int32)
        input2 = np.zeros(shape1, dtype=np.int32)

        input0_torch = torch.from_numpy(input0)
        input1_torch = torch.from_numpy(input1)
        input2_torch = torch.from_numpy(input2)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0_torch, input1_torch, input2_torch],
                    [torch.zeros(new_shape).int()])


if __name__ == '__main__':
    unittest.main()