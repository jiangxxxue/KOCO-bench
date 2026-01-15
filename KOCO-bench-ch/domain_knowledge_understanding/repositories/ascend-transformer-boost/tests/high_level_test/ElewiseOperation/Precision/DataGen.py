import os
import torch
import numpy as np
if __name__ == "__main__":
    x = torch.ones(5, dtype=torch.float16)
    y = torch.ones(5, dtype=torch.float16)
    x.numpy().astype(np.float16).tofile('x.bin')
    y.numpy().astype(np.float16).tofile('y.bin')
    (x.numpy() + y.numpy()).astype(np.float16).tofile('out1.bin')