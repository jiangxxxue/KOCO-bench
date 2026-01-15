import torch
import torch_npu
import threading
import torch_atb

# 定义线程函数
def tensor_computation(thread_id):
    # 在每个线程中创建张量并执行操作
    print(f"Thread {thread_id} started")
    m, k, n = 3, 4, 5
    x = torch.randn(m, k, dtype=torch.float16).npu()
    y = torch.randn(k, n, dtype=torch.float16).npu()

    linear_param = torch_atb.LinearParam(transpose_b=False, has_bias=False)
    linear = torch_atb.Operation(linear_param)
    linear_outputs = linear.forward([x, y])
    print(f"Thread {thread_id} completed computation. linear outputs: {linear_outputs[0]}")

# 创建并启动线程
num_threads = 3  # 可以根据需要调整线程数量
threads = []

for i in range(num_threads):
    thread = threading.Thread(target=tensor_computation, args=(i,))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

print("All threads completed.")
