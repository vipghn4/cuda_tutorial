import torch
import torch.nn as nn
import conv_cuda
import time

def test_forward():
    x = torch.randint(low=-10, high=10, size=(2, 3, 8, 16)).float().cuda(1)
    W = torch.randint(low=-10, high=10, size=(8, 3, 3, 3)).float().cuda(1)
    b = torch.randint(low=-10, high=10, size=(8, )).float().cuda(1)
    
    conv = nn.Conv2d(3, 8, 3).cuda(1)
    conv.weight.data = W
    conv.bias.data = b
    
    y1 = conv_cuda.forward(x, W, b)[0]
    y2 = conv(x)
    diff = torch.sum(torch.abs(y1 - y2))
    print(diff)

def test_backward():
    x = torch.randint(low=-10, high=10, size=(2, 3, 8, 16)).float().cuda(1)
    W = torch.randint(low=-10, high=10, size=(8, 3, 3, 3)).float().cuda(1)
    b = torch.randint(low=-10, high=10, size=(8, )).float().cuda(1)
    
    conv = nn.Conv2d(3, 8, 3).cuda(1)
    conv.weight.data = W
    conv.bias.data = b
    
    y1 = conv_cuda.forward(x, W, b)[0]
    
    grad_output = torch.ones_like(y1)
    grad_input, grad_W, grad_b = conv_cuda.backward(grad_output, x, W, b)

if __name__ == "__main__":
    device_idx = 0
    x = torch.randint(low=-10, high=10, size=(2, 3, 8, 16)).float().cuda(device_idx)
    x.requires_grad = True
    W = torch.randint(low=-10, high=10, size=(8, 3, 3, 3)).float().cuda(device_idx)
    b = torch.randint(low=-10, high=10, size=(8, )).float().cuda(device_idx)
    
    conv = nn.Conv2d(3, 8, 3).cuda(device_idx)
    conv.weight.data = W
    conv.bias.data = b
    
    start = time.time()
    y1 = conv_cuda.forward(x, W, b)[0]
    time1 = time.time() - start
    
    start = time.time()
    y2 = conv(x)
    time2 = time.time() - start
    print(f"Forward: Hand-code time {time1:.4f} - Torch time {time2:.4f}")
    
    grad_output = torch.ones_like(y1)
    
    start = time.time()
    grad_input1, grad_W1, grad_b1 = conv_cuda.backward(grad_output, x, W, b)
    time1 = time.time() - start
    
    start = time.time()
    grad_input2, grad_W2, grad_b2 = torch.autograd.grad([y2], [x, conv.weight, conv.bias], grad_output)
    time2 = time.time() - start
    print(f"Backward: Hand-code time {time1:.4f} - Torch time {time2:.4f}")
    
    diffInput = torch.sum(torch.abs(grad_input1 - grad_input2))
    diffW = torch.sum(torch.abs(grad_W1 - grad_W2))
    diffb = torch.sum(torch.abs(grad_b1 - grad_b2))
    print(f"Differences by input {diffInput} by weights {diffW} and by bias {diffb}")