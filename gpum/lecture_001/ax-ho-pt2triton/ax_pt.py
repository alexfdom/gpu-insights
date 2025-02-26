import torch

def square(a):
    a = torch. square(a)
    return torch. square(a)

opt_square = torch.compile(square)
opt_square(torch.rand(10000, 10000).cuda())