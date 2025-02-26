# Hands-on: lecture_001

Leverage built-in Torch logs to gather Triton kernel data. We can inspect the below code snippet and tweak parameters to gain insights into a reliable Triton kernel.

```python
# TORCH_LOGS=output_code python square_compile.py
import torch

def square(a):
    a = torch. square(a)
    return torch. square(a)

opt_square = torch.compile(square)
opt_square(torch.rand(10000, 10000).cuda())
```