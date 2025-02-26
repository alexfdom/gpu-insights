import torch

def square_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.square(x)

compiled_square = torch.compile(square_fn)  # Compile the function

# Now, call the compiled function with a tensor:
result = compiled_square(torch.randn(4, 4, device='cuda'))
print(result)
