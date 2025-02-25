import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device('cuda:0')


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def naive_softmax(x):

    x_max = x.max(dim=1)[0] 
    z = x - x_max[:, None] 
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1) 
    ret = numerator / denominator[:, None] 
    return ret

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    # print("row_start: ", row_start)
    row_step = tl.num_programs(0)
    # print("row_step: ", row_step)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols # Indicates which addresses are valid.
        row = tl.load(input_ptrs, mask=mask, other=-float('inf')) # other=-float('inf') For invalid addresses, fill with negative infinity to ensure it does not affect the result when calculating the maximum value in subsequent computations.
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0) # PyTorch Implementation: The tensor is 2D, so we sum along dim=1 (columns). 
                                                  # Triton Kernel Implementation: Each program processes one row (a 1D tensor), so we sum along axis 0 (the only axis).
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0) # PyTorch Implementation: The tensor is 2D, so we sum along dim=1 (columns). 
                                                # Triton Kernel Implementation: Each program processes one row (a 1D tensor), so we sum along axis 0 (the only axis).
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride # The starting address of the current row in the output tensor.
        output_ptrs = output_row_start_ptr + col_offsets # The address of each element in the current row of the output tensor.
        tl.store(output_ptrs, softmax_output, mask=mask) # Writes the results back to memory, using the same mask as loading to ensure only valid data is written back.

# ---- #
# We can create a wrapper function that enqueues the kernel along with its associated (meta-)arguments for any given input tensor.
# ---- #

properties = driver.active.utils.get_device_properties(DEVICE.index)
print(f"Properties: {properties}")
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# Driver program that sets a lot of meta information, such as block size, shared memory allocation, etc. A top-down approach is used to determine the optimal configuration.
def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # To fit an entire row into a single block. This allows us to fit all rows in the GPU’s SRAM, maximizing memory bandwidth and minimizing memory latency.
                                                # We have used a grid of 1 dimension, so we are using dim=0 (row dimension). Then, the width of the 2D matrix dimension (n_cols) is used to calculate the next power of 2.
                                                # Example: input = 2x3 matrix, n_cols = 3, next_power_of_2(3) = 4 -> 2^0 = 1; 2^1 = 2; 2^2 = 4 > 3; 2^3 = 8 > 3. So, BLOCK_SIZE = 4  

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                  num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols) # Since our programs are defined along the row dimension (dim=0), we need to compute the number of elements per row (i.e., the row stride) to properly space the memory accesses.
                                                                                 # We don't need to pass tl.constexpr variable twice, as they are compile-time constants. 
    return y

# ---- #
# Unit Test: 
# A matrix input with irregular numbers of rows and columns -- TO VERIFY --> Padding mechanism works.
# ---- #

torch.manual_seed(0)
x = torch.randn(2, 3, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1) # Apply softmax along axis 1 (aggregate over columns, resulting in row-wise normalization)

