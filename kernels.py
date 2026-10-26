"""Edit here, then run the cute or symmetric experiment. GPU imports only."""

import cutlass
import cutlass.cute as cute
import nccl.core.device.cute as nccl_cute
from cuda.bindings.driver import CUstream


@cute.kernel
def add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    i = block * 128 + thread
    if i < out.shape[0]:
        out[i] = a[i] + b[i]


@cute.jit
def add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, stream: CUstream):
    add_kernel(a, b, out).launch(
        grid=[cute.ceil_div(out.shape[0], 128), 1, 1],
        block=[128, 1, 1], stream=stream,
    )


# Registered-window probe using the documented NCCL4Py ReduceCopy API:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html
# Inputs are host-synchronized. This does NOT test a device ready-flag protocol.
@cute.kernel
def lsa_sum_kernel(
    src: nccl_cute.Window, dst: nccl_cute.Window,
    rank: cutlass.Int32, world: cutlass.Int32,
):
    layout = cute.make_layout(16)
    team = nccl_cute.Team(nRanks=world, rank=rank, stride=cutlass.Int32(1))
    nccl_cute.lsa_reduce_sum(
        nccl_cute.cta(), src, src.tensor(cutlass.Float32, layout),
        dst.tensor(cutlass.Float32, layout), 16, team=team,
    )


@cute.jit
def lsa_sum(
    src: nccl_cute.Window, dst: nccl_cute.Window,
    rank: cutlass.Int32, world: cutlass.Int32, stream: CUstream,
):
    lsa_sum_kernel(src, dst, rank, world).launch(
        grid=[1, 1, 1], block=[32, 1, 1], cooperative=True, stream=stream,
    )
