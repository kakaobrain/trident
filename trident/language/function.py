# Copyright 2023 ⓒ Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl

from trident import language


@triton.jit
def batch(index, num_channels, num_rows, num_cols):
    return index // (num_channels * num_rows * num_cols)


@triton.jit
def channel(index, num_channels, num_rows, num_cols):
    return (index % (num_channels * num_rows * num_cols)) // (num_rows * num_cols)


@triton.jit
def col(idx, num_col):
    return idx % num_col


@triton.jit
def distance(x, dim):
    return tl.sqrt(tl.sum(x * x, dim))


@triton.jit
def exp(x):
    if x.dtype is tl.float32 or x.dtype is tl.float64:
        return tl.exp(x)
    else:
        return tl.exp(x.to(tl.float32))


@triton.jit
def make_conv2d_blk(ch_st, w_st, ch_bs, h_bs, w_bs):
    blk = tl.arange(0, w_bs)[:, None] + (tl.arange(0, h_bs) * w_st)[None, :]
    return blk[:, :, None] + (tl.arange(0, ch_bs) * ch_st)[None, None, :]


@triton.jit
def make_conv2d_msk(ch, h, w, ch_bs, h_bs, w_bs):
    msk = (tl.arange(0, w_bs) < w)[:, None] & (tl.arange(0, h_bs) < h)[None, :]
    return msk[:, :, None] & (tl.arange(0, ch_bs) < ch)[None, None, :]


@triton.jit
def make_block(inp_sz, blk_sz, blk_off=0):
    blk = tl.arange(0, blk_sz) + blk_off
    return blk, blk < inp_sz


@triton.jit
def make_group_blk(blk, grp_sz, w):
    return blk[:, None] + (tl.arange(0, grp_sz) * w)[None, :]


@triton.jit
def make_group_msk(blk, grp_sz, off, h):
    return blk[:, None] & (tl.arange(0, grp_sz) + off < h)[None, :]


@triton.jit
def max(
    input_ptr,
    y_size,
    x_size,
    offset,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    dtype: tl.constexpr,
):
    if dim == 0:
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(x_size, y_size),
            strides=(1, x_size),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = y_size
    else:
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = x_size

    output = tl.full((1,), -float("inf"), dtype)
    index = tl.zeros((1,), tl.int32)

    for block_offset in range(0, size_along_dim, block_size):
        input = tl.load(input_block_ptr, boundary_check=(1,))
        condition = tl.arange(0, block_size) + block_offset < size_along_dim
        input = tl.where(condition, input, -float("inf"))
        position = tl.argmax(input, 1) + block_offset
        shift = position * x_size + offset if dim == 0 else position + offset * x_size
        peak = tl.load(input_ptr + shift)

        if peak > output:
            output = peak
            index = position

        input_block_ptr = tl.advance(input_block_ptr, (0, block_size))

    return output, index.to(tl.int64)


@triton.jit
def norm(x, mean, std):
    return (x - mean) / std


@triton.jit
def pow2(x):
    return x * x


@triton.jit
def row(idx, num_row, num_col):
    return (idx % (num_row * num_col)) // num_col


@triton.jit
def sigmoid(x, dtype):
    if dtype is tl.float32 or dtype is tl.float64:
        return tl.sigmoid(x)
    else:
        return tl.sigmoid(x.to(tl.float32))


@triton.jit
def std(var, eps=1e-05):
    return tl.sqrt(var + eps)


@triton.jit
def relu(x):
    return tl.where(x > 0, x, 0)


@triton.jit
def leaky_relu(x, a):
    return tl.where(x > 0, x, 0) + a * tl.where(x > 0, 0, x)
