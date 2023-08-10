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

from trident import language


@triton.jit
def batch(index, num_channels, num_rows, num_cols):
    return index // (num_channels * num_rows * num_cols)


@triton.jit
def cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def channel(index, num_channels, num_rows, num_cols):
    return (index % (num_channels * num_rows * num_cols)) // (num_rows * num_cols)


@triton.jit
def col(idx, num_col):
    return idx % num_col


@triton.jit
def diagflat(x, sz):
    vec = triton.language.arange(0, sz)
    dig = vec[:, None] == vec[None, :]
    return dig * triton.language.ravel(x)


@triton.jit
def exp(x):
    if x.dtype is triton.language.float32 or x.dtype is triton.language.float64:
        return triton.language.exp(x)
    else:
        return triton.language.exp(x.to(triton.language.float32))


@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tanh(0.797884560803 * (x + 0.044715 * pow3(x))))


@triton.jit
def gemv(a, x):
    return triton.language.sum(a * triton.language.trans(x[:, None]), 1)


@triton.jit
def make_conv2d_blk(ch_st, w_st, ch_bs, h_bs, w_bs):
    blk = (
        triton.language.arange(0, w_bs)[:, None]
        + (triton.language.arange(0, h_bs) * w_st)[None, :]
    )
    return blk[:, :, None] + (triton.language.arange(0, ch_bs) * ch_st)[None, None, :]


@triton.jit
def make_conv2d_msk(ch, h, w, ch_bs, h_bs, w_bs):
    msk = (triton.language.arange(0, w_bs) < w)[:, None] & (
        triton.language.arange(0, h_bs) < h
    )[None, :]
    return msk[:, :, None] & (triton.language.arange(0, ch_bs) < ch)[None, None, :]


@triton.jit
def make_block(inp_sz, blk_sz, blk_off=0):
    blk = triton.language.arange(0, blk_sz) + blk_off
    return blk, blk < inp_sz


@triton.jit
def make_group_blk(blk, grp_sz, w):
    return blk[:, None] + (triton.language.arange(0, grp_sz) * w)[None, :]


@triton.jit
def make_group_msk(blk, grp_sz, off, h):
    return blk[:, None] & (triton.language.arange(0, grp_sz) + off < h)[None, :]


@triton.jit
def max(
    input_ptr,
    y_size,
    x_size,
    offset,
    dim: triton.language.constexpr,
    block_size: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    if dim == 0:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(x_size, y_size),
            strides=(1, x_size),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = y_size
    else:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = x_size

    output = triton.language.full((1,), -float("inf"), dtype)
    index = triton.language.zeros((1,), triton.language.int32)

    for block_offset in range(0, size_along_dim, block_size):
        input = triton.language.load(input_block_ptr, boundary_check=(1,))
        condition = (
            triton.language.arange(0, block_size) + block_offset < size_along_dim
        )
        input = triton.language.where(condition, input, -float("inf"))
        position = triton.language.argmax(input, 1) + block_offset
        shift = position * x_size + offset if dim == 0 else position + offset * x_size
        peak = triton.language.load(input_ptr + shift)

        if peak > output:
            output = peak
            index = position

        input_block_ptr = triton.language.advance(input_block_ptr, (0, block_size))

    return output, index.to(triton.language.int64)


@triton.jit
def mean(
    input_ptr,
    y_size,
    x_size,
    offset,
    dim: triton.language.constexpr,
    block_size: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    accumulation = sum(input_ptr, y_size, x_size, offset, dim, block_size, dtype)
    size_along_dim = y_size if dim == 0 else x_size
    output = accumulation / size_along_dim

    return output.to(dtype)


@triton.jit
def norm(x, mean, std):
    return (x - mean) / std


@triton.jit
def pow2(x):
    return x * x


@triton.jit
def pow3(x):
    return x * x * x


@triton.jit
def row(idx, num_row, num_col):
    return (idx % (num_row * num_col)) // num_col


@triton.jit
def sigmoid(x, dtype):
    if dtype is triton.language.float32 or dtype is triton.language.float64:
        return triton.language.sigmoid(x)
    else:
        return triton.language.sigmoid(x.to(triton.language.float32))


@triton.jit
def sum(
    input_ptr,
    y_size,
    x_size,
    offset,
    dim: triton.language.constexpr,
    block_size: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    if dim == 0:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(0, offset),
            block_shape=(block_size, 1),
            order=(0, 1),
        )
        size_along_dim = y_size
        accumulation = triton.language.zeros((block_size, 1), triton.language.float32)
    else:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = x_size
        accumulation = triton.language.zeros((1, block_size), triton.language.float32)

    for _ in range(0, size_along_dim, block_size):
        input = triton.language.load(
            input_block_ptr, boundary_check=(dim,), padding_option="zero"
        ).to(triton.language.float32)
        accumulation += input
        input_block_ptr = triton.language.advance(
            input_block_ptr, (block_size, 0) if dim == 0 else (0, block_size)
        )

    output = triton.language.sum(accumulation, dim)

    return output.to(dtype)


@triton.jit
def std(var, eps=1e-05):
    return triton.language.sqrt(var + eps)


@triton.jit
def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


@triton.jit
def fast_var(
    input_ptr,
    y_size,
    x_size,
    offset,
    dim: triton.language.constexpr,
    correction: triton.language.constexpr,
    block_size: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    if dim == 0:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(x_size, y_size),
            strides=(1, x_size),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = y_size
    else:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = x_size

    count = triton.language.zeros((1, block_size), triton.language.float32)
    mean = triton.language.zeros((1, block_size), triton.language.float32)
    m2 = triton.language.zeros((1, block_size), triton.language.float32)

    for block_offset in range(0, size_along_dim, block_size):
        input = triton.language.load(input_block_ptr, boundary_check=(1,))
        mask = (triton.language.arange(0, block_size) + block_offset) < size_along_dim
        input = triton.language.where(mask, input, 0.0)
        count += 1
        m2 += (input - mean) * (input - (mean + (input - mean) / count))
        mean += (input - mean) / count
        input_block_ptr = triton.language.advance(input_block_ptr, (0, block_size))

    output, _, _ = triton.language.reduce(
        (m2, mean, count), 1, language.combine_welford
    )
    output /= size_along_dim - correction

    return output.to(dtype)


@triton.jit
def var(
    input_ptr,
    y_size,
    x_size,
    offset,
    mean,
    dim: triton.language.constexpr,
    correction: triton.language.constexpr,
    block_size: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    if dim == 0:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(x_size, y_size),
            strides=(1, x_size),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = y_size
    else:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_dim = x_size

    accumulation = triton.language.zeros((1, block_size), triton.language.float32)

    for block_offset in range(0, size_along_dim, block_size):
        input = triton.language.load(input_block_ptr, boundary_check=(1,))
        mask = (triton.language.arange(0, block_size) + block_offset) < size_along_dim
        input = triton.language.where(mask, input - mean, 0.0)
        accumulation += pow2(input)
        input_block_ptr = triton.language.advance(input_block_ptr, (0, block_size))

    output = triton.language.sum(accumulation, 1)
    output /= size_along_dim - correction

    return output.to(dtype)


@triton.jit
def relu(x):
    return triton.language.where(x > 0, x, 0)


@triton.jit
def leaky_relu(x, a):
    return triton.language.where(x > 0, x, 0) + a * triton.language.where(x > 0, 0, x)
