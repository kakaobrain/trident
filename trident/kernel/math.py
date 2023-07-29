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
def mean(
    x_ptr,
    x_sz,
    blk_sz: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    res = triton.language.zeros([blk_sz], dtype)

    for off in range(0, x_sz, blk_sz):
        blk = triton.language.arange(0, blk_sz) + off
        msk = blk < x_sz

        num = triton.language.load(x_ptr + blk, msk, 0)
        res += num

    return triton.language.sum(res, 0) / x_sz


@triton.jit
def max(inp_ptr, inp_sz, blk_sz: triton.language.constexpr):
    blk, msk = language.make_block(inp_sz, blk_sz, 0)
    inp = triton.language.load(inp_ptr + blk, msk, 0)
    res = triton.language.max(inp, 0)

    for blk_off in range(blk_sz, inp_sz, blk_sz):
        blk, msk = language.make_block(inp_sz, blk_sz, blk_off)
        inp = triton.language.load(inp_ptr + blk, msk, 0)
        num = triton.language.max(inp, 0)
        res = num if num > res else res

    return res


@triton.jit
def sum(
    output_ptr,
    input_ptr,
    height,
    width,
    axis: triton.language.constexpr,
    block_size: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    triton.language.device_assert(axis == 0 or axis == 1)

    pid = triton.language.program_id(0)

    if axis == 0:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(height, width),
            strides=(width, 1),
            offsets=(0, pid),
            block_shape=(block_size, 1),
            order=(0, 1),
        )
        size_along_axis = height
    else:
        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(height, width),
            strides=(width, 1),
            offsets=(pid, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        size_along_axis = width

    accumulation = triton.language.zeros((1,), triton.language.float32)

    for _ in range(0, size_along_axis, block_size):
        input = triton.language.load(
            input_block_ptr, boundary_check=(axis,), padding_option="zero"
        ).to(triton.language.float32)
        accumulation += triton.language.sum(input, axis)
        input_block_ptr = triton.language.advance(
            input_block_ptr, (block_size, 0) if axis == 0 else (0, block_size)
        )

    output_block_ptr = triton.language.make_block_ptr(
        output_ptr,
        shape=(size_along_axis,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )
    output = accumulation.to(dtype)
    triton.language.store(output_block_ptr, output)


@triton.jit
def var(
    x_ptr,
    x_sz,
    mean,
    blk_sz: triton.language.constexpr,
    dtype: triton.language.constexpr,
):
    res = triton.language.zeros([blk_sz], triton.language.float32)

    for blk_off in range(0, x_sz, blk_sz):
        blk, msk = language.make_block(x_sz, blk_sz, blk_off)
        num = triton.language.load(x_ptr + blk, msk, 0).to(triton.language.float32)
        num = triton.language.where(msk, num - mean, 0)
        res += language.pow2(num)

    return (triton.language.sum(res, axis=0) / x_sz).to(dtype)
