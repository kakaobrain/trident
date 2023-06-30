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


@triton.jit
def batch(index, num_channels, num_rows, num_cols):
    return index // (num_channels * num_rows * num_cols)


@triton.jit
def channel(index, num_channels, num_rows, num_cols):
    return (index % (num_channels * num_rows * num_cols)) // (num_rows * num_cols)


@triton.jit
def col(index, num_cols):
    return index % num_cols


@triton.jit
def exp(x, dtype):
    if dtype is triton.language.float32 or dtype is triton.language.float64:
        return triton.language.exp(x)
    else:
        return triton.language.exp(x.to(triton.language.float32))


@triton.jit
def make_conv2d_blk(ch_st, w_st, ch_bs, h_bs, w_bs):
    blk = triton.language.arange(0, w_bs)[:, None] + (triton.language.arange(0, h_bs) * w_st)[None, :]
    return blk[:, :, None] + (triton.language.arange(0, ch_bs) * ch_st)[None, None, :]


@triton.jit
def make_conv2d_msk(ch, h, w, ch_bs, h_bs, w_bs):
    msk = (triton.language.arange(0, w_bs) < w)[:, None] & (triton.language.arange(0, h_bs) < h)[None, :]
    return msk[:, :, None] & (triton.language.arange(0, ch_bs) < ch)[None, None, :]


@triton.jit
def make_group_blk(blk, grp_sz, w):
    return blk[:, None] + (triton.language.arange(0, grp_sz) * w)[None, :]


@triton.jit
def make_group_msk(blk, grp_sz, off, h):
    return blk[:, None] & (triton.language.arange(0, grp_sz) + off < h)[None, :]


@triton.jit
def max1d(x):
    return triton.language.max(x, 0)


@triton.jit
def max2d(x):
    return triton.language.max(triton.language.ravel(x), 0)


@triton.jit
def pow2(x):
    return x * x


@triton.jit
def cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def row(index, num_rows, num_cols):
    return (index % (num_rows * num_cols)) // num_cols


@triton.jit
def sigmoid(x, dtype):
    if dtype is triton.language.float32 or dtype is triton.language.float64:
        return triton.language.sigmoid(x)
    else:
        return triton.language.sigmoid(x.to(triton.language.float32))


@triton.jit
def sum(x):
    return triton.language.sum(triton.language.ravel(x), 0)


@triton.jit
def var(mask, x, size, mean, dim=0, correction=1):
    return triton.language.sum(pow2(triton.language.where(mask, x - mean, 0.0)), dim) / (size - correction)


@triton.jit
def relu(x):
    return triton.language.where(x > 0, x, 0)


@triton.jit
def leaky_relu(x, a):
    return triton.language.where(x > 0, x, 0) + a * triton.language.where(x > 0, 0, x)
