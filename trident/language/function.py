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
def pow2(x):
    return x * x


@triton.jit
def row(idx, num_row, num_col):
    return (idx % (num_row * num_col)) // num_col


@triton.jit
def std(var, eps=1e-05):
    return tl.sqrt(var + eps)
